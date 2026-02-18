"""
src/sae_analysis.py

Sparse Autoencoder (SAE) feature discovery for humor recognition.
Supports GPT-2 Small (local/MPS) and Gemma-2-2B (Modal/A100).

Architecture notes
──────────────────
• For Gemma-Scope SAEs the config object is a JumpReLUSAEConfig whose
  hook-point attribute is ``hook_point`` (NOT ``hook_name``).
• The TransformerLens cache keys follow the pattern
  ``blocks.<layer>.hook_resid_post`` for residual-stream SAEs, so we
  derive the hook name deterministically from the layer index rather
  than relying on any single SAE-config field name.
• Gemma-2 uses logit soft-capping, so ``center_unembed`` must be False.
• Gemma-2 uses RMSNorm (no bias), so ``center_writing_weights`` must be
  False to avoid the "can't center" warning.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Optional imports (fail gracefully for non-GPU environments)
# ---------------------------------------------------------------------------
try:
    from sae_lens import SAE
    from transformer_lens import HookedTransformer
except ImportError:
    SAE = None  # type: ignore[assignment,misc]
    HookedTransformer = None  # type: ignore[assignment,misc]
    warnings.warn(
        "sae-lens / transformer-lens not installed. "
        "Install with: pip install sae-lens transformer-lens",
        stacklevel=2,
    )

# ---------------------------------------------------------------------------
# Model / SAE configuration registry
# ---------------------------------------------------------------------------
# ``hook_name`` is the TransformerLens cache key for the residual stream
# at the specified layer.  We set it explicitly so the rest of the code
# never has to guess.

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "gpt2": {
        "model_name": "gpt2",
        "sae_release": "gpt2-small-res-jb",
        "sae_id": "blocks.7.hook_resid_post",
        "hook_name": "blocks.7.hook_resid_post",
        "layer": 7,
        "dtype": "float32",
    },
    "gemma-2-2b": {
        "model_name": "gemma-2-2b",
        "sae_release": "gemma-scope-2b-pt-res",
        "sae_id": "layer_15/width_16k/average_l0_78",
        "hook_name": "blocks.15.hook_resid_post",  # ← deterministic
        "layer": 15,
        "dtype": "bfloat16",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Model + SAE loader
# ═══════════════════════════════════════════════════════════════════════════

def load_sae_model(
    model_alias: str = "gemma-2-2b",
    device: Optional[str] = None,
):
    """Load a HookedTransformer and its corresponding SAE.

    Returns
    -------
    model : HookedTransformer
    sae   : SAE
    cfg   : dict   – the entry from ``MODEL_CONFIGS``
    """
    if SAE is None or HookedTransformer is None:
        raise ImportError("sae-lens and transformer-lens are required.")

    if model_alias not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_alias!r}. "
            f"Choose from {list(MODEL_CONFIGS)}"
        )

    cfg = MODEL_CONFIGS[model_alias]

    # ── Device selection ──────────────────────────────────────────────
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # ── Dtype selection ───────────────────────────────────────────────
    use_bf16 = (
        cfg["dtype"] == "bfloat16"
        and device == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    print(f"[load] model={cfg['model_name']}  device={device}  dtype={dtype}")

    # ── Load transformer ──────────────────────────────────────────────
    model = HookedTransformer.from_pretrained(
        cfg["model_name"],
        device=device,
        dtype=dtype,
        center_unembed=False,           # required: Gemma soft-cap
        center_writing_weights=False,   # required: RMSNorm
        default_padding_side="right",
    )
    print(f"[load] ✓ {cfg['model_name']} loaded ({model.cfg.n_layers} layers)")

    # ── Load SAE ──────────────────────────────────────────────────────
    print(f"[load] SAE release={cfg['sae_release']}  id={cfg['sae_id']}")
    sae, _, _ = SAE.from_pretrained(
        release=cfg["sae_release"],
        sae_id=cfg["sae_id"],
        device=device,
    )

    # Match dtype with the model for mixed-precision matmuls
    if use_bf16:
        sae = sae.to(dtype=torch.bfloat16)

    print(f"[load] ✓ SAE loaded  (d_sae={sae.cfg.d_sae})")
    return model, sae, cfg


# ═══════════════════════════════════════════════════════════════════════════
# Feature extraction pipeline
# ═══════════════════════════════════════════════════════════════════════════

def get_humor_features(
    model_alias: str = "gemma-2-2b",
    data_path: Union[str, Path] = "datasets/dataset_a_paired.xlsx",
    batch_size: int = 4,
    save_dir: Union[str, Path] = "results",
) -> List[Dict[str, Any]]:
    """Extract and rank SAE features that differentiate humor from non-humor.

    Pipeline
    --------
    1. Load model + SAE.
    2. Read dataset, split by ``humor`` label.
    3. For each text, run the model up to the target layer, encode the
       residual-stream activations with the SAE, and keep the *max*
       activation per feature across the sequence.
    4. Compute ``mean_joke − mean_non_joke`` per feature.
    5. Report the top-20 features plus a logit-lens interpretation.

    Parameters
    ----------
    model_alias : str
        ``"gpt2"`` or ``"gemma-2-2b"``.
    data_path : str | Path
        Path to the paired XLSX dataset (needs ``text`` and ``humor`` cols).
    batch_size : int
        Token batch size — use 4 for local/MPS, ≥128 for A100.
    save_dir : str | Path
        Directory to write the JSON results file.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────
    model, sae, cfg = load_sae_model(model_alias)
    hook_name: str = cfg["hook_name"]
    layer: int = cfg["layer"]

    # ── 2. Data (HuggingFace ColBERT) ────────────────────────────────
    print(f"[data] Loading CreativeLang/ColBERT_Humor_Detection from HuggingFace ...")
    try:
        from datasets import load_dataset
        
        dataset = load_dataset("CreativeLang/ColBERT_Humor_Detection")
        data = dataset['train']
        
        # Filter jokes (humor=True) and non-jokes (humor=False)
        jokes = [item['text'] for item in data if item['humor'] == True][:1000]
        non_jokes = [item['text'] for item in data if item['humor'] == False][:1000]
        
        print(f"[data] {len(jokes)} jokes  vs  {len(non_jokes)} non-jokes")
        
        if len(jokes) == 0 or len(non_jokes) == 0:
            raise ValueError(f"Not enough data! jokes={len(jokes)}, non_jokes={len(non_jokes)}")
            
    except Exception as exc:
        raise RuntimeError(f"Cannot load HuggingFace dataset: {exc}") from exc

    # ── 3. Feature extraction ─────────────────────────────────────────
    def _extract(texts: List[str], desc: str = "Extracting") -> torch.Tensor:
        """Return (N, d_sae) tensor of max SAE activations per text."""
        all_maxes: List[torch.Tensor] = []

        for start in tqdm(range(0, len(texts), batch_size), desc=desc):
            batch = texts[start : start + batch_size]
            tokens = model.to_tokens(batch, prepend_bos=True)

            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=[hook_name],
                    stop_at_layer=layer + 1,
                )
                # (batch, seq, d_model)
                residuals = cache[hook_name]

                # (batch, seq, d_sae)
                feature_acts = sae.encode(residuals)

                # Max activation per feature across the sequence dim
                max_acts = feature_acts.max(dim=1).values  # (batch, d_sae)
                all_maxes.append(max_acts.cpu())

        return torch.cat(all_maxes, dim=0)

    print(f"[extract] batch_size={batch_size}")
    joke_acts = _extract(jokes, desc="Jokes")
    nonjoke_acts = _extract(non_jokes, desc="Non-jokes")

    # ── 4. Feature ranking ────────────────────────────────────────────
    print("[rank] Computing mean-difference ranking ...")
    mean_joke = joke_acts.float().mean(dim=0)
    mean_nonjoke = nonjoke_acts.float().mean(dim=0)
    diff = mean_joke - mean_nonjoke

    top_k = 20
    top_vals, top_idxs = torch.topk(diff, top_k)

    # ── 5. Logit-lens interpretation ──────────────────────────────────
    W_U = model.W_U  # (d_model, vocab)
    results: List[Dict[str, Any]] = []

    print(f"\n{'─'*60}")
    print(f"  Top-{top_k} humor features  ({model_alias})")
    print(f"{'─'*60}")

    for rank, (idx_t, val_t) in enumerate(zip(top_idxs, top_vals)):
        idx = idx_t.item()
        score = val_t.item()

        # Decoder direction → logit space
        feat_dir = sae.W_dec[idx]
        if feat_dir.dtype != W_U.dtype:
            feat_dir = feat_dir.to(W_U.dtype)

        logits = feat_dir @ W_U
        top_token_ids = torch.topk(logits, 5).indices
        top_tokens = model.to_string(top_token_ids)

        print(
            f"  #{rank+1:2d}  Feature {idx:5d}  "
            f"Δ={score:+.4f}  "
            f"tokens: {top_tokens}"
        )
        results.append({
            "rank": rank + 1,
            "feature_idx": idx,
            "diff": round(score, 6),
            "top_tokens": top_tokens,
            "joke_mean": round(mean_joke[idx].item(), 6),
            "nonjoke_mean": round(mean_nonjoke[idx].item(), 6),
        })

    print(f"{'─'*60}\n")

    # ── 6. Save ───────────────────────────────────────────────────────
    out_path = save_dir / f"{model_alias}_sae_features.json"
    with open(out_path, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"[save] ✅ {out_path}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAE humor-feature discovery")
    parser.add_argument(
        "--model", default="gpt2", choices=list(MODEL_CONFIGS),
        help="Model alias to use",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--data", default="datasets/dataset_a_paired.xlsx")
    parser.add_argument("--save-dir", default="results")
    args = parser.parse_args()

    get_humor_features(
        args.model, args.data, args.batch_size, args.save_dir,
    )