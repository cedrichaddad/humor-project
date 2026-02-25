"""
src/sae_analysis.py

Sparse Autoencoder (SAE) feature discovery for humor recognition.
Gemma-2-2B only, sweeping layers 15-20 via GemmaScope canonical SAEs.

Architecture notes
──────────────────
• Uses gemma-scope-2b-pt-res-canonical — works for any layer without
  hardcoding an average_l0 value.
• Hook keys follow the pattern blocks.<layer>.hook_resid_post.
• Gemma-2 uses logit soft-capping → center_unembed=False.
• Gemma-2 uses RMSNorm (no bias) → center_writing_weights=False.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from tqdm import tqdm

try:
    from sae_lens import SAE
    from transformer_lens import HookedTransformer
except ImportError:
    SAE = None            # type: ignore[assignment,misc]
    HookedTransformer = None  # type: ignore[assignment,misc]
    warnings.warn(
        "sae-lens / transformer-lens not installed. "
        "Install with: pip install sae-lens transformer-lens",
        stacklevel=2,
    )

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

_GEMMA_CANONICAL_RELEASE = "gemma-scope-2b-pt-res-canonical"
SWEEP_LAYERS = [15, 16, 17, 18, 19, 20]

# ═══════════════════════════════════════════════════════════════════════════
# Artifact filter
# ═══════════════════════════════════════════════════════════════════════════
# The SAE was pretrained on a broad corpus (The Pile / C4 / etc.) and learned
# features for code, Arabic, Russian, Malay, CJK, etc. Some fire
# differentially on jokes vs non-jokes purely by chance. We filter them by
# inspecting each feature's top decoded tokens before ranking.

_CODE_RE = re.compile(
    r'[a-z][A-Z][a-zA-Z]{2,}'                       # camelCase
    r'|[A-Z]{2,}[a-z][A-Z]'                         # MixedUpper
    r'|\b(def|SELECT|FROM|WHERE|INSERT'
    r'|import|function|var|const|return|printf'
    r'|cout|endl|class|void|int|str)\b'              # code keywords
    r'|[{}\[\]<>]{2,}'                               # bracket clusters
    r'|;\s*$',                                        # line-ending semicolons
    re.MULTILINE,
)


def _is_artifact_feature(top_tokens: List[str]) -> bool:
    """Return True if the feature should be excluded (code / foreign script)."""
    for tok in top_tokens:
        if any(ord(c) > 127 for c in tok):
            return True
        if _CODE_RE.search(tok):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Transformer loader
# ═══════════════════════════════════════════════════════════════════════════

def load_sae_model(
    model_alias: str = "gemma-2-2b",
    device: Optional[str] = None,
):
    """Load the HookedTransformer (Gemma-2-2B).

    The SAE is loaded separately per layer in get_humor_features_for_layer
    so we never need to keep more than one SAE in memory at a time.

    Returns
    -------
    model : HookedTransformer
    None  : (placeholder — callers that destructure 3 values still work)
    None  : (placeholder)
    """
    if HookedTransformer is None:
        raise ImportError("transformer-lens is required.")

    if model_alias != "gemma-2-2b":
        raise ValueError(f"Only gemma-2-2b is supported, got {model_alias!r}")

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    dtype    = torch.bfloat16 if use_bf16 else torch.float32
    print(f"[load] model=gemma-2-2b  device={device}  dtype={dtype}")

    model = HookedTransformer.from_pretrained(
        "gemma-2-2b",
        device=device,
        dtype=dtype,
        center_unembed=False,
        center_writing_weights=False,
        default_padding_side="right",
    )
    print(f"[load] ✓ gemma-2-2b loaded ({model.cfg.n_layers} layers)")
    return model, None, None


# ═══════════════════════════════════════════════════════════════════════════
# Feature extraction pipeline
# ═══════════════════════════════════════════════════════════════════════════

def get_humor_features_for_layer(
    layer: int,
    model=None,
    batch_size: int = 128,
    save_dir: Union[str, Path] = "results",
) -> List[Dict[str, Any]]:
    """Extract and rank SAE features that differentiate humor from non-humor
    at a given Gemma-2-2B layer.

    Pipeline
    --------
    1. Load the canonical GemmaScope SAE for this layer.
    2. Load ColBERT dataset (1000 jokes + 1000 non-jokes).
    3. Run the model to the target layer, encode residuals with the SAE,
       keep max activation per feature across the sequence.
    4. Compute mean_joke - mean_nonjoke per feature.
    5. Oversample top-200 by diff, apply artifact filter, keep top-20 clean.
    6. Save results to save_dir/gemma-2-2b_sae_features.json.

    Parameters
    ----------
    layer      : Gemma-2-2B layer index (15-20).
    model      : Pre-loaded HookedTransformer. Loaded fresh if None.
    batch_size : Token batch size (4 for local/MPS, 128 for A100).
    save_dir   : Directory to write JSON results.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if model is None:
        model, _, _ = load_sae_model("gemma-2-2b")

    device   = next(model.parameters()).device
    use_bf16 = model.cfg.dtype == torch.bfloat16

    hook_name = f"blocks.{layer}.hook_resid_post"

    print(f"\n[load] SAE layer={layer}  id=layer_{layer}/width_16k/canonical")
    sae, _, _ = SAE.from_pretrained(
        release=_GEMMA_CANONICAL_RELEASE,
        sae_id=f"layer_{layer}/width_16k/canonical",
        device=device,
    )
    if use_bf16:
        sae = sae.to(dtype=torch.bfloat16)
    print(f"[load] ✓ SAE loaded  (d_sae={sae.cfg.d_sae})")

    # ── Data ─────────────────────────────────────────────────────────
    from datasets import load_dataset
    print("[data] Loading ColBERT...")
    data      = load_dataset("CreativeLang/ColBERT_Humor_Detection")["train"]
    jokes     = [item["text"] for item in data if item["humor"] == True][:1000]
    non_jokes = [item["text"] for item in data if item["humor"] == False][:1000]
    print(f"[data] {len(jokes)} jokes  vs  {len(non_jokes)} non-jokes")

    # ── Extraction ───────────────────────────────────────────────────
    def _extract(texts, desc):
        all_maxes = []
        for start in tqdm(range(0, len(texts), batch_size), desc=desc):
            batch  = texts[start : start + batch_size]
            tokens = model.to_tokens(batch, prepend_bos=True)
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=[hook_name],
                    stop_at_layer=layer + 1,
                )
                feature_acts = sae.encode(cache[hook_name])
                all_maxes.append(feature_acts.max(dim=1).values.cpu())
        return torch.cat(all_maxes, dim=0)

    joke_acts    = _extract(jokes,     desc=f"L{layer} jokes")
    nonjoke_acts = _extract(non_jokes, desc=f"L{layer} non-jokes")

    # ── Ranking ──────────────────────────────────────────────────────
    mean_joke    = joke_acts.float().mean(dim=0)
    mean_nonjoke = nonjoke_acts.float().mean(dim=0)
    diff         = mean_joke - mean_nonjoke

    _, top_idxs = torch.topk(diff, 200)
    top_vals    = diff[top_idxs]

    # ── Logit lens + artifact filter ─────────────────────────────────
    W_U     = model.W_U
    results = []
    skipped = 0

    print(f"\n{'─'*60}")
    print(f"  Top-20 humor features  (layer {layer})")
    print(f"{'─'*60}")

    for idx_t, val_t in zip(top_idxs, top_vals):
        if len(results) >= 20:
            break
        idx   = idx_t.item()
        score = val_t.item()

        feat_dir = sae.W_dec[idx]
        if feat_dir.dtype != W_U.dtype:
            feat_dir = feat_dir.to(W_U.dtype)

        top_tokens = model.to_string(torch.topk(feat_dir @ W_U, 5).indices)

        if _is_artifact_feature(top_tokens):
            skipped += 1
            continue

        rank = len(results) + 1
        print(f"  #{rank:2d}  F{idx:5d}  Δ={score:+.4f}  tokens: {top_tokens}")
        results.append({
            "rank":        rank,
            "feature_idx": idx,
            "diff":        round(score, 6),
            "top_tokens":  top_tokens,
            "joke_mean":   round(mean_joke[idx].item(), 6),
            "nonjoke_mean":round(mean_nonjoke[idx].item(), 6),
        })

    print(f"\n  Kept {len(results)}, skipped {skipped} artifacts")
    print(f"{'─'*60}")

    out_path = save_dir / "gemma-2-2b_sae_features.json"
    with open(out_path, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"[save] ✅ {out_path}")

    del sae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results