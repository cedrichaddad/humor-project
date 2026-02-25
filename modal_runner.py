"""
src/modal_runner.py

Modal cloud runner for the SAE humor experiment on A100.
Sweeps layers 15-20, saving each to results/gemma-2-2b/sae_results/layer_N/.

Usage:
    modal run modal_runner.py
    modal run modal_runner.py --batch-size 64
    modal run modal_runner.py --detach
"""

import modal

# ═══════════════════════════════════════════════════════════════════════════
# Volumes
# ═══════════════════════════════════════════════════════════════════════════
hf_cache_vol = modal.Volume.from_name("humor-hf-cache", create_if_missing=True)
results_vol  = modal.Volume.from_name("humor-results",  create_if_missing=True)

# ═══════════════════════════════════════════════════════════════════════════
# Container image
# ═══════════════════════════════════════════════════════════════════════════
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "numpy==1.26.4",
        "pandas==2.2.0",
        "transformer-lens==2.8.0",
        "sae-lens",
        "transformers==4.45.0",
        "typeguard",
        "einops",
        "huggingface_hub",
        "scikit-learn==1.5.0",
        "matplotlib==3.9.0",
        "seaborn==0.13.0",
        "datasets==2.21.0",
        "openpyxl==3.1.0",
        "tqdm",
    )
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App("humor-sae-analysis", image=image)

# ═══════════════════════════════════════════════════════════════════════════
# Main function: layers 15-20 sweep
# ═══════════════════════════════════════════════════════════════════════════

@app.function(
    gpu="A100-80GB",
    cpu=8.0,
    memory=65536,
    timeout=6 * 3600,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/results": results_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_sae_experiment(batch_size: int = 128):
    """Run discovery + causal validation + figures for layers 15-20.

    Output:
        results/gemma-2-2b/sae_results/
            layer_15/  gemma-2-2b_sae_complete_experiment.json + figures/
            layer_16/  ...
            ...
            layer_20/  ...
    """
    import json
    import os
    import sys
    import torch
    import importlib.util
    from datetime import datetime
    from pathlib import Path

    start = datetime.now()
    os.chdir("/root")
    sys.path.insert(0, "/root/src")

    from sae_analysis import (
        get_humor_features_for_layer,
        load_sae_model,
        SWEEP_LAYERS,
        SAE,
        _GEMMA_CANONICAL_RELEASE,
    )
    from experiment import set_model, set_seed, SEED
    from intervention_tests import HumorIntervention

    model_name = "gemma-2-2b"
    set_model(model_name)
    set_seed(SEED)

    sae_results_dir = Path(f"/root/results/{model_name}/sae_results")
    sae_results_dir.mkdir(parents=True, exist_ok=True)

    test_prompts = [
        "Why did the chicken cross the road?",
        "I told my friend a joke about",
        "The weather today is",
    ]

    print("=" * 60)
    print(f"  SAE EXPERIMENT  ({model_name})  layers {SWEEP_LAYERS}")
    print(f"  Started : {start:%Y-%m-%d %H:%M:%S}")
    print(f"  Batch   : {batch_size}")
    print("=" * 60)

    # Load generate_sae_figures
    _spec = importlib.util.spec_from_file_location(
        "generate_sae_figures", "/root/src/generate_sae_figures.py"
    )
    _fig_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_fig_mod)
    generate_figures_for_layer = _fig_mod.generate_figures_for_layer

    # Load transformer once — reused across all layers
    model    = load_sae_model(model_name)[0]
    device   = next(model.parameters()).device
    use_bf16 = model.cfg.dtype == torch.bfloat16

    dummy_dir = torch.zeros(model.cfg.d_model, device=device)
    if use_bf16:
        dummy_dir = dummy_dir.to(dtype=torch.bfloat16)

    summary = {}

    for layer in SWEEP_LAYERS:
        layer_start = datetime.now()
        layer_dir   = sae_results_dir / f"layer_{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}\n  LAYER {layer}\n{'='*60}")

        # ── Feature discovery ──────────────────────────────────────────
        features    = get_humor_features_for_layer(layer=layer, model=model,
                                                   batch_size=batch_size, save_dir=layer_dir)
        top_indices = [r["feature_idx"] for r in features[:10]]
        print(f"  Top 10: {top_indices}")

        # ── Causal validation ──────────────────────────────────────────
        layer_sae, _, _ = SAE.from_pretrained(
            release=_GEMMA_CANONICAL_RELEASE,
            sae_id=f"layer_{layer}/width_16k/canonical",
            device=device,
        )
        if use_bf16:
            layer_sae = layer_sae.to(dtype=torch.bfloat16)

        intervention = HumorIntervention(model, dummy_dir, layer=layer)
        validation   = []

        for idx in top_indices[:5]:
            feat_dir = layer_sae.W_dec[idx]
            feat_dir = feat_dir / feat_dir.norm()
            if feat_dir.dtype != dummy_dir.dtype:
                feat_dir = feat_dir.to(dtype=dummy_dir.dtype)

            entry = {"feature_idx": idx, "steering": {}, "ablation": {}}
            print(f"\n  Feature {idx}")

            for prompt in test_prompts:
                entry["steering"][prompt] = intervention.steer_direction(
                    feat_dir, prompt, alpha=30.0, max_new_tokens=20, temperature=0.7)
                entry["ablation"][prompt] = intervention.ablate_direction(
                    feat_dir, prompt, max_new_tokens=20, temperature=0.7)
                print(f"    [{prompt[:40]}]")
                print(f"      steered: {entry['steering'][prompt]}")
                print(f"      ablated: {entry['ablation'][prompt]}")

            validation.append(entry)

        del layer_sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Save ──────────────────────────────────────────────────────
        combined = {"model": model_name, "layer": layer,
                    "discovery": features, "validation": validation}
        out = layer_dir / "gemma-2-2b_sae_complete_experiment.json"
        with open(out, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\n  Saved: {out}")

        # ── Figures ───────────────────────────────────────────────────
        try:
            generate_figures_for_layer(layer, sae_results_dir)
        except Exception as e:
            print(f"  WARNING figures layer {layer}: {e}")

        elapsed = (datetime.now() - layer_start).total_seconds()
        summary[layer] = {"top_10": top_indices, "elapsed_min": round(elapsed / 60, 1)}
        print(f"  Layer {layer} done in {elapsed/60:.1f} min")

    results_vol.commit()

    total = (datetime.now() - start).total_seconds()
    print(f"\n{'='*60}")
    print(f"  COMPLETE  ({total/60:.1f} min total)")
    print(f"{'='*60}")
    for layer, info in summary.items():
        print(f"  Layer {layer}: {info['top_10']}  ({info['elapsed_min']} min)")

    return summary


# ═══════════════════════════════════════════════════════════════════════════
# Local entry point
# ═══════════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main(batch_size: int = 128, detach: bool = False):
    """
    Usage:
        modal run src/modal_runner.py
        modal run src/modal_runner.py --batch-size 64
        modal run src/modal_runner.py --detach
    """
    print("SAE Experiment  |  layers 15-20  |  A100-80GB")
    print(f"batch={batch_size}  detach={detach}")

    if detach:
        call = run_sae_experiment.spawn(batch_size=batch_size)
        print(f"Spawned (id: {call.object_id}) — check Modal dashboard.")
        return

    run_sae_experiment.remote(batch_size=batch_size)