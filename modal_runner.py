"""
src/modal_runner.py

Modal cloud runner for the *complete* SAE humor experiment on A100.

Pipeline (runs remotely)
────────────────────────
  Step 1 – SAE Feature Discovery   (sae_analysis.get_humor_features)
  Step 2 – Causal Validation        (run_sae_experiment.run_causal_validation)
  Step 3 – Save combined results    JSON → /root/results/

Architecture
────────────
• Pinned dependency versions for reproducible builds.
• Persistent HuggingFace cache volume (skip ~2 min re-download per run).
• Persistent results volume (survive pod restarts).
• HuggingFace secret for gated model access (Gemma-2-2B).
"""

import modal

# ═══════════════════════════════════════════════════════════════════════════
# Volumes
# ═══════════════════════════════════════════════════════════════════════════
hf_cache_vol = modal.Volume.from_name("humor-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("humor-results", create_if_missing=True)

# ═══════════════════════════════════════════════════════════════════════════
# Container image – all deps for experiment.py + sae_analysis.py
# ═══════════════════════════════════════════════════════════════════════════
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Core
        "torch==2.4.0",
        "numpy==1.26.4",
        "pandas==2.2.0",
        # ML / NLP
        "transformer-lens==2.8.0",
        "sae-lens",
        "transformers==4.45.0",
        "typeguard",
        "einops",
        "huggingface_hub",
        # Experiment deps (experiment.py)
        "scikit-learn==1.5.0",
        "matplotlib==3.9.0",
        "seaborn==0.13.0",
        "datasets==2.21.0",
        # Data
        "openpyxl==3.1.0",
        "tqdm",
    )
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("datasets", remote_path="/root/datasets")
)

app = modal.App("humor-sae-analysis", image=image)

# ═══════════════════════════════════════════════════════════════════════════
# Shared function kwargs
# ═══════════════════════════════════════════════════════════════════════════
GPU_KWARGS = dict(
    gpu="A100",
    cpu=4.0,
    memory=32768,
    timeout=2 * 3600,       # 2 hours
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/results": results_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)


# ═══════════════════════════════════════════════════════════════════════════
# Step 1 only: SAE Feature Discovery
# ═══════════════════════════════════════════════════════════════════════════

@app.function(**GPU_KWARGS)
def run_feature_discovery(batch_size: int = 128):
    """Run only SAE feature extraction (Step 1)."""
    import sys
    sys.path.insert(0, "/root")

    from src.sae_analysis import get_humor_features

    results = get_humor_features(
        model_alias="gemma-2-2b",
        data_path="/root/datasets/dataset_a_paired.xlsx",
        batch_size=batch_size,
        save_dir="/root/results",
    )
    results_vol.commit()
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Full pipeline: Discovery + Causal Validation
# ═══════════════════════════════════════════════════════════════════════════

@app.function(**GPU_KWARGS)
def run_full_experiment(batch_size: int = 128):
    """Run the complete SAE experiment: discovery → causal validation → save.

    This mirrors src/run_sae_experiment.py but runs entirely on an A100.
    """
    import json
    import os
    import sys
    from datetime import datetime
    from pathlib import Path

    start = datetime.now()

    # ── Environment setup ─────────────────────────────────────────────
    os.chdir("/root")
    sys.path.insert(0, "/root/src")

    from sae_analysis import get_humor_features, load_sae_model
    from experiment import set_model, set_seed, SEED
    from intervention_tests import HumorIntervention
    import torch

    model_name = "gemma-2-2b"
    set_model(model_name)
    set_seed(SEED)

    results_dir = Path(f"/root/results/{model_name}")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  COMPLETE SAE EXPERIMENT  ({model_name})")
    print(f"  Started : {start:%Y-%m-%d %H:%M:%S}")
    print(f"  Batch   : {batch_size}")
    print("=" * 60)

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: SAE Feature Discovery
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STEP 1 / 4 : SAE Feature Discovery")
    print("=" * 60)

    sae_results = get_humor_features(
        model_alias=model_name,
        data_path="/root/datasets/dataset_a_paired.xlsx",
        batch_size=batch_size,
        save_dir=str(results_dir),
    )

    if not sae_results:
        raise RuntimeError("SAE feature discovery returned no results.")

    top_indices = [r["feature_idx"] for r in sae_results[:10]]
    print(f"\n  Top 10 feature indices: {top_indices}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: Causal Validation (Steering + Ablation)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STEP 2 / 4 : Causal Validation")
    print("=" * 60)

    model, sae, cfg = load_sae_model(model_name)
    device = next(model.parameters()).device

    test_prompts = [
        "Why did the chicken cross the road?",
        "I told my friend a joke about",
        "The weather today is",
    ]

    # Dummy direction to initialise HumorIntervention
    dummy_dir = torch.zeros(model.cfg.d_model, device=device)
    if model.cfg.dtype == torch.bfloat16:
        dummy_dir = dummy_dir.to(dtype=torch.bfloat16)

    intervention = HumorIntervention(model, dummy_dir, layer=cfg["layer"])

    validation_results = []
    for idx in top_indices[:5]:
        feature_dir = sae.W_dec[idx]
        feature_dir = feature_dir / feature_dir.norm()
        if feature_dir.dtype != dummy_dir.dtype:
            feature_dir = feature_dir.to(dtype=dummy_dir.dtype)

        entry = {"feature_idx": idx, "steering": {}, "ablation": {}}

        print(f"\n{'─' * 60}")
        print(f"  Feature {idx}")
        print(f"{'─' * 60}")

        for prompt in test_prompts:
            print(f"\n    Prompt: '{prompt}'")

            steered = intervention.steer_direction(
                feature_dir, prompt, alpha=30.0,
                max_new_tokens=20, temperature=0.7,
            )
            entry["steering"][prompt] = steered
            print(f"      Steered (α=30): {steered}")

            ablated = intervention.ablate_direction(
                feature_dir, prompt,
                max_new_tokens=20, temperature=0.7,
            )
            entry["ablation"][prompt] = ablated
            print(f"      Ablated       : {ablated}")

        validation_results.append(entry)

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: Save Combined Results
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STEP 3 / 4 : Saving Results")
    print("=" * 60)

    complete = {
        "model": model_name,
        "layer": cfg["layer"],
        "discovery": sae_results,
        "validation": validation_results,
    }

    out_path = results_dir / f"{model_name}_sae_complete_experiment.json"
    with open(out_path, "w") as f:
        json.dump(complete, f, indent=2)

    # ══════════════════════════════════════════════════════════════════
    # STEP 4: Generate Figures
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STEP 4 / 4 : Generating Figures")
    print("=" * 60)
    
    try:
        import subprocess
        result = subprocess.run(
            ["python", "/root/src/generate_sae_figures.py"],
            cwd="/root",
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.returncode == 0:
            print(f"\n✅ Figures saved → {results_dir / 'figures'}")
        else:
            print(f"\n⚠️  Figure generation had errors:")
            print(result.stderr)
    except Exception as e:
        print(f"\n⚠️  Figure generation failed: {e}")

    results_vol.commit()

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n✅ Complete experiment saved → {out_path}")
    print(f"   Duration: {elapsed / 60:.1f} min")

    return complete


# ═══════════════════════════════════════════════════════════════════════════
# Local entry point
# ═══════════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main(
    batch_size: int = 128,
    detach: bool = False,
    discovery_only: bool = False,
):
    """
    Usage
    ─────
      modal run src/modal_runner.py                          # full experiment
      modal run src/modal_runner.py --discovery-only         # features only
      modal run src/modal_runner.py --detach                 # background
      modal run src/modal_runner.py --batch-size 64          # smaller batch
    """
    import json
    from pathlib import Path

    mode = "Feature Discovery Only" if discovery_only else "Full Experiment"
    print("🚀 HUMOR SAE ANALYSIS – Modal Runner")
    print("=" * 60)
    print(f"  GPU      : A100")
    print(f"  Batch    : {batch_size}")
    print(f"  Pipeline : {mode}")
    print(f"  Mode     : {'Background' if detach else 'Interactive'}")
    print("=" * 60)

    fn = run_feature_discovery if discovery_only else run_full_experiment

    if detach:
        call = fn.spawn(batch_size=batch_size)
        print(f"⏳ Spawned  (call id: {call.object_id})")
        print("   Check Modal dashboard for progress.")
        return

    results = fn.remote(batch_size=batch_size)

    # Save locally
    if discovery_only:
        out = Path("results/gemma-2-2b_sae_features.json")
    else:
        out = Path("results/gemma-2-2b_sae_complete_experiment.json")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📄 Local copy saved → {out}")