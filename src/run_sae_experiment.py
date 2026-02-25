#!/usr/bin/env python3
"""
Run Complete SAE Experiment: Feature Discovery + Causal Validation.

Sweeps layers 15-20, saving each to results/<model>/sae_results/layer_N/.

Usage:
    python src/run_sae_experiment.py                   # gemma-2-2b, batch 4
    python src/run_sae_experiment.py --batch-size 128
"""
import json
import torch
import argparse
from pathlib import Path

from sae_analysis import (
    get_humor_features_for_layer,
    load_sae_model,
    SWEEP_LAYERS,
    SAE,
    _GEMMA_CANONICAL_RELEASE,
)
from experiment import set_model, set_seed, SEED
from intervention_tests import HumorIntervention
from generate_sae_figures import generate_figures_for_layer


def run_causal_validation(model, sae, top_feature_indices, layer, model_name="gemma-2-2b"):
    test_prompts = [
        "Why did the chicken cross the road?",
        "I told my friend a joke about",
        "The weather today is",
    ]

    device = next(model.parameters()).device
    dummy  = torch.zeros(model.cfg.d_model, device=device)
    if model.cfg.dtype == torch.bfloat16:
        dummy = dummy.to(dtype=torch.bfloat16)

    intervention = HumorIntervention(model, dummy, layer=layer)
    results = []

    for idx in top_feature_indices[:5]:
        feat_dir = sae.W_dec[idx]
        feat_dir = feat_dir / feat_dir.norm()
        if feat_dir.dtype != dummy.dtype:
            feat_dir = feat_dir.to(dtype=dummy.dtype)

        entry = {"feature_idx": idx, "steering": {}, "ablation": {}}
        print(f"\n  Feature {idx}")

        for prompt in test_prompts:
            steered = intervention.steer_direction(feat_dir, prompt, alpha=30.0, max_new_tokens=20, temperature=0.7)
            ablated = intervention.ablate_direction(feat_dir, prompt, max_new_tokens=20, temperature=0.7)
            entry["steering"][prompt] = steered
            entry["ablation"][prompt] = ablated
            print(f"    [{prompt[:40]}]")
            print(f"      steered: {steered}")
            print(f"      ablated: {ablated}")

        results.append(entry)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str, default="gemma-2-2b", choices=["gemma-2-2b", "gpt2"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    model_name = args.model
    set_model(model_name)
    set_seed(SEED)

    sae_results_dir = Path(f"results/{model_name}/sae_results")
    sae_results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  SAE EXPERIMENT  ({model_name})  layers {SWEEP_LAYERS}")
    print("=" * 60)

    # Load transformer once — reused across all layers
    model, _, _ = load_sae_model(model_name)
    device   = next(model.parameters()).device
    use_bf16 = model.cfg.dtype == torch.bfloat16

    for layer in SWEEP_LAYERS:
        layer_dir = sae_results_dir / f"layer_{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}\n  Layer {layer}\n{'='*60}")

        # ── Feature discovery ──────────────────────────────────────────
        features = get_humor_features_for_layer(
            layer=layer,
            model=model,
            batch_size=args.batch_size,
            save_dir=layer_dir,
        )
        top_indices = [r["feature_idx"] for r in features[:10]]
        print(f"\n  Top 10: {top_indices}")

        # ── Causal validation ──────────────────────────────────────────
        layer_sae, _, _ = SAE.from_pretrained(
            release=_GEMMA_CANONICAL_RELEASE,
            sae_id=f"layer_{layer}/width_16k/canonical",
            device=device,
        )
        if use_bf16:
            layer_sae = layer_sae.to(dtype=torch.bfloat16)

        validation = run_causal_validation(model, layer_sae, top_indices, layer, model_name)

        del layer_sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Save ──────────────────────────────────────────────────────
        combined = {
            "model":      model_name,
            "layer":      layer,
            "discovery":  features,
            "validation": validation,
        }
        out = layer_dir / "gemma-2-2b_sae_complete_experiment.json"
        with open(out, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\n  Saved: {out}")

        # ── Figures ───────────────────────────────────────────────────
        try:
            generate_figures_for_layer(layer, sae_results_dir)
        except Exception as e:
            print(f"  WARNING figures layer {layer}: {e}")

    print("\n" + "=" * 60)
    print("  ALL LAYERS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()