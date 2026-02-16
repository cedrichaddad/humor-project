#!/usr/bin/env python3
"""
Run Complete SAE Experiment: Feature Discovery + Causal Validation.

This script combines:
1. SAE feature discovery (via sae_analysis.get_humor_features)
2. Causal validation via steering and ablation

Usage:
    python src/run_sae_experiment.py --model gpt2 --batch_size 4
    python src/run_sae_experiment.py --model gemma-2-2b --batch_size 128 (if local)
"""
import json
import sys
import torch
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

# Import the new SAE analysis function
from sae_analysis import get_humor_features, load_sae_model
from experiment import HumorIntervention, set_model, set_seed, SEED

def get_results_dir(model_name):
    """Get current model's results directory."""
    return Path(f"results/{model_name}")

def run_causal_validation(
    model,
    sae,
    top_feature_indices,
    layer: int,
    test_prompts=None,
    model_name="gemma-2-2b"
):
    """
    Run causal validation by steering and ablating SAE features.
    """
    if test_prompts is None:
        test_prompts = [
            "Why did the chicken cross the road?",
            "I told my friend a joke about",
            "The weather today is"
        ]
    
    device = next(model.parameters()).device
    
    # Initialize intervention helper
    # Dummy direction to init
    dummy_direction = torch.zeros(model.cfg.d_model, device=device)
    if model.cfg.dtype == torch.bfloat16:
        dummy_direction = dummy_direction.to(dtype=torch.bfloat16)

    intervention = HumorIntervention(model, dummy_direction, layer=layer)
    
    print("\n" + "="*60)
    print("CAUSAL VALIDATION: Steering & Ablation")
    print("="*60)
    
    validation_results = []
    
    # We need access to feature decoder directions
    for idx in top_feature_indices[:5]:  # Test top 5 features
        # Normalize direction
        feature_dir = sae.W_dec[idx]
        feature_dir = feature_dir / feature_dir.norm()
        
        # Ensure dtype match
        if feature_dir.dtype != dummy_direction.dtype:
            feature_dir = feature_dir.to(dtype=dummy_direction.dtype)

        feature_result = {
            'feature_idx': idx,
            'steering': {},
            'ablation': {}
        }
        
        print(f"\n{'─'*60}")
        print(f"Feature {idx}")
        print(f"{'─'*60}")
        
        for prompt in test_prompts:
            # SAE features are sparse; higher alpha needed for visible effects
            print(f"\n  Prompt: '{prompt}'")
            
            # Steering
            steered = intervention.steer_direction(
                feature_dir, prompt, alpha=30.0, max_new_tokens=20, temperature=0.7
            )
            feature_result['steering'][prompt] = steered
            print(f"    Steered (α=30): {steered}")
            
            # Ablation
            ablated = intervention.ablate_direction(
                feature_dir, prompt, max_new_tokens=20, temperature=0.7
            )
            feature_result['ablation'][prompt] = ablated
            print(f"    Ablated:        {ablated}")
        
        validation_results.append(feature_result)
    
    return validation_results


def main():
    """Run complete SAE experiment: discovery + validation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", choices=["gpt2", "gemma-2-2b"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--samples", type=int, default=2000)
    args = parser.parse_args()
    
    model_name = args.model
    # Set model context
    set_model(model_name)
    set_seed(SEED)
    
    results_dir = get_results_dir(model_name)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print(f"COMPLETE SAE EXPERIMENT: Discovery + Validation ({model_name})")
    print("="*60)
    
    # =========================================================================
    # STEP 1: Run SAE Analysis (Feature Discovery)
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 1: SAE Feature Discovery")
    print("="*60)
    
    # Call our new API
    sae_results = get_humor_features(
        model_alias=model_name,
        batch_size=args.batch_size,
        save_dir=results_dir
    )
    
    if not sae_results:
        print("\nError: SAE analysis failed. Exiting.")
        return
    
    # Extract top feature indices
    top_indices = [r['feature_idx'] for r in sae_results[:10]]
    print(f"\nTop 10 humor features identified: {top_indices}")
    
    # =========================================================================
    # STEP 2: Causal Validation (Steering + Ablation)
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 2: Causal Validation")
    print("="*60)
    
    # Load model and SAE for validation using our factory
    model, sae, cfg = load_sae_model(model_name)
    
    # Run causal tests
    validation_results = run_causal_validation(
        model, sae, top_indices, cfg['layer'], model_name=model_name
    )
    
    # =========================================================================
    # STEP 3: Save Combined Results
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 3: Saving Complete Results")
    print("="*60)
    
    complete_results = {
        'model': model_name,
        'layer': cfg['layer'],
        'discovery': sae_results,
        'validation': validation_results
    }
    
    output_path = results_dir / f"{model_name}_sae_complete_experiment.json"
    with open(output_path, 'w') as f:
        json.dump(complete_results, f, indent=2)
    
    print(f"\nComplete results saved to: {output_path}")
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()