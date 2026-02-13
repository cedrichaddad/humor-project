#!/usr/bin/env python3
"""
Run Complete SAE Experiment: Feature Discovery + Causal Validation.

This script combines:
1. SAE feature discovery (via sae_analysis.run_sae_analysis)
2. Causal validation via steering and ablation

Usage:
    python run_sae_experiment.py           # Uses layer 7, 5000 samples
    python run_sae_experiment.py 11        # Uses layer 11
    python run_sae_experiment.py 11 2000   # Layer 11, 2000 samples
"""
import json
import sys
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Import the complete SAE analysis function
from sae_analysis import run_sae_analysis, load_sae_model, analyze_features_with_model
from experiment import load_model, HumorIntervention, set_model, set_seed, SEED


def get_results_dir():
    """Get current model's results directory."""
    from experiment import RESULTS_DIR
    return RESULTS_DIR


def get_figures_dir():
    """Get current model's figures directory."""
    from experiment import FIGURES_DIR
    return FIGURES_DIR


def run_causal_validation(
    model,
    sae,
    top_feature_indices,
    layer: int,
    test_prompts=None
):
    """
    Run causal validation by steering and ablating SAE features.
    
    Args:
        model: HookedTransformer model
        sae: Loaded SAE model
        top_feature_indices: List of feature indices to test
        layer: Layer number
        test_prompts: Optional list of test prompts
        
    Returns:
        Dict with validation results
    """
    if test_prompts is None:
        test_prompts = [
            "Why did the chicken cross the road?",
            "I told my friend a joke about",
            "The weather today is"
        ]
    
    device = next(model.parameters()).device
    
    # Get feature interpretations
    feature_tokens = analyze_features_with_model(
        model, sae, top_feature_indices[:5], k=10
    )
    
    # Initialize intervention helper
    dummy_direction = torch.zeros(model.cfg.d_model).to(device)
    intervention = HumorIntervention(model, dummy_direction, layer=layer)
    
    print("\n" + "="*60)
    print("CAUSAL VALIDATION: Steering & Ablation")
    print("="*60)
    
    validation_results = []
    
    for idx in top_feature_indices[:5]:  # Test top 5 features
        feature_dir = sae.W_dec[idx]
        tokens = feature_tokens.get(idx, [])[:5]
        
        feature_result = {
            'feature_idx': idx,
            'top_tokens': tokens,
            'steering': {},
            'ablation': {}
        }
        
        print(f"\n{'─'*60}")
        print(f"Feature {idx} (Promotes: {', '.join(tokens)})")
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
    
    # Parse arguments
    layer = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    
    # Force GPT-2 (SAE only available for GPT-2)
    set_model("gpt2")
    set_seed(SEED)
    
    results_dir = get_results_dir()
    figures_dir = get_figures_dir()
    
    print("="*60)
    print("COMPLETE SAE EXPERIMENT: Discovery + Validation")
    print("="*60)
    print(f"Layer: {layer}")
    print(f"Samples: {n_samples}")
    print("="*60)
    
    # =========================================================================
    # STEP 1: Run SAE Analysis (Feature Discovery + Interpretation)
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 1: SAE Feature Discovery")
    print("="*60)
    
    # This does the complete SAE analysis and saves results
    sae_results = run_sae_analysis(
        model_name="gpt2",
        layer=layer,
        n_samples=n_samples
    )
    
    if sae_results is None:
        print("\nError: SAE analysis failed. Exiting.")
        return
    
    # Extract top features for validation
    top_features = sae_results['top_humor_features']
    top_indices = [f['feature_idx'] for f in top_features[:10]]
    
    print(f"\nTop 10 humor features identified: {top_indices}")
    
    # =========================================================================
    # STEP 2: Causal Validation (Steering + Ablation)
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 2: Causal Validation")
    print("="*60)
    
    # Load model and SAE for validation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model()[0]
    sae = load_sae_model(layer=layer, device=str(device))
    
    # Run causal tests
    validation_results = run_causal_validation(
        model, sae, top_indices, layer
    )
    
    # =========================================================================
    # STEP 3: Save Combined Results
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 3: Saving Complete Results")
    print("="*60)
    
    # Combine discovery + validation
    complete_results = {
        'layer': layer,
        'n_samples': n_samples,
        'discovery': {
            'top_features': sae_results['top_humor_features'],
            'feature_interpretations': sae_results['feature_interpretations']
        },
        'validation': validation_results
    }
    
    # Save to a different file to preserve the standalone sae_analysis.json
    output_path = results_dir / "sae_complete_experiment.json"
    with open(output_path, 'w') as f:
        json.dump(complete_results, f, indent=2)
    
    print(f"\nComplete results saved to: {output_path}")
    print(f"Discovery-only results: {results_dir / 'sae_analysis.json'}")
    print(f"Figures: {figures_dir / 'sae_humor_features.png'}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"\n✅ Discovered {len(top_features)} humor-specific SAE features")
    print(f"✅ Validated top {len(validation_results)} features via steering/ablation")
    print(f"\nKey finding: Feature {top_indices[0]} is most strongly correlated with humor")
    print(f"  Promotes tokens: {sae_results['feature_interpretations'][str(top_indices[0])][:5]}")
    
    return complete_results


if __name__ == "__main__":
    results = main()