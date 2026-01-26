#!/usr/bin/env python3
"""
Intervention Tests for Humor Recognition.

This script runs steering and ablation experiments on control prompts using
the trained humor direction from experiment.py.

Usage:
    python src/intervention_tests.py
"""

import os
import json
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

# Import from experiment module
from experiment import (
    HumorIntervention,
    compute_logit_difference,
    evaluate_ablation_impact,
    train_linear_probe,
    extract_activations,
    load_model,
    set_seed,
    SEED,
    RESULTS_DIR
)

warnings.filterwarnings('ignore')

# =============================================================================
# Control Prompts
# =============================================================================

CONTROL_PROMPTS = [
    # Joke setups
    "Why did the chicken cross the road?",
    "What do you call a fish without eyes?",
    "I told my friend a joke about",
    "Knock knock, who's there?",
    
    # Neutral prompts
    "The weather today is",
    "I went to the store and bought",
    "My favorite food is",
    
    # Serious/formal prompts  
    "The quarterly earnings report shows that",
    "According to the latest research,",
    "In conclusion, the evidence suggests",
    "The primary objective of this project is to",
]

# Funny continuation prompts
FUNNY_PROMPTS = [
    "Tell me something funny about",
    "Why did the mathematician",
    "A priest, a rabbi, and an imam walk into",
]

# =============================================================================
# Steering Experiments
# =============================================================================

def run_steering_experiments(
    model: HookedTransformer,
    humor_direction: torch.Tensor,
    layer: int = 7,
    alphas: List[float] = None
) -> Dict:
    """
    Run steering experiments with various alpha values.
    
    Args:
        model: HookedTransformer model
        humor_direction: Normalized humor direction
        layer: Layer to intervene at
        alphas: List of steering strengths to test
        
    Returns:
        Dict with steering results
    """
    if alphas is None:
        alphas = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    
    intervention = HumorIntervention(model, humor_direction, layer)
    results = []
    
    print(f"\n{'='*60}")
    print("Steering Experiments")
    print(f"{'='*60}")
    
    for prompt in tqdm(CONTROL_PROMPTS[:5], desc="Steering"):  # Limit for speed
        prompt_results = {
            'prompt': prompt,
            'generations': {},
            'logit_diffs': {}
        }
        
        for alpha in alphas:
            # Generate text with steering
            try:
                generated = intervention.steer_humor(
                    prompt, 
                    alpha=alpha, 
                    max_new_tokens=30,
                    temperature=0.7
                )
                prompt_results['generations'][str(alpha)] = generated
            except Exception as e:
                prompt_results['generations'][str(alpha)] = f"Error: {str(e)}"
            
            # Compute logit difference
            try:
                logit_diff = compute_logit_difference(
                    model, prompt, 
                    intervention=intervention, 
                    alpha=alpha
                )
                prompt_results['logit_diffs'][str(alpha)] = logit_diff
            except Exception as e:
                prompt_results['logit_diffs'][str(alpha)] = {'error': str(e)}
        
        results.append(prompt_results)
        
        # Print sample
        print(f"\nPrompt: {prompt}")
        print(f"  α=0.0: {prompt_results['generations'].get('0.0', 'N/A')[:60]}...")
        print(f"  α=2.0: {prompt_results['generations'].get('2.0', 'N/A')[:60]}...")
    
    return {'steering_results': results, 'alphas': alphas}


# =============================================================================
# Ablation Experiments
# =============================================================================

def run_ablation_experiments(
    model: HookedTransformer,
    humor_direction: torch.Tensor,
    layer: int = 7
) -> Dict:
    """
    Run ablation experiments to remove humor feature.
    
    Returns:
        Dict with ablation results
    """
    intervention = HumorIntervention(model, humor_direction, layer)
    results = []
    
    print(f"\n{'='*60}")
    print("Ablation Experiments")
    print(f"{'='*60}")
    
    for prompt in tqdm(CONTROL_PROMPTS[:5], desc="Ablating"):
        # Generate without ablation
        try:
            original = intervention.steer_humor(prompt, alpha=0.0, max_new_tokens=30)
        except Exception as e:
            original = f"Error: {str(e)}"
        
        # Generate with ablation
        try:
            ablated = intervention.ablate_humor(prompt, max_new_tokens=30)
        except Exception as e:
            ablated = f"Error: {str(e)}"
        
        # Compute logit differences
        try:
            original_logits = compute_logit_difference(model, prompt)
            ablated_logits = compute_logit_difference(
                model, prompt, 
                intervention=intervention,
                alpha=0.0  # We'll use ablation hook separately
            )
        except Exception as e:
            original_logits = {'error': str(e)}
            ablated_logits = {'error': str(e)}
        
        results.append({
            'prompt': prompt,
            'original_generation': original,
            'ablated_generation': ablated,
            'original_logits': original_logits,
            'ablated_logits': ablated_logits
        })
        
        print(f"\nPrompt: {prompt}")
        print(f"  Original: {original[:60]}...")
        print(f"  Ablated:  {ablated[:60]}...")
    
    return {'ablation_results': results}


# =============================================================================
# Ablation Impact on Probe Accuracy
# =============================================================================

def evaluate_probe_on_ablated_activations(
    model: HookedTransformer,
    humor_direction: torch.Tensor,
    layer: int = 7,
    n_samples: int = 200
) -> Dict:
    """
    Evaluate how ablation affects probe accuracy.
    
    Trains a probe on original activations, then tests on ablated activations.
    """
    from datasets import load_dataset
    
    print(f"\n{'='*60}")
    print("Evaluating Ablation Impact on Probe")
    print(f"{'='*60}")
    
    device = next(model.parameters()).device
    intervention = HumorIntervention(model, humor_direction, layer)
    
    # Load test data
    print("Loading test data...")
    dataset = load_dataset("CreativeLang/ColBERT_Humor_Detection")
    data = dataset['train'].shuffle(seed=SEED).select(range(n_samples))
    
    texts = [ex['text'] for ex in data]
    labels = np.array([1 if ex['humor'] else 0 for ex in data])
    
    # Extract original activations
    print("Extracting original activations...")
    original_activations = extract_activations(model, texts, device, batch_size=32)
    X_original = original_activations[layer]
    
    # Train probe on original
    print("Training probe...")
    split_idx = int(0.8 * len(texts))
    probe_result = train_linear_probe(
        X_original[:split_idx], labels[:split_idx],
        X_original[split_idx:], labels[split_idx:]
    )
    probe = probe_result['probe']
    
    # Get ablated activations
    print("Extracting ablated activations...")
    X_ablated = intervention.get_ablated_activations(texts, batch_size=32)
    
    # Evaluate
    impact = evaluate_ablation_impact(
        probe,
        X_original[split_idx:],
        X_ablated[split_idx:],
        labels[split_idx:]
    )
    
    print(f"\nAblation Impact Results:")
    print(f"  Original accuracy: {impact['original_accuracy']:.3f}")
    print(f"  Ablated accuracy:  {impact['ablated_accuracy']:.3f}")
    print(f"  Accuracy drop:     {impact['accuracy_drop']:.3f} ({impact['drop_percentage']:.1f}%)")
    
    return {
        'ablation_impact': impact,
        'n_test_samples': len(texts) - split_idx
    }


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all intervention tests."""
    set_seed(SEED)
    
    # Create results directory
    RESULTS_DIR.mkdir(exist_ok=True)
    
    print("="*60)
    print("Intervention Tests for Humor Recognition")
    print("="*60)
    
    # Load humor direction
    humor_direction_path = RESULTS_DIR / "humor_direction.pt"
    if not humor_direction_path.exists():
        print(f"\nError: {humor_direction_path} not found!")
        print("Please run experiment.py first to generate the humor direction.")
        return
    
    print(f"\nLoading humor direction from: {humor_direction_path}")
    humor_direction = torch.load(humor_direction_path)
    print(f"  Shape: {humor_direction.shape}")
    print(f"  Norm: {humor_direction.norm().item():.4f}")
    
    # Load model
    model, device = load_model()
    humor_direction = humor_direction.to(device)
    
    # Run experiments
    all_results = {}
    
    # 1. Steering experiments
    steering_results = run_steering_experiments(
        model, humor_direction, layer=7
    )
    all_results.update(steering_results)
    
    # 2. Ablation experiments
    ablation_results = run_ablation_experiments(
        model, humor_direction, layer=7
    )
    all_results.update(ablation_results)
    
    # 3. Probe ablation impact
    try:
        impact_results = evaluate_probe_on_ablated_activations(
            model, humor_direction, layer=7, n_samples=200
        )
        all_results.update(impact_results)
    except Exception as e:
        print(f"\nWarning: Could not evaluate probe impact: {e}")
        all_results['ablation_impact'] = {'error': str(e)}
    
    # Save results
    results_path = RESULTS_DIR / "intervention_results.json"
    
    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(i) for i in obj]
        return obj
    
    all_results_json = convert_for_json(all_results)
    
    with open(results_path, 'w') as f:
        json.dump(all_results_json, f, indent=2)
    
    print(f"\n{'='*60}")
    print("INTERVENTION TESTS COMPLETE")
    print(f"{'='*60}")
    print(f"\nResults saved to: {results_path}")
    
    return all_results


if __name__ == "__main__":
    results = main()
