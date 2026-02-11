#!/usr/bin/env python3
"""
Intervention Tests for Humor Recognition.

This script runs steering and ablation experiments on control prompts using
the trained humor direction from experiment.py.

Usage:
    python src/intervention_tests.py
"""
import matplotlib.pyplot as plt
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
# Steering Across All Layers
# =============================================================================

def run_steering_all_layers(
    model: HookedTransformer,
    humor_direction: torch.Tensor,
    alphas: List[float] = None,
    n_layers: int = 12
) -> Dict:
    """
    Run steering at each layer separately and measure the effect.
    Produces a bar chart showing which layer gives the strongest steering effect.
    """
    if alphas is None:
        alphas = [-10.0, -5.0, 0.0, 5.0, 10.0]
    
    prompts = CONTROL_PROMPTS[:5]
    
    print(f"\n{'='*60}")
    print("Steering Across All Layers")
    print(f"{'='*60}")
    
    # For each layer, compute average effect across prompts
    layer_effects = []
    
    for layer in tqdm(range(n_layers), desc="Layers"):
        intervention = HumorIntervention(model, humor_direction, layer)
        
        prompt_effects = []
        for prompt in prompts:
            diffs = {}
            for alpha in alphas:
                try:
                    logit_diff = compute_logit_difference(
                        model, prompt,
                        intervention=intervention,
                        alpha=alpha
                    )
                    diffs[alpha] = logit_diff['logit_difference']
                except:
                    diffs[alpha] = None
            
            # Effect = difference between highest and lowest alpha
            if diffs[alphas[-1]] is not None and diffs[alphas[0]] is not None:
                effect = diffs[alphas[-1]] - diffs[alphas[0]]
                prompt_effects.append(effect)
        
        avg_effect = np.mean(prompt_effects) if prompt_effects else 0
        layer_effects.append({
            'layer': layer,
            'avg_effect': avg_effect,
            'per_prompt_effects': prompt_effects
        })
        print(f"  Layer {layer}: avg effect = {avg_effect:.4f}")
    
    # Plot: bar chart of steering effect by layer
    fig, ax = plt.subplots(figsize=(10, 6))
    layers = [r['layer'] for r in layer_effects]
    effects = [r['avg_effect'] for r in layer_effects]
    colors = ['steelblue' if e >= 0 else 'salmon' for e in effects]
    ax.bar(layers, effects, color=colors)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Avg Steering Effect\n(Logit Diff at α=10 minus α=-10)', fontsize=11)
    ax.set_title('Steering Effect by Layer (Humor Direction from Layer 11)', fontsize=14)
    ax.set_xticks(range(n_layers))
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig('figures/steering_by_layer.png', dpi=150)
    plt.close()
    print("Saved: figures/steering_by_layer.png")
    
    return {'steering_by_layer': layer_effects}


# =============================================================================
# Steering Experiments (single layer, detailed)
# =============================================================================

def run_steering_experiments(
    model: HookedTransformer,
    humor_direction: torch.Tensor,
    layer: int = 11,
    alphas: List[float] = None
) -> Dict:
    """
    Run steering experiments with various alpha values at a single layer.
    """
    if alphas is None:
        alphas = [-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0]
    
    intervention = HumorIntervention(model, humor_direction, layer)
    results = []
    
    print(f"\n{'='*60}")
    print(f"Steering Experiments (Layer {layer})")
    print(f"{'='*60}")
    
    for prompt in tqdm(CONTROL_PROMPTS[:5], desc="Steering"):
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
        print(f"  α=10.0: {prompt_results['generations'].get('10.0', 'N/A')[:60]}...")
    
    # Plot logit difference vs alpha
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in results:
        alphas_plot = []
        diffs_plot = []
        for alpha in alphas:
            ld = r['logit_diffs'].get(str(alpha), {})
            if 'logit_difference' in ld:
                alphas_plot.append(alpha)
                diffs_plot.append(ld['logit_difference'])
        ax.plot(alphas_plot, diffs_plot, '-o', label=r['prompt'][:30])

    ax.set_xlabel('Alpha (Steering Strength)')
    ax.set_ylabel('Logit Difference (Humor - Serious)')
    ax.set_title(f'Steering: Humor Logit Difference vs Alpha (Layer {layer})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig('figures/steering_logit_diff.png', dpi=150)
    plt.close()
    print("Saved: figures/steering_logit_diff.png")
    return {'steering_results': results, 'alphas': alphas}


# =============================================================================
# Ablation Experiments
# =============================================================================

def run_ablation_experiments(
    model: HookedTransformer,
    humor_direction: torch.Tensor,
    layer: int = 11
) -> Dict:
    """
    Run ablation experiments to remove humor feature.
    """
    intervention = HumorIntervention(model, humor_direction, layer)
    results = []
    
    print(f"\n{'='*60}")
    print(f"Ablation Experiments (Layer {layer})")
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
                alpha=0.0
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
    layer: int = 11,
    n_samples: int = 200
) -> Dict:
    """
    Evaluate how ablation affects probe accuracy.
    
    Trains a probe on original activations at the SAME layer,
    then tests on ablated activations. This ensures the probe
    and the ablation direction are in the same activation space.
    """
    from datasets import load_dataset
    
    print(f"\n{'='*60}")
    print(f"Evaluating Ablation Impact on Probe (Layer {layer})")
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
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        ['Original', 'Ablated'],
        [impact['original_accuracy'], impact['ablated_accuracy']],
        color=['steelblue', 'salmon']
    )
    ax.set_ylabel('Probe Accuracy')
    ax.set_title(f'Ablation Impact on Humor Probe at Layer {layer}\n(Drop: {impact["drop_percentage"]:.1f}%)')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='r', linestyle='--', label='Chance')
    ax.legend()
    plt.tight_layout()
    plt.savefig('figures/ablation_impact.png', dpi=150)
    plt.close()
    print("Saved: figures/ablation_impact.png")

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
    Path("figures").mkdir(exist_ok=True)
    
    print("="*60)
    print("Intervention Tests for Humor Recognition")
    print("="*60)
    
    # Load humor direction (extracted from layer 11 by experiment.py)
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
    
    # The humor direction was extracted from layer 11.
    # We use layer 11 for ablation (must match probe layer).
    # For steering, we test ALL layers to find the most effective one.
    BEST_LAYER = 11
    
    # Run experiments
    all_results = {}
    
    # 1. Steering across all layers (find best layer for steering)
    layer_steering = run_steering_all_layers(model, humor_direction)
    all_results.update(layer_steering)
    
    # 2. Detailed steering at layer 11 (with generations)
    steering_results = run_steering_experiments(
        model, humor_direction, layer=BEST_LAYER
    )
    all_results.update(steering_results)
    
    # 3. Ablation experiments at layer 11 (matches humor direction)
    ablation_results = run_ablation_experiments(
        model, humor_direction, layer=BEST_LAYER
    )
    all_results.update(ablation_results)
    
    # 4. Probe ablation impact at layer 11 (matches humor direction)
    try:
        impact_results = evaluate_probe_on_ablated_activations(
            model, humor_direction, layer=BEST_LAYER, n_samples=200
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
    print(f"\nFigures generated:")
    print(f"  - figures/steering_by_layer.png")
    print(f"  - figures/steering_logit_diff.png")
    print(f"  - figures/ablation_impact.png")
    
    return all_results


if __name__ == "__main__":
    results = main()