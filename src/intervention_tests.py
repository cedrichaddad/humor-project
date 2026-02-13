#!/usr/bin/env python3
"""
Intervention Tests for Humor Recognition.

Supports multiple models via command line or function argument.

Usage:
    python intervention_tests.py                  # uses default (gpt2)
    python intervention_tests.py gemma-2-9b       # uses gemma
"""
import matplotlib.pyplot as plt
import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from experiment import (
    HumorIntervention,
    compute_logit_difference,
    evaluate_ablation_impact,
    train_linear_probe,
    extract_activations,
    load_model,
    set_model,
    get_display_name,
    set_seed,
    SEED,
)

warnings.filterwarnings('ignore')

CONTROL_PROMPTS = [
    "Why did the chicken cross the road?",
    "What do you call a fish without eyes?",
    "I told my friend a joke about",
    "Knock knock, who's there?",
    "The weather today is",
    "I went to the store and bought",
    "My favorite food is",
    "The quarterly earnings report shows that",
    "According to the latest research,",
    "In conclusion, the evidence suggests",
    "The primary objective of this project is to",
]

FUNNY_PROMPTS = [
    "Tell me something funny about",
    "Why did the mathematician",
    "A priest, a rabbi, and an imam walk into",
]


def get_results_dir():
    """Get current model's results directory."""
    from experiment import RESULTS_DIR
    return RESULTS_DIR


def get_figures_dir():
    """Get current model's figures directory."""
    from experiment import FIGURES_DIR
    return FIGURES_DIR


def run_steering_all_layers(
    model: HookedTransformer,
    humor_direction: torch.Tensor,
    alphas: List[float] = None,
    n_layers: int = None
) -> Dict:
    if n_layers is None:
        n_layers = model.cfg.n_layers
    if alphas is None:
        alphas = [-10.0, -5.0, 0.0, 5.0, 10.0]
    
    prompts = CONTROL_PROMPTS[:5]
    display_name = get_display_name()
    
    print(f"\n{'='*60}")
    print(f"Steering Across All Layers ({display_name})")
    print(f"{'='*60}")
    
    layer_effects = []
    
    for layer in tqdm(range(n_layers), desc="Layers"):
        intervention = HumorIntervention(model, humor_direction, layer)
        prompt_effects = []
        for prompt in prompts:
            diffs = {}
            for alpha in alphas:
                try:
                    logit_diff = compute_logit_difference(
                        model, prompt, intervention=intervention, alpha=alpha
                    )
                    diffs[alpha] = logit_diff['logit_difference']
                except:
                    diffs[alpha] = None
            if diffs[alphas[-1]] is not None and diffs[alphas[0]] is not None:
                effect = diffs[alphas[-1]] - diffs[alphas[0]]
                prompt_effects.append(effect)
        avg_effect = np.mean(prompt_effects) if prompt_effects else 0
        layer_effects.append({'layer': layer, 'avg_effect': avg_effect, 'per_prompt_effects': prompt_effects})
        print(f"  Layer {layer}: avg effect = {avg_effect:.4f}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    layers = [r['layer'] for r in layer_effects]
    effects = [r['avg_effect'] for r in layer_effects]
    colors = ['steelblue' if e >= 0 else 'salmon' for e in effects]
    ax.bar(layers, effects, color=colors)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Avg Steering Effect\n(Logit Diff at α=10 minus α=-10)', fontsize=11)
    ax.set_title(f'Steering Effect by Layer ({display_name})', fontsize=14)
    ax.set_xticks(range(n_layers))
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='r', linestyle='--')
    plt.tight_layout()
    
    figures_dir = get_figures_dir()
    plt.savefig(figures_dir / 'steering_by_layer.png', dpi=150)
    plt.close()
    print(f"Saved: {figures_dir / 'steering_by_layer.png'}")
    return {'steering_by_layer': layer_effects}


def run_steering_experiments(
    model: HookedTransformer,
    humor_direction: torch.Tensor,
    layer: int = 11,
    alphas: List[float] = None
) -> Dict:
    if alphas is None:
        alphas = [-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0]
    
    display_name = get_display_name()
    intervention = HumorIntervention(model, humor_direction, layer)
    results = []
    
    print(f"\n{'='*60}")
    print(f"Steering Experiments (Layer {layer}, {display_name})")
    print(f"{'='*60}")
    
    for prompt in tqdm(CONTROL_PROMPTS[:5], desc="Steering"):
        prompt_results = {'prompt': prompt, 'generations': {}, 'logit_diffs': {}}
        for alpha in alphas:
            try:
                generated = intervention.steer_humor(prompt, alpha=alpha, max_new_tokens=30, temperature=0.7)
                prompt_results['generations'][str(alpha)] = generated
            except Exception as e:
                prompt_results['generations'][str(alpha)] = f"Error: {str(e)}"
            try:
                logit_diff = compute_logit_difference(model, prompt, intervention=intervention, alpha=alpha)
                prompt_results['logit_diffs'][str(alpha)] = logit_diff
            except Exception as e:
                prompt_results['logit_diffs'][str(alpha)] = {'error': str(e)}
        results.append(prompt_results)
        print(f"\nPrompt: {prompt}")
        print(f"  α=0.0: {prompt_results['generations'].get('0.0', 'N/A')[:60]}...")
        print(f"  α=10.0: {prompt_results['generations'].get('10.0', 'N/A')[:60]}...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in results:
        alphas_plot, diffs_plot = [], []
        for alpha in alphas:
            ld = r['logit_diffs'].get(str(alpha), {})
            if 'logit_difference' in ld:
                alphas_plot.append(alpha)
                diffs_plot.append(ld['logit_difference'])
        ax.plot(alphas_plot, diffs_plot, '-o', label=r['prompt'][:30])
    ax.set_xlabel('Alpha (Steering Strength)')
    ax.set_ylabel('Logit Difference (Humor - Serious)')
    ax.set_title(f'Steering: Logit Diff vs Alpha (Layer {layer}, {display_name})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--')
    plt.tight_layout()
    
    figures_dir = get_figures_dir()
    plt.savefig(figures_dir / 'steering_logit_diff.png', dpi=150)
    plt.close()
    print(f"Saved: {figures_dir / 'steering_logit_diff.png'}")
    return {'steering_results': results, 'alphas': alphas}


def run_ablation_experiments(
    model: HookedTransformer,
    humor_direction: torch.Tensor,
    layer: int = 11
) -> Dict:
    display_name = get_display_name()
    intervention = HumorIntervention(model, humor_direction, layer)
    results = []
    
    print(f"\n{'='*60}")
    print(f"Ablation Experiments (Layer {layer}, {display_name})")
    print(f"{'='*60}")
    
    for prompt in tqdm(CONTROL_PROMPTS[:5], desc="Ablating"):
        try:
            original = intervention.steer_humor(prompt, alpha=0.0, max_new_tokens=30)
        except Exception as e:
            original = f"Error: {str(e)}"
        try:
            ablated = intervention.ablate_humor(prompt, max_new_tokens=30)
        except Exception as e:
            ablated = f"Error: {str(e)}"
        try:
            original_logits = compute_logit_difference(model, prompt)
            ablated_logits = compute_logit_difference(model, prompt, intervention=intervention, alpha=0.0)
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


def evaluate_probe_on_ablated_activations(
    model: HookedTransformer,
    humor_direction: torch.Tensor,
    layer: int = 11,
    n_samples: int = 200
) -> Dict:
    from datasets import load_dataset
    display_name = get_display_name()
    
    print(f"\n{'='*60}")
    print(f"Evaluating Ablation Impact on Probe (Layer {layer}, {display_name})")
    print(f"{'='*60}")
    
    device = next(model.parameters()).device
    intervention = HumorIntervention(model, humor_direction, layer)
    
    print("Loading test data...")
    dataset = load_dataset("CreativeLang/ColBERT_Humor_Detection")
    data = dataset['train'].shuffle(seed=SEED).select(range(n_samples))
    texts = [ex['text'] for ex in data]
    labels = np.array([1 if ex['humor'] else 0 for ex in data])
    
    print("Extracting original activations...")
    original_activations = extract_activations(model, texts, device, batch_size=32)
    X_original = original_activations[layer]
    
    print("Training probe...")
    split_idx = int(0.8 * len(texts))
    probe_result = train_linear_probe(
        X_original[:split_idx], labels[:split_idx],
        X_original[split_idx:], labels[split_idx:]
    )
    probe = probe_result['probe']
    
    print("Extracting ablated activations...")
    X_ablated = intervention.get_ablated_activations(texts, batch_size=32)
    
    impact = evaluate_ablation_impact(probe, X_original[split_idx:], X_ablated[split_idx:], labels[split_idx:])
    
    print(f"\nAblation Impact Results:")
    print(f"  Original accuracy: {impact['original_accuracy']:.3f}")
    print(f"  Ablated accuracy:  {impact['ablated_accuracy']:.3f}")
    print(f"  Accuracy drop:     {impact['accuracy_drop']:.3f} ({impact['drop_percentage']:.1f}%)")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(['Original', 'Ablated'], [impact['original_accuracy'], impact['ablated_accuracy']], color=['steelblue', 'salmon'])
    ax.set_ylabel('Probe Accuracy')
    ax.set_title(f'Ablation Impact at Layer {layer}\n({display_name}, Drop: {impact["drop_percentage"]:.1f}%)')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='r', linestyle='--', label='Chance')
    ax.legend()
    plt.tight_layout()
    
    figures_dir = get_figures_dir()
    plt.savefig(figures_dir / 'ablation_impact.png', dpi=150)
    plt.close()
    print(f"Saved: {figures_dir / 'ablation_impact.png'}")
    return {'ablation_impact': impact, 'n_test_samples': len(texts) - split_idx}


def run_interventions(model_name: str = None):
    """Run all intervention tests for a given model."""
    if model_name:
        set_model(model_name)
    
    set_seed(SEED)
    results_dir = get_results_dir()
    figures_dir = get_figures_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    display_name = get_display_name()
    
    print("="*60)
    print(f"Intervention Tests ({display_name})")
    print("="*60)
    
    humor_direction_path = results_dir / "humor_direction.pt"
    if not humor_direction_path.exists():
        print(f"\nError: {humor_direction_path} not found!")
        print("Please run experiment.py first.")
        return
    
    print(f"\nLoading humor direction from: {humor_direction_path}")
    humor_direction = torch.load(humor_direction_path)
    print(f"  Shape: {humor_direction.shape}")
    print(f"  Norm: {humor_direction.norm().item():.4f}")
    
    model, device = load_model()
    humor_direction = humor_direction.to(device)
    
    # Read best layer from saved results
    config_path = results_dir / "rank_analysis.json"
    if config_path.exists():
        with open(config_path) as f:
            BEST_LAYER = json.load(f).get('best_layer', model.cfg.n_layers - 1)
    else:
        BEST_LAYER = model.cfg.n_layers - 1
        print(f"Warning: rank_analysis.json not found, using last layer ({BEST_LAYER})")
    print(f"Using best layer: {BEST_LAYER}")
    
    all_results = {}
    
    layer_steering = run_steering_all_layers(model, humor_direction)
    all_results.update(layer_steering)
    
    steering_results = run_steering_experiments(model, humor_direction, layer=BEST_LAYER)
    all_results.update(steering_results)
    
    ablation_results = run_ablation_experiments(model, humor_direction, layer=BEST_LAYER)
    all_results.update(ablation_results)
    
    try:
        impact_results = evaluate_probe_on_ablated_activations(model, humor_direction, layer=BEST_LAYER, n_samples=200)
        all_results.update(impact_results)
    except Exception as e:
        print(f"\nWarning: Could not evaluate probe impact: {e}")
        all_results['ablation_impact'] = {'error': str(e)}
    
    results_path = results_dir / "intervention_results.json"
    
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)): return float(obj)
        elif isinstance(obj, (np.int32, np.int64)): return int(obj)
        elif isinstance(obj, dict): return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list): return [convert_for_json(i) for i in obj]
        return obj
    
    with open(results_path, 'w') as f:
        json.dump(convert_for_json(all_results), f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"INTERVENTION TESTS COMPLETE ({display_name})")
    print(f"{'='*60}")
    print(f"\nResults saved to: {results_path}")
    return all_results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else None
    results = run_interventions(model_name)