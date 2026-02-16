#!/usr/bin/env python3
"""
Intervention Tests for Humor Recognition.

Supports multiple models via command line or function argument.

Usage:
    python intervention_tests.py                  # uses default (gpt2)
    python intervention_tests.py gemma-2-2b       # uses gemma
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
    compute_activation_projection,
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

# ---------------------------------------------------------------------------
# Prompt sets — 7 per category, 21 total
# These are used for qualitative testing of steering and ablation effects
# ---------------------------------------------------------------------------

# Humor prompts: baseline already humor-biased; tests amplification
HUMOR_PROMPTS = [
    "Why did the chicken cross the road?",
    "What do you call a fish without eyes?",
    "Knock knock, who's there?",
    "A man walked into a bar and",
    "A priest, a rabbi, and an imam walk into",
    "Tell me something funny about",
    "So a penguin walks into a",
]

# Neutral prompts: no humor or formality bias; tests creation from nothing
NEUTRAL_PROMPTS = [
    "The weather today is",
    "I went to the store and bought",
    "My favorite food is",
    "She opened the door and saw",
    "The dog ran across the",
    "Last weekend I decided to",
    "When I was a kid, I used to",
]

# Serious prompts: formal/academic baseline; tests creation against anti-humor
SERIOUS_PROMPTS = [
    "The quarterly earnings report shows that",
    "According to the latest research,",
    "In conclusion, the evidence suggests",
    "The primary objective of this project is to",
    "The committee has determined that",
    "Recent developments in the field indicate",
    "The data clearly demonstrates that",
]

ALL_PROMPTS = HUMOR_PROMPTS + NEUTRAL_PROMPTS + SERIOUS_PROMPTS

# Map each prompt to its category for per-category analysis
PROMPT_CATEGORIES = {}
for p in HUMOR_PROMPTS:
    PROMPT_CATEGORIES[p] = 'humor'
for p in NEUTRAL_PROMPTS:
    PROMPT_CATEGORIES[p] = 'neutral'
for p in SERIOUS_PROMPTS:
    PROMPT_CATEGORIES[p] = 'serious'


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
    humor_direction: torch.Tensor = None,  # Now optional
    alphas: List[float] = None,
    n_layers: int = None,
    use_layer_specific: bool = True
) -> Dict:
    """
    Test steering effectiveness across all layers.
    
    For each layer:
    - Applies steering at different strengths (alphas) to 21 prompts
    - Measures the effect on humor vs serious logit difference
    - Creates visualizations showing which layers are most effective for steering
    
    Produces three plots: overall, neutral+serious only, humor only.
    """
    if n_layers is None:
        n_layers = model.cfg.n_layers
    if alphas is None:
        # Negative alpha = steer away from humor, positive = toward humor
        alphas = [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0]
    
    prompts = ALL_PROMPTS
    display_name = get_display_name()
    results_dir = get_results_dir()
    directions_dir = results_dir / "directions"
    device = next(model.parameters()).device
    
    # Check if layer-specific directions exist (from main experiment)
    layer_specific_exists = (directions_dir / "layer0.pt").exists()
    
    if use_layer_specific and layer_specific_exists:
        print(f"\n{'='*60}")
        print(f"Steering Across All Layers with Layer-Specific Directions ({display_name})")
        print(f"  Loading from: {directions_dir}/")
        print(f"  Using {len(prompts)} prompts (7 humor, 7 neutral, 7 serious)")
        print(f"  Alphas: {alphas}")
        print(f"{'='*60}")
        use_specific = True
    else:
        print(f"\n{'='*60}")
        print(f"Steering Across All Layers ({display_name})")
        print(f"{'='*60}")
        if use_layer_specific and not layer_specific_exists:
            print("⚠️  Layer-specific directions not found, using fallback direction")
            print(f"    (Checked: {directions_dir / 'layer0.pt'})")
        use_specific = False
        if humor_direction is None:
            # Load the best layer's humor direction as fallback
            humor_direction = torch.load(results_dir / "humor_direction.pt").to(device)
        fallback_direction = humor_direction
    
    layer_effects = []
    
    # Test each layer independently
    for layer in tqdm(range(n_layers), desc="Layers"):
        # Load appropriate direction for this layer
        if use_specific:
            # Use layer-specific direction learned from Dataset A
            direction_path = directions_dir / f"layer{layer}.pt"
            humor_direction = torch.load(direction_path).to(device)
        else:
            # Use the best layer's direction for all layers
            humor_direction = fallback_direction
        
        # Create intervention object for this layer
        intervention = HumorIntervention(model, humor_direction, layer)
        
        # Track effects per prompt with category
        prompt_results = []
        for prompt in prompts:
            diffs = {}
            # Test different steering strengths
            for alpha in alphas:
                try:
                    # Measure logit difference (humor tokens vs serious tokens) with steering applied
                    logit_diff = compute_logit_difference(
                        model, prompt, humor_direction, intervention=intervention, alpha=alpha
                    )
                    diffs[alpha] = logit_diff['logit_difference']
                except:
                    diffs[alpha] = None
            # Calculate steering effect: difference between max positive and max negative steering
            if diffs[alphas[-1]] is not None and diffs[alphas[0]] is not None:
                effect = diffs[alphas[-1]] - diffs[alphas[0]]
                prompt_results.append({
                    'prompt': prompt,
                    'category': PROMPT_CATEGORIES[prompt],
                    'effect': effect,
                })
        
        # Compute per-category averages
        all_effects = [r['effect'] for r in prompt_results]
        humor_effects = [r['effect'] for r in prompt_results if r['category'] == 'humor']
        control_effects = [r['effect'] for r in prompt_results if r['category'] in ('neutral', 'serious')]
        
        layer_effects.append({
            'layer': layer, 
            'avg_effect': np.mean(all_effects) if all_effects else 0,
            'avg_humor': np.mean(humor_effects) if humor_effects else 0,
            'avg_control': np.mean(control_effects) if control_effects else 0,
            'per_prompt': prompt_results,
            'used_layer_specific': use_specific,
        })
        print(f"  Layer {layer}: all={layer_effects[-1]['avg_effect']:.4f}  "
              f"control={layer_effects[-1]['avg_control']:.4f}  "
              f"humor={layer_effects[-1]['avg_humor']:.4f}")
    
    # --- Create visualizations ---
    figures_dir = get_figures_dir()
    title_suffix = "(Layer-Specific)" if use_specific else ""
    
    def _save_bar_chart(effects_list, title_extra, filename):
        """Helper function to create bar charts of steering effects by layer."""
        fig, ax = plt.subplots(figsize=(10, 6))
        layers = list(range(n_layers))
        # Blue for positive (increases humor), red for negative (decreases humor)
        colors = ['steelblue' if e >= 0 else 'salmon' for e in effects_list]
        ax.bar(layers, effects_list, color=colors)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel(f'Avg Steering Effect\n(Logit Diff at α={alphas[-1]} minus α={alphas[0]})', fontsize=11)
        ax.set_title(f'Steering Effect by Layer {title_suffix} {title_extra} ({display_name})', fontsize=13)
        ax.set_xticks(range(n_layers))
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='r', linestyle='--')
        plt.tight_layout()
        plt.savefig(figures_dir / filename, dpi=150)
        plt.close()
        print(f"Saved: {figures_dir / filename}")
    
    # Plot 1: Overall (all 21 prompts)
    all_effs = [r['avg_effect'] for r in layer_effects]
    suffix = '_fixed' if use_specific else ''
    _save_bar_chart(all_effs, '— All Prompts', f'steering_by_layer{suffix}.png')
    
    # Plot 2: Neutral + Serious only (creation from non-humorous baseline)
    ctrl_effs = [r['avg_control'] for r in layer_effects]
    _save_bar_chart(ctrl_effs, '— Neutral + Serious', f'steering_by_layer_control{suffix}.png')
    
    # Plot 3: Humor only (amplification of existing humor)
    humor_effs = [r['avg_humor'] for r in layer_effects]
    _save_bar_chart(humor_effs, '— Humor Prompts', f'steering_by_layer_humor{suffix}.png')
    
    return {'steering_by_layer': layer_effects, 'used_layer_specific': use_specific}



def run_steering_experiments(
    model: HookedTransformer,
    humor_direction: torch.Tensor,
    layer: int = 11,
    alphas: List[float] = None
) -> Dict:
    """
    QUALITATIVE steering experiment: Generate text at different steering strengths.
    
    This is a text generation experiment (not activation extraction).
    For each of 21 prompts:
    - Generate text at various alpha values (steering strengths)
    - Measure internal metrics (logit differences, activation projections)
    - Show how steering strength affects both generated text and internal representations
    
    Uses the humor_direction learned from Dataset A.
    """
    if alphas is None:
        # Range of steering strengths to test
        alphas = [-15.0, -10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, 15.0]
    
    display_name = get_display_name()
    intervention = HumorIntervention(model, humor_direction, layer)
    results = []
    
    print(f"\n{'='*60}")
    print(f"Steering Experiments (Layer {layer}, {display_name})")
    print(f"  Using {len(ALL_PROMPTS)} prompts, alphas={alphas}")
    print(f"{'='*60}")
    
    for prompt in tqdm(ALL_PROMPTS, desc="Steering"):
        prompt_results = {'prompt': prompt, 'generations': {}, 'logit_diffs': {}, 'projections': {}}

        # Test each steering strength
        for alpha in alphas:
            try:
                # GENERATE TEXT with steering applied at this layer
                # alpha = 0.0 means no steering (baseline)
                # alpha > 0 means steer toward humor
                # alpha < 0 means steer away from humor
                generated = intervention.steer_humor(prompt, alpha=alpha, max_new_tokens=30, temperature=0.7)
                prompt_results['generations'][str(alpha)] = generated
                
            except Exception as e:
                prompt_results['generations'][str(alpha)] = f"Error: {str(e)}"
            try:
                # Measure logit difference (humor tokens vs serious tokens) with steering
                logit_diff = compute_logit_difference(
                    model, prompt, humor_direction, intervention=intervention, alpha=alpha
                )
                prompt_results['logit_diffs'][str(alpha)] = logit_diff
            except Exception as e:
                prompt_results['logit_diffs'][str(alpha)] = {'error': str(e)}
            try:
                # Measure how much the activation aligns with humor direction
                proj = compute_activation_projection(
                    model, prompt, humor_direction, layer, 
                    intervention=intervention, alpha=alpha
                )
                prompt_results['projections'][str(alpha)] = proj
            except Exception as e:
                prompt_results['projections'][str(alpha)] = None
        results.append(prompt_results)
        # Show examples of generated text at baseline vs max steering
        print(f"\nPrompt: {prompt}")
        print(f"  α=0.0:  {prompt_results['generations'].get('0.0', 'N/A')[:60]}...")
        print(f"  α=15.0: {prompt_results['generations'].get('15.0', 'N/A')[:60]}...")
    
    # Visualization 1: How logit difference changes with steering strength
    fig, ax = plt.subplots(figsize=(12, 7))
    for r in results:
        alphas_plot, diffs_plot = [], []
        for alpha in alphas:
            ld = r['logit_diffs'].get(str(alpha), {})
            if 'logit_difference' in ld:
                alphas_plot.append(alpha)
                diffs_plot.append(ld['logit_difference'])
        ax.plot(alphas_plot, diffs_plot, '-o', label=r['prompt'][:30], markersize=4)
    ax.set_xlabel('Alpha (Steering Strength)')
    ax.set_ylabel('Logit Difference (Humor - Serious)')
    ax.set_title(f'Steering: Logit Diff vs Alpha (Layer {layer}, {display_name})')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--')
    plt.tight_layout()
    
    figures_dir = get_figures_dir()
    plt.savefig(figures_dir / 'steering_logit_diff.png', dpi=150)
    plt.close()
    print(f"Saved: {figures_dir / 'steering_logit_diff.png'}")

    # Visualization 2: Triangulation plot showing internal vs external measures
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Activation Projection Plot (internal alignment with humor direction)
    for r in results:
        alphas_p = sorted([float(a) for a in r['projections'].keys()])
        projs = [r['projections'][str(a)] for a in alphas_p]
        axes[0].plot(alphas_p, projs, alpha=0.3)
    axes[0].set_title("Internal: Activation Projection")
    axes[0].set_xlabel("Alpha")
    axes[0].set_ylabel("Alignment Score")
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Logit Difference Plot (external behavior - humor vs serious tokens)
    for r in results:
        alphas_l = sorted([float(a) for a in r['logit_diffs'].keys()])
        ldiffs = [r['logit_diffs'][str(a)]['logit_difference'] for a in alphas_l]
        axes[1].plot(alphas_l, ldiffs, alpha=0.3)
    axes[1].set_title("Internal: Robust Logit Diff")
    axes[1].set_xlabel("Alpha")
    axes[1].set_ylabel("Logit Difference")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Steering Analysis (Layer {layer}, {display_name})")
    plt.tight_layout()
    plt.savefig(figures_dir / 'steering_triangulation.png', dpi=150)
    plt.close()
    print(f"Saved: {figures_dir / 'steering_triangulation.png'}")

    return {'steering_results': results, 'alphas': alphas}

def run_ablation_experiments(
    model: HookedTransformer,
    humor_direction: torch.Tensor,
    layer: int = 11
) -> Dict:
    """
    QUALITATIVE ablation experiment: Generate text with humor direction removed.
    
    This is a text generation experiment (not the quantitative probe accuracy test).
    For each of 21 prompts:
    - Generate text normally (baseline)
    - Generate text with humor direction ablated (removed)
    - Compare how the generated text changes
    
    This provides qualitative/illustrative evidence of what removing humor does to text.
    The quantitative causal test is in evaluate_probe_on_ablated_activations().
    
    Uses the humor_direction learned from Dataset A.
    """
    display_name = get_display_name()
    intervention = HumorIntervention(model, humor_direction, layer)
    results = []
    
    print(f"\n{'='*60}")
    print(f"Ablation Experiments (Layer {layer}, {display_name})")
    print(f"  Using {len(ALL_PROMPTS)} prompts")
    print(f"{'='*60}")
    
    for prompt in tqdm(ALL_PROMPTS, desc="Ablating"):
        try:
            # BASELINE: Generate text with no intervention
            # alpha=0.0 means no steering, just normal generation
            original = intervention.steer_humor(prompt, alpha=0.0, max_new_tokens=30)
        except Exception as e:
            original = f"Error: {str(e)}"
        try:
            # ABLATED: Generate text with humor direction removed from activations
            # This removes the component of layer activations that aligns with humor_direction
            ablated = intervention.ablate_humor(prompt, max_new_tokens=30)
        except Exception as e:
            ablated = f"Error: {str(e)}"
        try:
            # Measure logit differences for both conditions
            original_logits = compute_logit_difference(model, prompt, humor_direction)
            ablated_logits = compute_logit_difference(model, prompt, humor_direction, intervention=intervention, alpha=0.0)
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
        # Show side-by-side comparison of generated text
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
    """
    QUANTITATIVE ablation experiment: Measure causal necessity of humor direction.
    
    This is the main causal test that produces the ablation_impact.png graph.
    
    Dataset: ColBERT Humor Detection (different from Dataset A used to learn humor_direction)
    
    Steps:
    1. Load 200 samples from ColBERT dataset
    2. Extract NORMAL activations (straight from model) at specified layer for all 200 samples
    3. Train a NEW probe on first 160 normal activations (this probe learns to use the humor direction)
    4. Test probe on last 40 normal activations → "Original" accuracy (~97.8%)
    5. Extract ABLATED activations for all 200 samples (humor direction removed during forward pass)
    6. Test same probe on last 40 ablated activations → "Ablated" accuracy (~46.5%)
    
    The drop from ~97.8% to ~46.5% (near chance) proves the humor direction is causally necessary,
    not just correlated with humor classification.
    """
    from datasets import load_dataset
    display_name = get_display_name()
    
    print(f"\n{'='*60}")
    print(f"Evaluating Ablation Impact on Probe (Layer {layer}, {display_name})")
    print(f"{'='*60}")
    
    device = next(model.parameters()).device
    # Create intervention object with humor_direction learned from Dataset A
    intervention = HumorIntervention(model, humor_direction, layer)
    
    print("Loading test data...")
    # Load ColBERT dataset (completely different from Dataset A)
    dataset = load_dataset("CreativeLang/ColBERT_Humor_Detection")
    data = dataset['train'].shuffle(seed=SEED).select(range(n_samples))
    texts = [ex['text'] for ex in data]  # 200 text strings
    labels = np.array([1 if ex['humor'] else 0 for ex in data])  # 200 binary labels
    
    print("Extracting original activations...")
    # Run all 200 texts through model NORMALLY (no intervention)
    # This is just a forward pass to capture layer activations
    original_activations = extract_activations(model, texts, device, batch_size=32)
    X_original = original_activations[layer]  # Shape: (200, hidden_dim)
    
    print("Training probe...")
    # Split: first 160 for training, last 40 for testing
    split_idx = int(0.8 * len(texts))  # = 160
    # Train a NEW probe on the NORMAL activations from ColBERT
    # This probe will learn to use the humor direction for classification
    probe_result = train_linear_probe(
        X_original[:split_idx], labels[:split_idx],      # Train on 160 normal activations
        X_original[split_idx:], labels[split_idx:]       # Test on 40 normal activations
    )
    probe = probe_result['probe']  # This is a scikit-learn LogisticRegression model
    
    print("Extracting ablated activations...")
    # Run all 200 texts through model again, BUT with ablation hook active
    # The hook removes the humor_direction component from activations during forward pass
    # mathematically: activation_ablated = activation - (activation · humor_direction) * humor_direction
    X_ablated = intervention.get_ablated_activations(texts, batch_size=32)  # Shape: (200, hidden_dim)
    
    # Test the probe on both normal and ablated activations
    # The probe was trained expecting the humor direction to be present
    impact = evaluate_ablation_impact(
        probe,                      # Probe trained on normal activations
        X_original[split_idx:],     # Last 40 normal test activations → ~97.8% accuracy
        X_ablated[split_idx:],      # Last 40 ablated test activations → ~46.5% accuracy
        labels[split_idx:]          # True labels for last 40 samples
    )
    
    print(f"\nAblation Impact Results:")
    print(f"  Original accuracy: {impact['original_accuracy']:.3f}")
    print(f"  Ablated accuracy:  {impact['ablated_accuracy']:.3f}")
    print(f"  Accuracy drop:     {impact['accuracy_drop']:.3f} ({impact['drop_percentage']:.1f}%)")
    
    # Create the bar chart visualization (ablation_impact.png)
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
    """
    Run all intervention tests for a given model.
    
    This orchestrates all the causal experiments:
    1. Steering across all layers - which layers are most effective?
    2. Steering experiments - how does text generation change with steering?
    3. Ablation experiments (qualitative) - what happens to generated text when humor is removed?
    4. Ablation impact (quantitative) - causal proof via probe accuracy drop
    """
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
    print(f"  Prompts: {len(ALL_PROMPTS)} total ({len(NEUTRAL_PROMPTS)} neutral, {len(SERIOUS_PROMPTS)} serious)")
    print("="*60)
    
    # Load the humor direction learned from Dataset A in the main experiment
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
    
    # Read best layer from saved results (the layer with highest probe accuracy)
    config_path = results_dir / "rank_analysis.json"
    if config_path.exists():
        with open(config_path) as f:
            BEST_LAYER = json.load(f).get('best_layer', model.cfg.n_layers - 1)
    else:
        BEST_LAYER = model.cfg.n_layers - 1
        print(f"Warning: rank_analysis.json not found, using last layer ({BEST_LAYER})")
    print(f"Using best layer: {BEST_LAYER}")
    
    all_results = {}
    
    # Test 1: Which layers are most effective for steering?
    layer_steering = run_steering_all_layers(model, humor_direction)
    all_results.update(layer_steering)
    
    # Test 2: Qualitative steering - how does generated text change?
    steering_results = run_steering_experiments(model, humor_direction, layer=BEST_LAYER)
    all_results.update(steering_results)
    
    # Test 3: Qualitative ablation - what happens to text when humor is removed?
    ablation_results = run_ablation_experiments(model, humor_direction, layer=BEST_LAYER)
    all_results.update(ablation_results)
    
    # Test 4: Quantitative ablation - causal proof via probe accuracy
    # This is the key experiment that proves causal necessity
    try:
        impact_results = evaluate_probe_on_ablated_activations(model, humor_direction, layer=BEST_LAYER, n_samples=200)
        all_results.update(impact_results)
    except Exception as e:
        print(f"\nWarning: Could not evaluate probe impact: {e}")
        all_results['ablation_impact'] = {'error': str(e)}
    
    # Save all results to JSON
    results_path = results_dir / "intervention_results.json"
    
    def convert_for_json(obj):
        """Convert numpy types to JSON-serializable types."""
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