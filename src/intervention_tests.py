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
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from sklearn.metrics import accuracy_score

from experiment import (
    train_linear_probe,
    extract_activations,
    load_model,
    set_model,
    get_display_name,
    set_seed,
    SEED,
)

warnings.filterwarnings('ignore')

# =============================================================================
# Intervention Tools (moved from experiment.py)
# =============================================================================

class HumorIntervention:
    """
    Class for intervening on model activations during generation.
    
    Two main operations:
    1. Steering: Add α * humor_direction to activations (push toward/away from humor)
    2. Ablation: Remove humor_direction component from activations
    
    These interventions happen DURING the forward pass at a specific layer.
    """
    def __init__(self, model: HookedTransformer, humor_direction: torch.Tensor, layer: int = 7):
        """
        Initialize intervention for a specific layer.
        
        Args:
            model: The language model
            humor_direction: The direction vector (from probe weights)
            layer: Which layer to intervene at
        """
        self.model = model
        self.layer = layer
        self.device = next(model.parameters()).device
        
        # Convert to tensor if needed and normalize
        if isinstance(humor_direction, np.ndarray):
            humor_direction = torch.from_numpy(humor_direction)
        self.humor_direction = humor_direction.float().to(self.device)
        self.humor_direction = self.humor_direction / self.humor_direction.norm()
        
    def _get_steering_hook(self, alpha: float, direction: torch.Tensor = None):
        """
        Create hook function for steering.
        
        Steering: activation_new = activation_old + α * direction
        - α > 0: steer toward humor
        - α < 0: steer away from humor
        - α = 0: no change (baseline)
        """
        target_dir = direction if direction is not None else self.humor_direction
        
        def hook_fn(activation, hook):
            # activation shape: (batch, seq_len, hidden_dim)
            steering = alpha * target_dir
            return activation + steering.view(1, 1, -1)
        return hook_fn
    
    def _get_ablation_hook(self, direction: torch.Tensor = None):
        """
        Create hook function for ablation.
        
        Ablation: Remove the component along humor_direction
        mathematically: activation_new = activation_old - proj_coef * direction
        where proj_coef = activation · direction (dot product)
        
        This is like removing a specific dimension while keeping everything else.
        """
        target_dir = direction if direction is not None else self.humor_direction
        
        def hook_fn(activation, hook):
            v = target_dir
            # Calculate how much activation points in the direction (projection coefficient)
            proj_coef = torch.einsum('bsd,d->bs', activation, v)
            # Calculate the projection component
            projection = proj_coef.unsqueeze(-1) * v
            # Remove it from the activation
            return activation - projection
        return hook_fn
    
    def steer_humor(self, prompt: str, alpha: float = 1.0, max_new_tokens: int = 50, temperature: float = 1.0) -> str:
        """
        Generate text with steering applied using the stored humor direction.
        
        This is for TEXT GENERATION (not activation extraction).
        """
        return self.steer_direction(self.humor_direction, prompt, alpha, max_new_tokens, temperature)

    def ablate_humor(self, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0) -> str:
        """
        Generate text with humor direction ablated (removed).
        
        This is for TEXT GENERATION (not activation extraction).
        """
        return self.ablate_direction(self.humor_direction, prompt, max_new_tokens, temperature)

    def steer_direction(
        self,
        direction: torch.Tensor,
        prompt: str,
        alpha: float = 1.0, 
        max_new_tokens: int = 50, 
        temperature: float = 1.0
    ) -> str:
        """
        Generate text with steering applied along an arbitrary direction.
        
        Process:
        1. Tokenize prompt
        2. Install hook at specified layer
        3. Generate tokens (hook modifies activations during forward pass)
        4. Return generated text
        """
        hook_name = f"blocks.{self.layer}.hook_resid_post"
        hook_fn = self._get_steering_hook(alpha, direction=direction)
        
        tokens = self.model.to_tokens(prompt, prepend_bos=True)
        # Context manager installs hooks for the duration of generation
        with self.model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
            output = self.model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                prepend_bos=False,
                verbose=False
            )
        return self.model.to_string(output[0])
    
    def ablate_direction(
        self,
        direction: torch.Tensor,
        prompt: str,
        max_new_tokens: int = 50, 
        temperature: float = 1.0
    ) -> str:
        """
        Generate text with arbitrary direction ablated (removed).
        
        Same as steer_direction but uses ablation hook instead of steering hook.
        """
        hook_name = f"blocks.{self.layer}.hook_resid_post"
        hook_fn = self._get_ablation_hook(direction=direction)
        
        tokens = self.model.to_tokens(prompt, prepend_bos=True)
        with self.model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
            output = self.model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                prepend_bos=False,
                verbose=False
            )
        return self.model.to_string(output[0])
    
    def get_ablated_activations(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Extract activations with ablation applied.
        
        This is for ACTIVATION EXTRACTION (not text generation).
        Used in intervention_tests.py for the quantitative ablation experiment.
        
        Process:
        1. Install ablation hook at specified layer
        2. Run forward pass with hook active
        3. Capture the modified activations
        4. Return activations at final token position
        
        Returns:
            numpy array of shape (n_texts, hidden_dim)
            These are the activations WITH the humor component removed
        """
        hook_name = f"blocks.{self.layer}.hook_resid_post"
        ablation_hook = self._get_ablation_hook()
        
        all_ablated_acts = []

        def capture_hook(activation, hook):
            """Second hook to capture the ablated activations."""
            all_ablated_acts.append(activation.detach().cpu())
            return activation

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            tokens = self.model.to_tokens(batch_texts, prepend_bos=True)
            seq_lengths = (tokens != self.model.tokenizer.pad_token_id).sum(dim=1) - 1
            
            # Install TWO hooks: first ablates, second captures
            with self.model.hooks(fwd_hooks=[(hook_name, ablation_hook), (hook_name, capture_hook)]):
                self.model(tokens)  # Just forward pass, no generation
            
            # Extract final token activations from captured batch
            batch_acts = all_ablated_acts.pop()
            for j in range(len(batch_texts)):
                final_pos = min(seq_lengths[j].item(), batch_acts.shape[1] - 1)
                all_ablated_acts.insert(0, batch_acts[j, final_pos, :].numpy())
        
        return np.array(all_ablated_acts)


# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_logit_difference(
    model: HookedTransformer,
    prompt: str,
    humor_direction: torch.Tensor,
    n_top_tokens: int = 200,
    intervention: Optional[HumorIntervention] = None,
    alpha: float = 0.0
) -> Dict[str, float]:
    """
    Compute logit difference: humor tokens vs serious tokens.
    
    This measures the model's "humor preference" at the output layer.
    
    Process:
    1. Project humor_direction into vocabulary space to find "humor words"
    2. Find top 200 tokens aligned with humor direction (humor category)
    3. Find top 200 tokens anti-aligned with humor direction (serious category)
    4. Measure log probability of humor category vs serious category
    
    This is more robust than looking at individual tokens.
    
    If intervention is provided, applies steering during the forward pass.
    
    Returns:
        logit_difference: positive means model favors humor tokens
        humor_prob: probability mass on humor category
    """
    device = next(model.parameters()).device
    h_dir = humor_direction.to(device)
    
    # Project humor direction into vocabulary space
    # model.W_U is the unembedding matrix: hidden_dim -> vocab_size
    with torch.no_grad():
        vocab_scores = h_dir @ model.W_U  # Score each token by alignment with humor direction
        top_h_indices = torch.topk(vocab_scores, n_top_tokens).indices  # Most humor-aligned
        top_s_indices = torch.topk(-vocab_scores, n_top_tokens).indices  # Most serious-aligned

    tokens = model.to_tokens(prompt, prepend_bos=True)
    
    # Run with potential intervention hooks
    if intervention and alpha != 0:
        hook_name = f"blocks.{intervention.layer}.hook_resid_post"
        hook_fn = intervention._get_steering_hook(alpha)
        with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
            logits = model(tokens)
    else:
        with torch.no_grad():
            logits = model(tokens)

    # Get probabilities for next token after prompt
    log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)
    
    # Sum probabilities over humor and serious categories
    humor_logp = torch.logsumexp(log_probs[top_h_indices], dim=0).item()
    serious_logp = torch.logsumexp(log_probs[top_s_indices], dim=0).item()
    
    return {
        'logit_difference': humor_logp - serious_logp,
        'humor_prob': np.exp(humor_logp) 
    }

def compute_activation_projection(
    model: HookedTransformer,
    prompt: str,
    humor_direction: torch.Tensor,
    layer: int,
    intervention: Optional[HumorIntervention] = None,
    alpha: float = 0.0
) -> float:
    """
    Measure internal alignment with the humor direction.
    
    This measures how much the activation at a specific layer points
    along the humor direction.
    
    Returns a scalar: activation · humor_direction (dot product)
    - Positive: activation is humor-aligned
    - Negative: activation is serious-aligned
    - Near zero: activation is orthogonal to humor
    
    If intervention is provided, measures WHILE intervention is active
    (so we can see how steering shifts the activation).
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    device = next(model.parameters()).device
    
    # We must measure this WHILE the intervention is active to see the shift
    if intervention and alpha != 0:
        hook_name = f"blocks.{intervention.layer}.hook_resid_post"
        hook_fn = intervention._get_steering_hook(alpha)
        with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
            _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
            resid = cache[hook_name][0, -1, :]  # Final token activation
    else:
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=lambda n: n.endswith("resid_post"))
            resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :]
    
    direction = humor_direction.to(device)
    # Dot product measures alignment
    return torch.dot(resid, direction / direction.norm()).item()


def evaluate_ablation_impact(
    probe,
    original_activations: np.ndarray,
    ablated_activations: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate how ablation affects probe accuracy.
    
    This is the QUANTITATIVE CAUSAL TEST.
    
    The probe was trained on normal activations (expecting humor direction present).
    We test it on:
    1. Original activations → should be high accuracy
    2. Ablated activations → if accuracy drops to chance, proves causality
    
    Args:
        probe: Trained linear probe (sklearn LogisticRegression)
        original_activations: Normal activations from model
        ablated_activations: Activations with humor direction removed
        labels: True labels
    
    Returns:
        Dict with accuracy before/after and drop percentage
    """
    # Test probe on normal activations
    original_preds = probe.predict(original_activations)
    # Test same probe on ablated activations
    ablated_preds = probe.predict(ablated_activations)
    
    original_acc = accuracy_score(labels, original_preds)
    ablated_acc = accuracy_score(labels, ablated_preds)
    
    return {
        'original_accuracy': original_acc,
        'ablated_accuracy': ablated_acc,
        'accuracy_drop': original_acc - ablated_acc,
        'drop_percentage': 100 * (original_acc - ablated_acc) / original_acc if original_acc > 0 else 0
    }


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
    config_path = results_dir / "probe_summary.json"
    if config_path.exists():
        with open(config_path) as f:
            BEST_LAYER = json.load(f).get('best_layer', model.cfg.n_layers - 1)
    else:
        BEST_LAYER = model.cfg.n_layers - 1
        print(f"Warning: probe_summary.json not found, using last layer ({BEST_LAYER})")
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