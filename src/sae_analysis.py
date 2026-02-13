#!/usr/bin/env python3
"""
Sparse Autoencoder (SAE) Analysis for Humor Features.

This script uses pre-trained SAEs to identify interpretable features
that correspond to humor detection in GPT-2.

Note: SAE analysis currently only supports GPT-2 small.
"""
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformer_lens import HookedTransformer

warnings.filterwarnings('ignore')

try:
    from sae_lens import SAE
except ImportError:
    print("Warning: sae-lens not installed. Install with: pip install sae-lens")
    SAE = None

# Import from experiment.py for consistency
from experiment import set_seed, SEED, set_model, get_display_name


def get_results_dir():
    """Get current model's results directory."""
    from experiment import RESULTS_DIR
    return RESULTS_DIR


def get_figures_dir():
    """Get current model's figures directory."""
    from experiment import FIGURES_DIR
    return FIGURES_DIR


def load_sae_model(
    layer: int = 7,
    release: str = "gpt2-small-res-jb",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Load a pre-trained Sparse Autoencoder (SAE).
    
    Args:
        layer: Layer number in GPT-2 small (default: 7, the best probing layer)
        release: Name of the SAE release (default: 'gpt2-small-res-jb')
        device: Device to load model on
        
    Returns:
        sae: The loaded SAE model
    """
    if SAE is None:
        raise ImportError("sae-lens is required. Install it using: pip install sae-lens")

    sae_id = f"blocks.{layer}.hook_resid_post"
    print(f"Loading SAE: {release}, layer {layer}, id: {sae_id}...")
    
    sae, _, _ = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device
    )
    
    print(f"  SAE loaded successfully")
    print(f"  Number of features: {sae.cfg.d_sae}")
    print(f"  Model dimension: {sae.cfg.d_in}")
    
    return sae


def extract_sae_features(
    model: HookedTransformer,
    sae,
    texts: List[str],
    batch_size: int = 32,
    max_length: int = 128
) -> torch.Tensor:
    """
    Extract SAE feature activations for a list of texts.
    
    Returns:
        activations: Tensor of shape (n_samples, n_features)
                     (Max activation across sequence position for each feature)
    """
    all_acts = []
    
    sae.eval()  # Ensure SAE is in eval mode
    
    # Get the hook name from SAE config
    if hasattr(sae.cfg, "hook_name"):
        hook_name = sae.cfg.hook_name
    elif hasattr(sae.cfg, "hook_point"):
        hook_name = sae.cfg.hook_point
    else:
        # Fallback - extract layer from ID
        hook_name = "blocks.7.hook_resid_post"  # Default for GPT-2 small layer 7
        print(f"Warning: Could not find hook_name in sae.cfg. Using default: {hook_name}")
    
    print(f"\nExtracting SAE features from {len(texts)} samples...")
    print(f"Using hook: {hook_name}")
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting SAE features"):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        tokens = model.to_tokens(batch_texts, prepend_bos=True)
        if tokens.shape[1] > max_length:
            tokens = tokens[:, :max_length]
            
        with torch.no_grad():
            # Get model activations
            _, cache = model.run_with_cache(
                tokens,
                names_filter=[hook_name]
            )
            model_acts = cache[hook_name]  # (batch, seq, d_model)
            
            # Run SAE
            # Encode: (batch, seq, d_model) -> (batch, seq, d_sae)
            feature_acts = sae.encode(model_acts)
            
            # Aggregation: Max over sequence length
            # We want to know if a feature activated *anywhere* in the text
            # This is robust for variable length jokes
            max_acts, _ = feature_acts.max(dim=1)  # (batch, d_sae)
            
            all_acts.append(max_acts.cpu())
    
    return torch.cat(all_acts, dim=0)


def identify_humor_features(
    feature_acts: torch.Tensor,
    labels: List[int],
    top_k: int = 20
) -> pd.DataFrame:
    """
    Identify features that differentiate between Humor (1) and Non-Humor (0).
    
    Args:
        feature_acts: (n_samples, n_features) tensor
        labels: List of binary labels
        top_k: Number of top features to return
        
    Returns:
        DataFrame with feature stats
    """
    labels = torch.tensor(labels, device=feature_acts.device)
    
    # Split by class
    humor_mask = (labels == 1)
    
    humor_acts = feature_acts[humor_mask]
    unfun_acts = feature_acts[~humor_mask]
    
    # Compute means
    humor_mean = humor_acts.mean(dim=0)
    unfun_mean = unfun_acts.mean(dim=0)
    
    # Difference (how much more this feature fires for humor)
    diff = humor_mean - unfun_mean
    
    # Find top K features favored by humor
    top_vals, top_indices = torch.topk(diff, k=top_k)
    
    results = []
    for i in range(top_k):
        idx = top_indices[i].item()
        results.append({
            'feature_idx': idx,
            'diff': top_vals[i].item(),
            'humor_mean': humor_mean[idx].item(),
            'unfun_mean': unfun_mean[idx].item(),
            'ratio': (humor_mean[idx] / (unfun_mean[idx] + 1e-8)).item()
        })
        
    return pd.DataFrame(results)


def analyze_features_with_model(
    model: HookedTransformer,
    sae,
    feature_indices: List[int],
    k: int = 10
) -> Dict[int, List[str]]:
    """
    Get top tokens promoted by each feature using model's unembedding matrix.
    
    This helps interpret what each SAE feature represents.
    """
    W_U = model.W_U  # (d_model, vocab)
    
    results = {}
    for idx in feature_indices:
        # Feature direction: (d_model,)
        feature_dir = sae.W_dec[idx]
        
        # Project to logits
        logits = feature_dir @ W_U
        
        # Top tokens
        top_vals, top_ids = torch.topk(logits, k=k)
        
        tokens = [model.to_string(t) for t in top_ids]
        results[idx] = tokens
        
    return results


def run_sae_analysis(model_name: str = "gpt2", layer: int = 7, n_samples: int = 5000):
    """
    Run complete SAE analysis for humor features.
    
    Args:
        model_name: Model to use (only "gpt2" supported for SAE analysis)
        layer: Layer to analyze
        n_samples: Number of samples to use
    """
    if model_name != "gpt2":
        print(f"Warning: SAE analysis currently only supports GPT-2 small.")
        print(f"Requested model '{model_name}' - switching to 'gpt2'")
        model_name = "gpt2"
    
    # Set up directories
    set_model(model_name)
    set_seed(SEED)
    results_dir = get_results_dir()
    figures_dir = get_figures_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    display_name = get_display_name()
    
    print("="*60)
    print(f"SAE Analysis for Humor Features ({display_name})")
    print("="*60)
    
    # Check if sae-lens is available
    if SAE is None:
        print("\nError: sae-lens not installed.")
        print("Install with: pip install sae-lens")
        return None
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading model...")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()
    
    # Load SAE
    print(f"\nLoading SAE for layer {layer}...")
    sae = load_sae_model(layer=layer, device=str(device))
    
    # Load data
    print(f"\nLoading humor dataset...")
    dataset = load_dataset("CreativeLang/ColBERT_Humor_Detection")
    data = dataset['train'].shuffle(seed=SEED).select(range(n_samples))
    
    texts = [ex['text'] for ex in data]
    labels = [1 if ex['humor'] else 0 for ex in data]
    
    print(f"  Total samples: {len(texts)}")
    print(f"  Humor: {sum(labels)} ({100*sum(labels)/len(labels):.1f}%)")
    print(f"  Non-humor: {len(labels)-sum(labels)} ({100*(len(labels)-sum(labels))/len(labels):.1f}%)")
    
    # Extract SAE features
    print(f"\nExtracting SAE features...")
    feature_acts = extract_sae_features(model, sae, texts, batch_size=32)
    print(f"  Feature activations shape: {feature_acts.shape}")
    
    # Identify humor-specific features
    print(f"\nIdentifying top humor features...")
    top_features = identify_humor_features(feature_acts, labels, top_k=20)
    
    print("\n" + "="*60)
    print("TOP HUMOR-SPECIFIC FEATURES")
    print("="*60)
    print(top_features.to_string(index=False))
    
    # Interpret features using logit lens
    print(f"\nInterpreting features...")
    feature_indices = top_features['feature_idx'].tolist()[:10]  # Top 10
    feature_tokens = analyze_features_with_model(model, sae, feature_indices, k=10)
    
    print("\n" + "="*60)
    print("FEATURE INTERPRETATION (Top Promoted Tokens)")
    print("="*60)
    for feat_idx, tokens in feature_tokens.items():
        print(f"\nFeature {feat_idx}:")
        print(f"  Tokens: {', '.join(tokens)}")
    
    # Save results
    results = {
        'model': model_name,
        'layer': layer,
        'n_samples': n_samples,
        'n_features': int(sae.cfg.d_sae),
        'top_humor_features': top_features.to_dict('records'),
        'feature_interpretations': {
            int(k): v for k, v in feature_tokens.items()
        }
    }
    
    results_path = results_dir / "sae_analysis.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n" + "="*60)
    print(f"SAE ANALYSIS COMPLETE ({display_name})")
    print("="*60)
    print(f"\nResults saved to: {results_path}")
    
    # Create visualization
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    top_10 = top_features.head(10)
    ax.barh(range(len(top_10)), top_10['diff'], color='steelblue')
    ax.set_yticks(range(len(top_10)))
    ax.set_yticklabels([f"Feature {idx}" for idx in top_10['feature_idx']])
    ax.set_xlabel('Mean Activation Difference (Humor - Non-Humor)')
    ax.set_title(f'Top 10 Humor-Specific SAE Features\n({display_name}, Layer {layer})')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    fig_path = figures_dir / 'sae_humor_features.png'
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    return results


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    layer = 7  # Default best layer for GPT-2
    n_samples = 5000  # Default sample size
    
    if len(sys.argv) > 1:
        layer = int(sys.argv[1])
    if len(sys.argv) > 2:
        n_samples = int(sys.argv[2])
    
    results = run_sae_analysis(model_name="gpt2", layer=layer, n_samples=n_samples)