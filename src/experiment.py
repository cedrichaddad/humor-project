#!/usr/bin/env python3
"""
Main experiment script for investigating low-rank humor representations in LLMs.

Research Question: Is there a low-rank linear basis for humor recognition
in LLM hidden representations?

Methodology:
1. Extract activations from LLM for humor/non-humor texts
2. Train linear probes at each layer
3. Analyze rank via PCA
4. Validate with multiple probing methods

Supports multiple models (Gemma-2 2B, GPT-2, etc.) via MODEL_NAME config.
"""

import os
import json
import random
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
from transformer_lens import HookedTransformer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

SEED = 42
MODEL_NAME = "gemma-2-2b"  # Default model; override via set_model() or run_experiment(model_name=...)
BATCH_SIZE = 32
MAX_LENGTH = 128
N_SAMPLES = 10000
RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"

# Display name mapping for plot titles
MODEL_DISPLAY_NAMES = {
    "gemma-2-2b": "Gemma-2 2B",
    "gpt2": "GPT-2 Small",
}

def set_model(model_name: str):
    """Set the global model name and update output directories accordingly."""
    global MODEL_NAME, RESULTS_DIR, FIGURES_DIR
    MODEL_NAME = model_name
    RESULTS_DIR = Path(f"results/{model_name}")
    FIGURES_DIR = RESULTS_DIR / "figures"

def get_display_name(model_name: str = None) -> str:
    """Get human-readable model name for plot titles."""
    name = model_name or MODEL_NAME
    return MODEL_DISPLAY_NAMES.get(name, name)

def set_seed(seed: int = SEED):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =============================================================================
# Data Loading
# =============================================================================

def load_unfun_dataset(n_samples: int = None) -> Dict[str, Dict]:
    """
    Load Dataset A: Aligned humor/serious pairs from local Excel file.
    
    THIS IS THE MAIN DATASET used to train probes and learn humor directions.
    
    Structure: Each "pair" has two rows:
    - One humorous version
    - One serious version (same semantic content, different style)
    
    Example pair:
    - Humor: "I'm not lazy, I'm just on energy-saving mode"
    - Serious: "I prefer to conserve my energy when possible"
    
    Split strategy: Split by pair_id to prevent data leakage
    (ensures humor/serious versions of the same content never cross train/test boundary)
    
    Returns:
        Dict with train, val, test splits (each containing texts, labels, pair_ids)
    """
    print("Loading Dataset A (aligned pairs) from local file...")

    dataset_path = Path("datasets/dataset_a_paired.xlsx")
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset A not found at {dataset_path}. "
            "Please generate dataset_a_paired.xlsx with pair_id, text, humor."
        )

    df = pd.read_excel(dataset_path)

    # Basic cleaning
    required_cols = {"pair_id", "text", "humor"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset A is missing columns: {missing}. Need {required_cols}.")

    df = df.dropna(subset=["pair_id", "text", "humor"]).copy()
    df["text"] = df["text"].astype(str)
    df["humor"] = df["humor"].astype(bool)

    # Keep only complete pairs (exactly 2 rows: one humor True, one False)
    g = df.groupby("pair_id")
    df = df[g["humor"].transform("nunique") == 2]  # has both classes
    df = df[g["text"].transform("size") == 2]      # exactly 2 rows per pair

    # Optional subsample by number of pairs (not rows)
    pair_ids = df["pair_id"].drop_duplicates().sample(frac=1, random_state=SEED).tolist()
    if n_samples is not None:
        # Interpret n_samples as "number of pairs"
        pair_ids = pair_ids[: min(n_samples, len(pair_ids))]
        df = df[df["pair_id"].isin(pair_ids)].copy()

    # Split by pair_id (CRITICAL: prevents data leakage)
    # If pair #42 is in train, BOTH its humor and serious versions are in train
    n_pairs = len(pair_ids)
    train_pairs = int(0.8 * n_pairs)
    val_pairs = int(0.1 * n_pairs)

    train_pair_ids = set(pair_ids[:train_pairs])
    val_pair_ids   = set(pair_ids[train_pairs:train_pairs + val_pairs])
    test_pair_ids  = set(pair_ids[train_pairs + val_pairs:])

    def make_split(pids: set) -> Dict[str, List]:
        """Create a split from a set of pair_ids."""
        sub = df[df["pair_id"].isin(pids)].copy()
        # Shuffle rows within split for batching, but pairs remain inside split
        sub = sub.sample(frac=1, random_state=SEED).reset_index(drop=True)
        return {
            "texts": sub["text"].tolist(),
            "labels": [1 if h else 0 for h in sub["humor"].tolist()],
            "pair_ids": sub["pair_id"].tolist(),
        }

    splits = {
        "train": make_split(train_pair_ids),
        "val":   make_split(val_pair_ids),
        "test":  make_split(test_pair_ids),
    }

    print(f"  Loaded {len(df)} rows = {n_pairs} pairs from {dataset_path}")
    print("Dataset A splits (PAIR-WISE):")
    for split_name, split_data in splits.items():
        humor_count = sum(split_data["labels"])
        total = len(split_data["labels"])
        n_unique_pairs = len(set(split_data["pair_ids"]))
        print(f"  {split_name}: {humor_count}/{total} humor ({100*humor_count/total:.1f}%) | pairs={n_unique_pairs}")

    # Safety check: no pair_id overlap across splits
    assert set(splits["train"]["pair_ids"]).isdisjoint(set(splits["val"]["pair_ids"]))
    assert set(splits["train"]["pair_ids"]).isdisjoint(set(splits["test"]["pair_ids"]))
    assert set(splits["val"]["pair_ids"]).isdisjoint(set(splits["test"]["pair_ids"]))

    return splits


def load_randomized_dataset(n_samples: int = None) -> Dict[str, Dict]:
    """
    Load Dataset B: Randomized (not aligned) humor/non-humor samples.
    
    This is used for comparison to Dataset A to test if the humor direction
    learned from aligned pairs generalizes to non-paired data.
    
    Unlike Dataset A:
    - No paired structure
    - Just random humor and non-humor texts
    - Tests whether findings are specific to aligned pairs or general
    
    Returns:
        Dict with train, val, test splits (each containing texts and labels)
    """
    print("Loading Dataset B (randomized) from local file...")
    
    dataset_path = Path("datasets/dataset_b_randomized.xlsx")
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset B not found at {dataset_path}. "
            "Please ensure the randomized dataset file exists."
        )
    
    df = pd.read_excel(dataset_path)
    
    df.dropna(subset=['text'], inplace=True)
    df['text'] = df['text'].astype(str)
    
    print(f"  Loaded {len(df)} samples from {dataset_path}")
    
    # Shuffle all rows
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    if n_samples and n_samples < len(df):
        df = df.head(n_samples)
    
    texts = df['text'].tolist()
    labels = [1 if h else 0 for h in df['humor'].tolist()]
    
    # Simple 80/10/10 split (not pair-wise since no pairs)
    n = len(texts)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    
    splits = {
        'train': {
            'texts': texts[:train_size],
            'labels': labels[:train_size]
        },
        'val': {
            'texts': texts[train_size:train_size + val_size],
            'labels': labels[train_size:train_size + val_size]
        },
        'test': {
            'texts': texts[train_size + val_size:],
            'labels': labels[train_size + val_size:]
        }
    }
    
    print(f"Dataset B splits:")
    for split_name, split_data in splits.items():
        humor_count = sum(split_data['labels'])
        print(f"  {split_name}: {humor_count}/{len(split_data['labels'])} humor ({100*humor_count/len(split_data['labels']):.1f}%)")
    
    return splits


# =============================================================================
# Activation Extraction
# =============================================================================

def load_model(model_name: str = None) -> Tuple[HookedTransformer, torch.device]:
    """
    Load model with TransformerLens.
    
    TransformerLens provides hooks for accessing internal activations,
    which is essential for mechanistic interpretability.
    """
    name = model_name or MODEL_NAME
    print(f"\nLoading {name} model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = HookedTransformer.from_pretrained(name, device=device)
    model.eval()  # Set to evaluation mode (no dropout, etc.)

    print(f"Model loaded: {name}")
    print(f"  Layers: {model.cfg.n_layers}")
    print(f"  Hidden dim: {model.cfg.d_model}")
    print(f"  Vocabulary: {model.cfg.d_vocab}")

    return model, device


def extract_activations(
    model: HookedTransformer,
    texts: List[str],
    device: torch.device,
    batch_size: int = BATCH_SIZE,
    layer_hook: str = "resid_post"
) -> Dict[int, np.ndarray]:
    """
    Extract hidden state activations from all layers.
    
    This is the core function for getting the model's internal representations.
    
    Process:
    1. Tokenize text
    2. Run forward pass
    3. Capture activations at each layer
    4. Extract activation at final token position (following Tigges et al. 2023)
    
    Why final token? 
    - For text classification, the final token typically contains the most
      task-relevant information after processing the entire sequence
    - This is standard practice in transformer interpretability
    
    Returns:
        Dict mapping layer_idx -> numpy array of shape (n_samples, hidden_dim)
        Each row is one text's representation at that layer
    """
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    # Initialize storage for all layers
    all_activations = {layer: [] for layer in range(n_layers)}

    print(f"\nExtracting activations from {len(texts)} samples...")

    # Process in batches for memory efficiency
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
        batch_texts = texts[i:i+batch_size]

        # Tokenize with BOS (beginning of sequence) token
        tokens = model.to_tokens(batch_texts, prepend_bos=True)

        # Calculate actual sequence lengths (excluding padding)
        seq_lengths = (tokens != model.tokenizer.pad_token_id).sum(dim=1) - 1

        # Run forward pass and cache all activations
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        # Extract final token activation from each layer
        for layer in range(n_layers):
            hook_name = f"blocks.{layer}.hook_{layer_hook}"
            activations = cache[hook_name]  # Shape: (batch, seq_len, hidden_dim)

            # Get activation at final non-padding token for each sample
            final_acts = []
            for j in range(len(batch_texts)):
                final_pos = min(seq_lengths[j].item(), activations.shape[1] - 1)
                final_acts.append(activations[j, final_pos, :].cpu().numpy())

            all_activations[layer].extend(final_acts)

        # Clean up to save memory
        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Convert lists to numpy arrays
    for layer in range(n_layers):
        all_activations[layer] = np.array(all_activations[layer])
        print(f"Layer {layer}: {all_activations[layer].shape}")

    return all_activations

# =============================================================================
# Linear Probing
# =============================================================================

def train_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    regularization: float = 1.0
) -> Dict:
    """
    Train a logistic regression probe and evaluate.
    
    A linear probe tests if a representation is linearly separable:
    - If classes can be separated by a linear boundary, they're linearly separable
    - High accuracy means the representation encodes the concept (humor) clearly
    
    The probe's weight vector becomes our "humor direction" - the direction
    in activation space that most distinguishes humor from non-humor.
    
    Returns:
        Dict containing:
        - accuracy, f1, auc: performance metrics
        - weights: raw probe coefficients
        - normalized_weights: unit-length humor direction
        - probe: the trained sklearn model
    """
    probe = LogisticRegression(
        C=1.0/regularization,  # Inverse of regularization strength
        max_iter=1000,
        random_state=SEED,
        solver='lbfgs'
    )

    probe.fit(X_train, y_train)

    y_pred = probe.predict(X_test)
    y_prob = probe.predict_proba(X_test)[:, 1]

    # Extract the humor direction (probe's decision boundary normal)
    weights = probe.coef_[0]  # Points from non-humor toward humor (sklearn convention)
    normalized_weights = weights / np.linalg.norm(weights)

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_prob),
        'weights': weights,
        'normalized_weights': normalized_weights,  # This is our "humor direction"
        'bias': probe.intercept_[0],
        'probe': probe
    }


def compute_mean_difference_direction(
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """
    Compute the mean difference direction: μ_humor - μ_non_humor.
    
    This is an alternative way to find a separating direction:
    - Calculate average activation for humor samples
    - Calculate average activation for non-humor samples
    - The difference points from non-humor toward humor
    
    Simpler than a trained probe, but often gives similar results.
    Used to validate that the probe is finding a sensible direction.
    """
    humor_mask = y == 1
    mean_humor = X[humor_mask].mean(axis=0)
    mean_non_humor = X[~humor_mask].mean(axis=0)

    direction = mean_humor - mean_non_humor
    direction = direction / np.linalg.norm(direction)  # Normalize to unit length

    return direction


def probe_all_layers(
    activations_train: Dict[int, np.ndarray],
    activations_test: Dict[int, np.ndarray],
    y_train: np.ndarray,
    y_test: np.ndarray
) -> pd.DataFrame:
    """
    Train probes at each layer and collect results.
    
    This answers: "At which layer is humor most clearly represented?"
    
    For each layer, we:
    1. Train a linear probe (logistic regression)
    2. Compute mean difference direction
    3. Test a random direction (baseline)
    4. Measure cosine similarity between probe and mean difference
    
    If probe and mean difference directions are similar (high cosine similarity),
    it suggests the representation is cleanly linearly separable.
    
    Returns:
        DataFrame with one row per layer containing all metrics
    """
    results = []

    n_layers = len(activations_train)
    print(f"\nTraining probes at {n_layers} layers...")

    for layer in tqdm(range(n_layers), desc="Probing"):
        X_train = activations_train[layer]
        X_test = activations_test[layer]

        # Method 1: Trained linear probe
        probe_result = train_linear_probe(X_train, y_train, X_test, y_test)

        # Method 2: Mean difference direction (simpler baseline)
        mean_dir = compute_mean_difference_direction(X_train, y_train)

        # Test mean difference method
        projections = X_test @ mean_dir
        mean_proj = projections.mean()
        y_pred_mean = (projections > mean_proj).astype(int)
        mean_acc = accuracy_score(y_test, y_pred_mean)

        # Compare probe direction with mean difference direction
        probe_dir = probe_result['weights'] / np.linalg.norm(probe_result['weights'])
        cosine_sim = np.abs(np.dot(probe_dir, mean_dir))

        # Method 3: Random direction (sanity check - should be ~50% accuracy)
        random_dir = np.random.randn(X_train.shape[1])
        random_dir = random_dir / np.linalg.norm(random_dir)
        random_proj = X_test @ random_dir
        y_pred_random = (random_proj > random_proj.mean()).astype(int)
        random_acc = accuracy_score(y_test, y_pred_random)

        results.append({
            'layer': layer,
            'probe_accuracy': probe_result['accuracy'],
            'probe_f1': probe_result['f1'],
            'probe_auc': probe_result['auc'],
            'mean_diff_accuracy': mean_acc,
            'random_accuracy': random_acc,
            'direction_cosine_sim': cosine_sim
        })

    return pd.DataFrame(results)

# =============================================================================
# Rank Analysis (PCA)
# =============================================================================

def analyze_humor_direction_rank(
    layer_directions: Dict,
    save_path: Path
) -> Dict:
    """
    PCA over per-layer humor directions to test the low-rank hypothesis.

    Stacks the normalized probe direction from every layer into a matrix of
    shape (n_layers, d_model) and runs PCA on it.  If humor is encoded in a
    consistent low-rank subspace across layers — i.e. the directions are
    nearly parallel — a small number of principal components will capture
    most of the variance and the cumulative-explained-variance curve will
    rise sharply before flattening.

    This directly answers the central research question:
      "Is there a low-rank linear basis for humor recognition?"

    A steep early rise (PC1 alone explaining >50 %) supports a single
    dominant humor direction that is reused across depth.  A slow rise
    suggests the model re-encodes humor in geometrically distinct ways at
    different layers.

    Args:
        layer_directions: Dict mapping layer_idx -> {'direction': torch.Tensor, ...}
                          as built in Step 3b of run_experiment().
        save_path:        Where to write the figure PNG.

    Returns:
        Dict with summary statistics:
          - pc1_variance      : fraction of variance explained by PC1
          - n_dims_90pct      : number of dims needed to reach 90 % cumvar
          - cumulative_variance: full list of cumulative variance values (%)
    """
    # Stack normalized probe directions: (n_layers, d_model)
    directions = np.stack([
        layer_directions[i]['direction'].numpy()
        for i in sorted(layer_directions.keys())
    ])

    pca = PCA(random_state=SEED)
    pca.fit(directions)

    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    n_components = np.arange(1, len(cumvar) + 1)

    # How many dims needed to reach 90 %?
    n_dims_90 = int(np.searchsorted(cumvar, 90.0)) + 1

    display_name = get_display_name()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(n_components, cumvar, 'b-o', linewidth=2, markersize=5,
            label='Cumulative explained variance')
    ax.axhline(y=90, color='r', linestyle='--', linewidth=1.5,
               label='90 % threshold')
    ax.axvline(x=n_dims_90, color='r', linestyle=':', linewidth=1.5,
               label=f'{n_dims_90} dim{"s" if n_dims_90 > 1 else ""} → 90 % var')

    ax.set_xlabel('Number of Principal Components', fontsize=12)
    ax.set_ylabel('Cumulative Explained Variance (%)', fontsize=12)
    ax.set_title(
        f'Rank of Humor Direction Subspace Across Layers ({display_name})',
        fontsize=13
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(cumvar))
    ax.set_ylim(0, 105)

    # Annotate PC1 variance in-plot
    ax.annotate(
        f"PC1: {pca.explained_variance_ratio_[0]*100:.1f}%",
        xy=(1, cumvar[0]),
        xytext=(max(2, len(cumvar) // 8), cumvar[0] - 12),
        arrowprops=dict(arrowstyle='->', color='gray'),
        fontsize=10,
        color='gray'
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_path}")
    print(f"  PC1 explains:          {pca.explained_variance_ratio_[0]*100:.1f}% of cross-layer variance")
    print(f"  Dims needed for 90 %:  {n_dims_90}")

    return {
        'pc1_variance': float(pca.explained_variance_ratio_[0]),
        'n_dims_90pct': n_dims_90,
        'cumulative_variance': cumvar.tolist(),
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_probe_accuracy_by_layer(results_df: pd.DataFrame, save_path: Path):
    """
    Plot probe accuracy across layers.
    
    Shows which layers have the clearest humor representation.
    Typically middle-to-late layers perform best.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    display_name = get_display_name()

    ax.plot(results_df['layer'], results_df['probe_accuracy'],
            'b-o', label='Logistic Probe', linewidth=2, markersize=8)
    ax.plot(results_df['layer'], results_df['mean_diff_accuracy'],
            'g-s', label='Mean Difference', linewidth=2, markersize=8)
    ax.axhline(y=0.5, color='r', linestyle='--', label='Random', linewidth=2)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title(f'Humor Classification Accuracy by Layer ({display_name})', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_direction_similarity(results_df: pd.DataFrame, save_path: Path):
    """
    Plot cosine similarity between probing methods across layers.
    
    If probe direction and mean-difference direction are similar,
    it suggests clean linear separability.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    display_name = get_display_name()

    ax.plot(results_df['layer'], results_df['direction_cosine_sim'],
            'purple', marker='o', linewidth=2, markersize=8)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title(f'Probe vs Mean Difference Direction ({display_name})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(model_name: str = None):
    """
    Run the full experiment pipeline.
    
    This is the MAIN PROBING EXPERIMENT that:
    1. Loads Dataset A (aligned humor/serious pairs)
    2. Extracts activations from all layers
    3. Trains linear probes at each layer
    3b. Extracts layer-specific humor directions
    4. Rank analysis via PCA over layer directions
    5. Compares with Dataset B (randomized)
    6. Saves all results and visualizations
    
    The humor_direction learned here is used by intervention_tests.py
    for causal experiments (steering and ablation).
    
    Args:
        model_name: Model to use (e.g. "gemma-2-2b", "gpt2").
                    If None, uses the global MODEL_NAME.
    
    Returns:
        Dict with config, probe results, rank analysis, and best layer index
    """
    if model_name:
        set_model(model_name)
    
    set_seed(SEED)

    # Create output directories (model-specific)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    display_name = get_display_name()

    config = {
        'seed': SEED,
        'model': MODEL_NAME,
        'model_display_name': display_name,
        'n_samples': N_SAMPLES,
        'batch_size': BATCH_SIZE,
        'max_length': MAX_LENGTH,
        'timestamp': datetime.now().isoformat()
    }

    print("="*60)
    print(f"Experiment: Low-Rank Humor Recognition ({display_name})")
    print("="*60)
    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # =========================================================================
    # Step 1: Load Data (Dataset A - Aligned Pairs)
    # =========================================================================
    print("\n" + "="*60)
    print("Step 1: Loading Data")
    print("="*60)

    # Load Dataset A: aligned humor/serious pairs
    # This is the MAIN dataset used throughout the experiment
    splits_a = load_unfun_dataset()
    
    train_texts = splits_a['train']['texts']
    train_labels = np.array(splits_a['train']['labels'])
    
    test_texts = splits_a['test']['texts']
    test_labels = np.array(splits_a['test']['labels'])

    # Show example samples
    humor_indices = np.where(train_labels == 1)[0]
    non_humor_indices = np.where(train_labels == 0)[0]
    if len(humor_indices) > 0:
        print("\nSample humor text:", train_texts[humor_indices[0]])
    if len(non_humor_indices) > 0:
        print("Sample non-humor text:", train_texts[non_humor_indices[0]])

    # =========================================================================
    # Step 2: Load Model and Extract Activations
    # =========================================================================
    print("\n" + "="*60)
    print("Step 2: Extracting Activations")
    print("="*60)

    model, device = load_model()

    # Extract activations at ALL layers for train and test sets
    # This is just forward passes - no intervention, just capturing internal states
    train_activations = extract_activations(model, train_texts, device)
    test_activations = extract_activations(model, test_texts, device)

    # =========================================================================
    # Step 3: Linear Probing
    # =========================================================================
    print("\n" + "="*60)
    print("Step 3: Linear Probing")
    print("="*60)

    # Train probes at every layer to find where humor is most clearly represented
    probe_results = probe_all_layers(
        train_activations, test_activations,
        train_labels, test_labels
    )

    print("\nProbe Results Summary:")
    print(probe_results.to_string(index=False))

    # Find the best performing layer
    best_layer = probe_results.loc[probe_results['probe_accuracy'].idxmax()]
    print(f"\nBest layer: {int(best_layer['layer'])}")
    print(f"  Accuracy: {best_layer['probe_accuracy']:.4f}")
    print(f"  F1: {best_layer['probe_f1']:.4f}")
    print(f"  AUC: {best_layer['probe_auc']:.4f}")

    best_layer_idx = int(best_layer['layer'])
    
    # =========================================================================
    # Step 3b: Extract Layer-Specific Humor Directions
    # =========================================================================
    print("\n" + "="*60)
    print("Step 3b: Extracting Layer-Specific Humor Directions")
    print("="*60)
    
    # Create directions subdirectory
    directions_dir = RESULTS_DIR / "directions"
    directions_dir.mkdir(parents=True, exist_ok=True)
    
    layer_directions = {}
    
    # For each layer, train a probe and extract its direction
    # This gives us layer-specific humor directions for steering experiments
    for layer_idx in range(model.cfg.n_layers):
        print(f"\nLayer {layer_idx}...")
        
        # Train probe for this layer (reuse if it's the best layer)
        if layer_idx == best_layer_idx:
            layer_probe_result = train_linear_probe(
                train_activations[layer_idx],
                train_labels,
                test_activations[layer_idx],
                test_labels
            )
            best_probe_result = layer_probe_result  # Save for later use
        else:
            layer_probe_result = train_linear_probe(
                train_activations[layer_idx],
                train_labels,
                test_activations[layer_idx],
                test_labels
            )
        
        # Extract the probe's weight vector as the humor direction.
        # The probe's coef_[0] points from non-humor toward humor by sklearn
        # convention.
        # Whether steering along this direction causally increases humor
        # output is an empirical question tested in intervention_tests.py.
        # Negative steering effects at some layers reveal a mismatch between
        # discriminative and generative roles, which is itself interesting.
        probe = layer_probe_result['probe']
        direction = torch.tensor(
            layer_probe_result['normalized_weights'], 
            dtype=torch.float32
        )

        # Record representational separation for metadata
        decisions = probe.decision_function(test_activations[layer_idx])
        humor_decisions = decisions[test_labels == 1]
        serious_decisions = decisions[test_labels == 0]
        mean_humor_proj = np.mean(humor_decisions)
        mean_serious_proj = np.mean(serious_decisions)
        
        # Save this layer's direction (probe direction, unmodified)
        direction_path = directions_dir / f"layer{layer_idx}.pt"
        torch.save(direction, direction_path)
        
        layer_directions[layer_idx] = {
            'direction': direction,
            'accuracy': layer_probe_result['accuracy'],
            'mean_humor_proj': float(mean_humor_proj),
            'mean_serious_proj': float(mean_serious_proj),
            'separation': abs(mean_humor_proj - mean_serious_proj),
        }
        
        print(f"  Saved to: {direction_path.name}")
        print(f"  Accuracy: {layer_probe_result['accuracy']:.4f}")
        print(f"  Separation: {abs(mean_humor_proj - mean_serious_proj):.4f}")
        print(f"  Humor proj: {mean_humor_proj:.4f}, Serious proj: {mean_serious_proj:.4f}")
    
    # Save summary of all layer directions
    directions_summary = {
        str(layer): {
            'accuracy': float(info['accuracy']),
            'separation': float(info['separation']),
            'mean_humor_proj': info['mean_humor_proj'],
            'mean_serious_proj': info['mean_serious_proj'],
        }
        for layer, info in layer_directions.items()
    }
    
    with open(directions_dir / "summary.json", 'w') as f:
        json.dump(directions_summary, f, indent=2)
    
    print(f"\nLayer-specific directions saved!")
    print(f"  Location: {directions_dir}/")
    print(f"  Summary: {directions_dir / 'summary.json'}")
    
    # Also save the best layer direction as "humor_direction.pt" for backward compatibility
    # This is the direction used in intervention_tests.py
    humor_direction = layer_directions[best_layer_idx]['direction']
    humor_direction_path = RESULTS_DIR / "humor_direction.pt"
    torch.save(humor_direction, humor_direction_path)
    print(f"\nBest layer ({best_layer_idx}) direction also saved to: {humor_direction_path.name}")
    print(f"  Direction shape: {humor_direction.shape}")
    print(f"  Direction norm: {humor_direction.norm().item():.4f}")
    
    # Save a copy in the directions folder too
    torch.save(humor_direction, directions_dir / f"best_layer{best_layer_idx}.pt")
    print(f"  Also saved as: {directions_dir / f'best_layer{best_layer_idx}.pt'}")
    
    best_probe = best_probe_result['probe']

    # =========================================================================
    # Step 4: Rank Analysis (PCA over layer directions)
    # =========================================================================
    print("\n" + "="*60)
    print("Step 4: Rank Analysis")
    print("="*60)

    rank_results = analyze_humor_direction_rank(
        layer_directions,
        save_path=FIGURES_DIR / "pca_rank_analysis.png"
    )

    # =========================================================================
    # Step 5: Dataset B (Randomized) Comparison
    # =========================================================================
    print("\n" + "="*60)
    print("Step 5: Dataset B (Randomized) Comparison")
    print("="*60)
    
    # Load Dataset B for comparison
    splits_b = load_randomized_dataset()
    
    train_texts_b = splits_b['train']['texts']
    train_labels_b = np.array(splits_b['train']['labels'])
    test_texts_b = splits_b['test']['texts']
    test_labels_b = np.array(splits_b['test']['labels'])
    
    print("\nExtracting activations for Dataset B...")
    train_activations_b = extract_activations(model, train_texts_b, device)
    test_activations_b = extract_activations(model, test_texts_b, device)
    
    # Train probe on Dataset B at the same layer
    dataset_b_probe_result = train_linear_probe(
        train_activations_b[best_layer_idx],
        train_labels_b,
        test_activations_b[best_layer_idx],
        test_labels_b
    )
    
    print(f"\nDataset Comparison at Layer {best_layer_idx}:")
    print(f"  Dataset A (aligned pairs): {best_probe_result['accuracy']:.4f} accuracy")
    print(f"  Dataset B (randomized):    {dataset_b_probe_result['accuracy']:.4f} accuracy")
    
    # Compare humor directions learned from A and B
    cos_sim_ab = np.abs(np.dot(
        best_probe_result['normalized_weights'],
        dataset_b_probe_result['normalized_weights']
    ))
    print(f"  Cosine similarity (A vs B directions): {cos_sim_ab:.4f}")
    
    dataset_comparison = {
        'dataset_a_accuracy': float(best_probe_result['accuracy']),
        'dataset_b_accuracy': float(dataset_b_probe_result['accuracy']),
        'accuracy_difference': float(best_probe_result['accuracy'] - dataset_b_probe_result['accuracy']),
        'direction_cosine_similarity': float(cos_sim_ab),
        'layer': best_layer_idx
    }

    # =========================================================================
    # Step 6: Generate Visualizations
    # =========================================================================
    print("\n" + "="*60)
    print("Step 6: Generating Visualizations")
    print("="*60)

    plot_probe_accuracy_by_layer(probe_results, FIGURES_DIR / "probe_accuracy_by_layer.png")
    plot_direction_similarity(probe_results, FIGURES_DIR / "direction_similarity.png")
    # Note: pca_rank_analysis.png already saved in Step 4

    # =========================================================================
    # Step 7: Save Results
    # =========================================================================
    print("\n" + "="*60)
    print("Step 7: Saving Results")
    print("="*60)

    with open(RESULTS_DIR / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    probe_results.to_csv(RESULTS_DIR / "probe_results.csv", index=False)

    probe_summary = {
        'best_layer': best_layer_idx,
        'best_probe_accuracy': float(best_layer['probe_accuracy']),
        'best_probe_f1': float(best_layer['probe_f1']),
        'best_probe_auc': float(best_layer['probe_auc']),
    }

    with open(RESULTS_DIR / "probe_summary.json", 'w') as f:
        json.dump(probe_summary, f, indent=2)

    with open(RESULTS_DIR / "dataset_comparison.json", 'w') as f:
        json.dump(dataset_comparison, f, indent=2)

    with open(RESULTS_DIR / "rank_analysis.json", 'w') as f:
        json.dump(rank_results, f, indent=2)

    print(f"\nResults saved to: {RESULTS_DIR}/")
    print(f"Figures saved to: {FIGURES_DIR}/")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print(f"EXPERIMENT COMPLETE ({display_name})")
    print("="*60)

    print("\nKey Findings:")
    print(f"1. Best linear probe accuracy: {best_layer['probe_accuracy']:.1%} at layer {best_layer_idx}")
    print(f"   (Random baseline: 50%)")
    print(f"2. Average cosine similarity between layers: {probe_results['direction_cosine_sim'].mean():.4f}")
    print(f"3. Humor/non-humor are linearly separable: {'Yes' if best_layer['probe_accuracy'] > 0.6 else 'Partially'}")
    print(f"4. Humor direction rank: {rank_results['n_dims_90pct']} dim(s) explain 90% of variance across layers")
    print(f"   (PC1 alone: {rank_results['pc1_variance']:.1%}) — {'low-rank' if rank_results['n_dims_90pct'] <= 3 else 'distributed'} representation")

    return {
        'config': config,
        'probe_results': probe_results,
        'probe_summary': probe_summary,
        'rank_results': rank_results,
        'best_layer': best_layer_idx
    }


if __name__ == "__main__":
    results = run_experiment()