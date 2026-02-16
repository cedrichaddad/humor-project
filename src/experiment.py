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

Supports multiple models (GPT-2, Gemma-2, etc.) via MODEL_NAME config.
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
from datasets import load_dataset, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
from transformer_lens import HookedTransformer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

SEED = 42
MODEL_NAME = "gpt2"  # Default model; override via set_model() or run_experiment(model_name=...)
BATCH_SIZE = 32
MAX_LENGTH = 128
N_SAMPLES = 10000
RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"

# Display name mapping for plot titles
MODEL_DISPLAY_NAMES = {
    "gpt2": "GPT-2 Small",
    "gemma-2-2b": "Gemma-2 2B",
    "gemma-2-2b": "Gemma-2 2B",
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

def load_humor_dataset(n_samples: int = N_SAMPLES) -> Dict[str, Dataset]:
    """
    Load the ColBERT Humor Detection dataset from HuggingFace.
    
    This dataset is NOT used for the main probing experiment - it's used
    in intervention_tests.py for the ablation experiment.
    
    The main experiment uses Dataset A (aligned pairs) from local Excel file.

    Returns:
        Dict with train, val, test splits
    """
    print("Loading ColBERT Humor Detection dataset...")
    dataset = load_dataset("CreativeLang/ColBERT_Humor_Detection")

    # Get the train split and shuffle
    data = dataset['train'].shuffle(seed=SEED)

    # Take a subset for efficiency
    if n_samples and n_samples < len(data):
        data = data.select(range(n_samples))

    # Create splits: 80% train, 10% val, 10% test
    n = len(data)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)

    splits = {
        'train': data.select(range(train_size)),
        'val': data.select(range(train_size, train_size + val_size)),
        'test': data.select(range(train_size + val_size, n))
    }

    print(f"Dataset splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    for split_name, split_data in splits.items():
        humor_count = sum(1 for ex in split_data if ex['humor'])
        print(f"  {split_name}: {humor_count}/{len(split_data)} humor ({100*humor_count/len(split_data):.1f}%)")

    return splits


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


def load_subtype_dataset(n_samples: int = 2000) -> Dict[str, Dict]:
    """
    Load humor samples classified by subtype (puns, sarcasm, irony).
    
    This is used for subtype analysis to see if different types of humor
    use the same underlying representation or different ones.
    
    Uses simple pattern matching to categorize humor types:
    - Puns: "What do you call...", "Why did the chicken..."
    - Sarcasm: "Oh great...", "Thanks for..."
    - Irony: "Except...", "Turns out..."
    
    Returns:
        Dict mapping subtype name to {texts, labels, count}
    """
    print("Loading humor subtype dataset...")
    dataset = load_dataset("CreativeLang/ColBERT_Humor_Detection")
    
    # Get only humor samples
    humor_samples = [ex['text'] for ex in dataset['train'] if ex['humor']]
    random.shuffle(humor_samples)
    
    subtypes = {
        'puns': [],
        'sarcasm': [],
        'irony': [],
        'other': []
    }
    
    # Pattern lists for categorization (simple heuristics)
    pun_patterns = [
        'what do you call', 'why did', 'why do', 'what did',
        'what\'s the difference', 'walked into a bar',
        'knock knock', 'how do you', 'what happens when'
    ]
    
    sarcasm_patterns = [
        'oh great', 'totally', 'definitely', 'absolutely',
        'my life', 'thanks for', 'love it when', 'best thing',
        'worst', 'nothing like', 'perfect', 'just great'
    ]
    
    irony_patterns = [
        'except', 'until', 'but then', 'turns out',
        'apparently', 'somehow', 'meanwhile', 'ironically'
    ]
    
    # Categorize each humor sample
    for text in humor_samples:
        text_lower = text.lower()
        
        if any(p in text_lower for p in pun_patterns):
            subtypes['puns'].append(text)
        elif any(p in text_lower for p in sarcasm_patterns):
            subtypes['sarcasm'].append(text)
        elif any(p in text_lower for p in irony_patterns):
            subtypes['irony'].append(text)
        else:
            subtypes['other'].append(text)
    
    # Limit samples per subtype for balanced analysis
    max_per_subtype = n_samples // 3
    result = {}
    
    for subtype, texts in subtypes.items():
        if subtype == 'other':
            continue
        
        limited_texts = texts[:max_per_subtype]
        result[subtype] = {
            'texts': limited_texts,
            'labels': [1] * len(limited_texts),  # All are humor
            'count': len(limited_texts)
        }
        print(f"  {subtype}: {len(limited_texts)} samples")
    
    return result

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
# Causal Interventions
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


def train_subtype_probes(
    model: HookedTransformer,
    subtype_data: Dict[str, Dict],
    layer: int,
    device: torch.device
) -> Dict[str, Dict]:
    """
    Train linear probes for each humor subtype.
    
    Tests whether different humor types (puns, sarcasm, irony) use
    the same representation or different ones.
    
    If cosine similarity between subtype directions is high,
    they share similar representations.
    
    Returns:
        Dict containing subtype directions, accuracies, and similarity matrix
    """
    subtype_directions = {}
    subtype_results = {}
    
    print("Loading non-humor samples for subtype probing...")
    dataset = load_dataset("CreativeLang/ColBERT_Humor_Detection")
    non_humor_texts = [ex['text'] for ex in dataset['train'] if not ex['humor']][:2000]
    
    for subtype, data in subtype_data.items():
        print(f"\nTraining probe for {subtype}...")
        
        texts = data['texts']
        if len(texts) < 50:
            print(f"  Skipping {subtype}: too few samples ({len(texts)})")
            continue
        
        # Create balanced dataset: subtype humor vs non-humor
        n_samples = min(len(texts), len(non_humor_texts))
        combined_texts = texts[:n_samples] + non_humor_texts[:n_samples]
        labels = np.array([1] * n_samples + [0] * n_samples)
        
        # Extract activations at specified layer
        activations = extract_activations(model, combined_texts, device, batch_size=32)
        X = activations[layer]
        
        # Train probe for this subtype
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=SEED, stratify=labels
        )
        
        result = train_linear_probe(X_train, y_train, X_test, y_test)
        
        # Save direction for this subtype
        subtype_directions[subtype] = result['normalized_weights']
        subtype_results[subtype] = {
            'accuracy': result['accuracy'],
            'f1': result['f1'],
            'n_samples': n_samples
        }
        print(f"  Accuracy: {result['accuracy']:.3f}, F1: {result['f1']:.3f}")
    
    # Compute pairwise cosine similarities between subtype directions
    subtypes = list(subtype_directions.keys())
    n = len(subtypes)
    similarity_matrix = np.zeros((n, n))
    
    for i, st1 in enumerate(subtypes):
        for j, st2 in enumerate(subtypes):
            sim = np.abs(np.dot(subtype_directions[st1], subtype_directions[st2]))
            similarity_matrix[i, j] = sim
    
    return {
        'directions': subtype_directions,
        'results': subtype_results,
        'similarity_matrix': similarity_matrix.tolist(),
        'subtype_names': subtypes
    }


def cross_subtype_evaluation(
    model: HookedTransformer,
    subtype_data: Dict[str, Dict],
    layer: int,
    device: torch.device
) -> Dict[str, Dict]:
    """
    Evaluate cross-subtype generalization.
    
    Train on one subtype, test on another.
    
    If a probe trained on puns can detect sarcasm, the representations
    are shared. If not, each subtype has its own representation.
    
    Returns:
        Dict with cross-accuracy matrix and summary statistics
    """
    from sklearn.model_selection import train_test_split
    
    print("\n" + "="*60)
    print("Cross-Subtype Generalization Evaluation")
    print("="*60)
    
    dataset = load_dataset("CreativeLang/ColBERT_Humor_Detection")
    non_humor_texts = [ex['text'] for ex in dataset['train'] if not ex['humor']][:2000]
    
    subtypes = list(subtype_data.keys())
    n_subtypes = len(subtypes)
    
    # Extract activations for each subtype
    subtype_activations = {}
    for subtype, data in subtype_data.items():
        print(f"Extracting activations for {subtype}...")
        texts = data['texts'][:500]
        activations = extract_activations(model, texts, device, batch_size=32)
        subtype_activations[subtype] = activations[layer]
    
    print("Extracting non-humor activations...")
    non_humor_activations = extract_activations(model, non_humor_texts[:500], device, batch_size=32)
    X_non_humor = non_humor_activations[layer]
    
    # Cross-evaluation matrix: rows = train subtype, cols = test subtype
    cross_accuracy = np.zeros((n_subtypes, n_subtypes))
    
    for i, train_subtype in enumerate(subtypes):
        # Train on this subtype
        X_pos = subtype_activations[train_subtype]
        n = min(len(X_pos), len(X_non_humor))
        
        X_train = np.vstack([X_pos[:n], X_non_humor[:n]])
        y_train = np.array([1]*n + [0]*n)
        
        # Shuffle training data
        idx = np.random.permutation(len(X_train))
        X_train, y_train = X_train[idx], y_train[idx]
        
        # Train probe
        probe_result = train_linear_probe(
            X_train[:int(0.8*len(X_train))], y_train[:int(0.8*len(y_train))],
            X_train[int(0.8*len(X_train)):], y_train[int(0.8*len(y_train)):]
        )
        probe = probe_result['probe']
        
        # Test on each subtype (including itself)
        for j, test_subtype in enumerate(subtypes):
            X_test_pos = subtype_activations[test_subtype]
            n_test = min(len(X_test_pos), 200)
            
            X_test = np.vstack([X_test_pos[:n_test], X_non_humor[:n_test]])
            y_test = np.array([1]*n_test + [0]*n_test)
            
            y_pred = probe.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            cross_accuracy[i, j] = acc
        
        print(f"  Train on {train_subtype}: " + 
              " | ".join([f"{subtypes[j]}: {cross_accuracy[i,j]:.2f}" for j in range(n_subtypes)]))
    
    return {
        'cross_accuracy_matrix': cross_accuracy.tolist(),
        'subtypes': subtypes,
        'diagonal_mean': np.diag(cross_accuracy).mean(),  # Average accuracy on same subtype
        'off_diagonal_mean': (cross_accuracy.sum() - np.trace(cross_accuracy)) / (n_subtypes * (n_subtypes - 1)) if n_subtypes > 1 else 0  # Average cross-subtype accuracy
    }


# =============================================================================
# Rank Analysis (PCA)
# =============================================================================

def analyze_rank(
    X: np.ndarray,
    y: np.ndarray,
    max_components: int = 100
) -> Dict:
    """
    Analyze the effective rank of humor representation using PCA.
    
    Question: How many dimensions are needed to represent humor?
    
    If humor is truly "low-rank", we should be able to:
    1. Explain most variance with few principal components
    2. Achieve good classification with few dimensions
    
    This tests the hypothesis that humor is a low-dimensional feature
    in the high-dimensional activation space.
    
    Returns:
        Dict containing:
        - Variance explained by each component
        - Ranks for 90%, 95%, 99% variance
        - Accuracy vs rank curve
    """
    n_components = min(max_components, X.shape[1], X.shape[0])

    # Fit PCA
    pca = PCA(n_components=n_components, random_state=SEED)
    X_pca = pca.fit_transform(X)

    # Variance explained
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    # Find rank needed for different variance thresholds
    rank_90 = np.searchsorted(cumulative_var, 0.90) + 1
    rank_95 = np.searchsorted(cumulative_var, 0.95) + 1
    rank_99 = np.searchsorted(cumulative_var, 0.99) + 1

    # Test classification accuracy at different ranks
    accs_by_rank = []
    ranks_to_test = [1, 2, 3, 5, 10, 20, 50, 100]
    ranks_to_test = [r for r in ranks_to_test if r <= n_components]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for r in ranks_to_test:
        # Use only top r principal components
        X_reduced = X_pca[:, :r]
        # Cross-validate a probe on these r dimensions
        scores = cross_val_score(
            LogisticRegression(max_iter=1000, random_state=SEED),
            X_reduced, y,
            cv=cv,
            scoring='accuracy'
        )
        accs_by_rank.append({
            'rank': r,
            'accuracy': scores.mean(),
            'std': scores.std()
        })

    return {
        'explained_variance': explained_var,
        'cumulative_variance': cumulative_var,
        'rank_90': int(rank_90),
        'rank_95': int(rank_95),
        'rank_99': int(rank_99),
        'accuracy_by_rank': pd.DataFrame(accs_by_rank),
        'pca_components': pca.components_
    }


def analyze_class_separation(
    X: np.ndarray,
    y: np.ndarray
) -> Dict:
    """
    Analyze how humor and non-humor classes separate in activation space.
    
    Metrics:
    - Between-class norm: Distance between class means (larger = more separated)
    - Within-class variance: How spread out each class is (smaller = more compact)
    - Fisher ratio: Between-class distance / within-class variance
      (larger = better separation)
    
    Good linear separability means:
    - Large between-class distance
    - Small within-class variance
    - High Fisher ratio
    """
    humor_mask = y == 1
    X_humor = X[humor_mask]
    X_non_humor = X[~humor_mask]

    # Calculate class means
    mean_humor = X_humor.mean(axis=0)
    mean_non_humor = X_non_humor.mean(axis=0)

    # Between-class separation
    between_class = mean_humor - mean_non_humor
    between_norm = np.linalg.norm(between_class)

    # Within-class variance (average across both classes)
    var_humor = np.var(X_humor, axis=0).mean()
    var_non_humor = np.var(X_non_humor, axis=0).mean()
    within_var = (var_humor + var_non_humor) / 2

    # Fisher ratio: signal-to-noise ratio for separation
    fisher_ratio = between_norm**2 / within_var if within_var > 0 else 0

    return {
        'between_class_norm': between_norm,
        'within_class_var': within_var,
        'fisher_ratio': fisher_ratio,
        'mean_humor': mean_humor,
        'mean_non_humor': mean_non_humor
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


def plot_rank_analysis(rank_results: Dict, save_path: Path):
    """
    Plot rank analysis results.
    
    Left panel: Variance explained by each component
    Right panel: Classification accuracy vs number of components
    
    Shows how many dimensions are really needed for humor.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    display_name = get_display_name()

    # Panel 1: Explained variance
    ax = axes[0]
    n_comp = len(rank_results['cumulative_variance'])
    ax.bar(range(1, min(21, n_comp+1)),
           rank_results['explained_variance'][:20],
           alpha=0.7, label='Individual')
    ax.plot(range(1, min(21, n_comp+1)),
            rank_results['cumulative_variance'][:20],
            'r-o', label='Cumulative', linewidth=2)
    ax.axhline(y=0.9, color='g', linestyle='--', alpha=0.7,
               label=f'90% (rank={rank_results["rank_90"]})')
    ax.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7,
               label=f'95% (rank={rank_results["rank_95"]})')
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax.set_title(f'PCA Explained Variance ({display_name})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 2: Accuracy vs rank
    ax = axes[1]
    acc_df = rank_results['accuracy_by_rank']
    ax.errorbar(acc_df['rank'], acc_df['accuracy'],
                yerr=acc_df['std'], fmt='b-o', capsize=4,
                linewidth=2, markersize=8)
    ax.axhline(y=0.5, color='r', linestyle='--', label='Random', linewidth=2)
    ax.set_xlabel('Number of PCA Components (Rank)', fontsize=12)
    ax.set_ylabel('5-Fold CV Accuracy', fontsize=12)
    ax.set_title(f'Accuracy vs. Rank ({display_name})', fontsize=14)
    ax.set_xscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

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


def plot_pca_2d(X: np.ndarray, y: np.ndarray, save_path: Path, title: str = ""):
    """
    Plot 2D PCA projection of activations.
    
    Visualizes how humor and non-humor samples separate
    in the top 2 principal components.
    
    If well-separated, shows clean linear separability.
    """
    pca = PCA(n_components=2, random_state=SEED)
    X_2d = pca.fit_transform(X)

    display_name = get_display_name()

    fig, ax = plt.subplots(figsize=(10, 8))

    humor_mask = y == 1
    ax.scatter(X_2d[~humor_mask, 0], X_2d[~humor_mask, 1],
               alpha=0.5, c='blue', label='Non-Humor', s=20)
    ax.scatter(X_2d[humor_mask, 0], X_2d[humor_mask, 1],
               alpha=0.5, c='red', label='Humor', s=20)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_title(f'2D PCA of Activations {title} ({display_name})', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

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
    4. Finds the best layer and extracts humor direction
    5. Analyzes rank (how many dimensions needed)
    6. Compares with Dataset B (randomized)
    7. Saves all results and visualizations
    
    The humor_direction learned here is used by intervention_tests.py
    for causal experiments (steering and ablation).
    
    Args:
        model_name: Model to use (e.g. "gpt2", "gemma-2-2b").
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
        # convention. We save it as-is without flipping.
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
    # Step 4: Rank Analysis
    # =========================================================================
    print("\n" + "="*60)
    print("Step 4: Rank Analysis")
    print("="*60)

    # Combine train and test for rank analysis
    X_all = np.vstack([train_activations[best_layer_idx],
                       test_activations[best_layer_idx]])
    y_all = np.concatenate([train_labels, test_labels])

    # Perform PCA and measure effective rank
    rank_results = analyze_rank(X_all, y_all)

    print(f"\nRank Analysis at Layer {best_layer_idx}:")
    print(f"  Rank for 90% variance: {rank_results['rank_90']}")
    print(f"  Rank for 95% variance: {rank_results['rank_95']}")
    print(f"  Rank for 99% variance: {rank_results['rank_99']}")

    print("\nAccuracy vs. Rank:")
    print(rank_results['accuracy_by_rank'].to_string(index=False))

    # Measure class separation
    separation = analyze_class_separation(X_all, y_all)
    print(f"\nClass Separation:")
    print(f"  Between-class norm: {separation['between_class_norm']:.4f}")
    print(f"  Within-class variance: {separation['within_class_var']:.4f}")
    print(f"  Fisher ratio: {separation['fisher_ratio']:.4f}")

    # =========================================================================
    # Step 4b: Dataset B (Randomized) Comparison
    # =========================================================================
    print("\n" + "="*60)
    print("Step 4b: Dataset B (Randomized) Comparison")
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
    # Step 5: Generate Visualizations
    # =========================================================================
    print("\n" + "="*60)
    print("Step 5: Generating Visualizations")
    print("="*60)

    plot_probe_accuracy_by_layer(probe_results, FIGURES_DIR / "probe_accuracy_by_layer.png")
    plot_rank_analysis(rank_results, FIGURES_DIR / "rank_analysis.png")
    plot_direction_similarity(probe_results, FIGURES_DIR / "direction_similarity.png")
    plot_pca_2d(X_all, y_all, FIGURES_DIR / f"pca_2d_layer{best_layer_idx}.png",
                title=f"(Layer {best_layer_idx})")

    # =========================================================================
    # Step 6: Save Results
    # =========================================================================
    print("\n" + "="*60)
    print("Step 6: Saving Results")
    print("="*60)

    with open(RESULTS_DIR / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    probe_results.to_csv(RESULTS_DIR / "probe_results.csv", index=False)

    rank_summary = {
        'best_layer': best_layer_idx,
        'best_probe_accuracy': float(best_layer['probe_accuracy']),
        'best_probe_f1': float(best_layer['probe_f1']),
        'best_probe_auc': float(best_layer['probe_auc']),
        'rank_90': rank_results['rank_90'],
        'rank_95': rank_results['rank_95'],
        'rank_99': rank_results['rank_99'],
        'between_class_norm': float(separation['between_class_norm']),
        'within_class_var': float(separation['within_class_var']),
        'fisher_ratio': float(separation['fisher_ratio']),
        'accuracy_by_rank': rank_results['accuracy_by_rank'].to_dict('records')
    }

    with open(RESULTS_DIR / "rank_analysis.json", 'w') as f:
        json.dump(rank_summary, f, indent=2)

    with open(RESULTS_DIR / "dataset_comparison.json", 'w') as f:
        json.dump(dataset_comparison, f, indent=2)

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
    print(f"2. Effective rank for 90% variance: {rank_results['rank_90']} dimensions")
    print(f"3. Effective rank for 95% variance: {rank_results['rank_95']} dimensions")
    print(f"4. Humor/non-humor are linearly separable: {'Yes' if best_layer['probe_accuracy'] > 0.6 else 'Partially'}")

    return {
        'config': config,
        'probe_results': probe_results,
        'rank_results': rank_summary,
        'best_layer': best_layer_idx
    }


if __name__ == "__main__":
    results = run_experiment()