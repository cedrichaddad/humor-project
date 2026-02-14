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
    global MODEL_NAME, RESULTS_DIR, FIGURES_DIR
    MODEL_NAME = model_name
    RESULTS_DIR = Path(f"results/{model_name}")
    FIGURES_DIR = RESULTS_DIR / "figures"

def get_display_name(model_name: str = None) -> str:
    """Get human-readable model name for plot titles."""
    name = model_name or MODEL_NAME
    return MODEL_DISPLAY_NAMES.get(name, name)

def set_seed(seed: int = SEED):
    """Set random seeds for reproducibility."""
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
    Load Dataset A: Aligned humor/serious pairs from local Excel file,
    and split *by pair_id* so pairs never leak across train/val/test.
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

    # Now do *pair-wise* split
    n_pairs = len(pair_ids)
    train_pairs = int(0.8 * n_pairs)
    val_pairs = int(0.1 * n_pairs)

    train_pair_ids = set(pair_ids[:train_pairs])
    val_pair_ids   = set(pair_ids[train_pairs:train_pairs + val_pairs])
    test_pair_ids  = set(pair_ids[train_pairs + val_pairs:])

    def make_split(pids: set) -> Dict[str, List]:
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
    
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    if n_samples and n_samples < len(df):
        df = df.head(n_samples)
    
    texts = df['text'].tolist()
    labels = [1 if h else 0 for h in df['humor'].tolist()]
    
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
    """
    print("Loading humor subtype dataset...")
    dataset = load_dataset("CreativeLang/ColBERT_Humor_Detection")
    
    humor_samples = [ex['text'] for ex in dataset['train'] if ex['humor']]
    random.shuffle(humor_samples)
    
    subtypes = {
        'puns': [],
        'sarcasm': [],
        'irony': [],
        'other': []
    }
    
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
    
    max_per_subtype = n_samples // 3
    result = {}
    
    for subtype, texts in subtypes.items():
        if subtype == 'other':
            continue
        
        limited_texts = texts[:max_per_subtype]
        result[subtype] = {
            'texts': limited_texts,
            'labels': [1] * len(limited_texts),
            'count': len(limited_texts)
        }
        print(f"  {subtype}: {len(limited_texts)} samples")
    
    return result

# =============================================================================
# Activation Extraction
# =============================================================================

def load_model(model_name: str = None) -> Tuple[HookedTransformer, torch.device]:
    """Load model with TransformerLens."""
    name = model_name or MODEL_NAME
    print(f"\nLoading {name} model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = HookedTransformer.from_pretrained(name, device=device)
    model.eval()

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

    We extract the activation at the final token position, following the
    methodology from the sentiment paper (Tigges et al. 2023).
    """
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    all_activations = {layer: [] for layer in range(n_layers)}

    print(f"\nExtracting activations from {len(texts)} samples...")

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
        batch_texts = texts[i:i+batch_size]

        tokens = model.to_tokens(batch_texts, prepend_bos=True)

        seq_lengths = (tokens != model.tokenizer.pad_token_id).sum(dim=1) - 1

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        for layer in range(n_layers):
            hook_name = f"blocks.{layer}.hook_{layer_hook}"
            activations = cache[hook_name]

            final_acts = []
            for j in range(len(batch_texts)):
                final_pos = min(seq_lengths[j].item(), activations.shape[1] - 1)
                final_acts.append(activations[j, final_pos, :].cpu().numpy())

            all_activations[layer].extend(final_acts)

        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    """
    probe = LogisticRegression(
        C=1.0/regularization,
        max_iter=1000,
        random_state=SEED,
        solver='lbfgs'
    )

    probe.fit(X_train, y_train)

    y_pred = probe.predict(X_test)
    y_prob = probe.predict_proba(X_test)[:, 1]

    weights = probe.coef_[0]
    normalized_weights = weights / np.linalg.norm(weights)

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_prob),
        'weights': weights,
        'normalized_weights': normalized_weights,
        'bias': probe.intercept_[0],
        'probe': probe
    }


def compute_mean_difference_direction(
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """
    Compute the mean difference direction: μ_humor - μ_non_humor.
    """
    humor_mask = y == 1
    mean_humor = X[humor_mask].mean(axis=0)
    mean_non_humor = X[~humor_mask].mean(axis=0)

    direction = mean_humor - mean_non_humor
    direction = direction / np.linalg.norm(direction)

    return direction


def probe_all_layers(
    activations_train: Dict[int, np.ndarray],
    activations_test: Dict[int, np.ndarray],
    y_train: np.ndarray,
    y_test: np.ndarray
) -> pd.DataFrame:
    """
    Train probes at each layer and collect results.
    """
    results = []

    n_layers = len(activations_train)
    print(f"\nTraining probes at {n_layers} layers...")

    for layer in tqdm(range(n_layers), desc="Probing"):
        X_train = activations_train[layer]
        X_test = activations_test[layer]

        probe_result = train_linear_probe(X_train, y_train, X_test, y_test)

        mean_dir = compute_mean_difference_direction(X_train, y_train)

        projections = X_test @ mean_dir
        mean_proj = projections.mean()
        y_pred_mean = (projections > mean_proj).astype(int)
        mean_acc = accuracy_score(y_test, y_pred_mean)

        probe_dir = probe_result['weights'] / np.linalg.norm(probe_result['weights'])
        cosine_sim = np.abs(np.dot(probe_dir, mean_dir))

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
    def __init__(self, model: HookedTransformer, humor_direction: torch.Tensor, layer: int = 7):
        self.model = model
        self.layer = layer
        self.device = next(model.parameters()).device
        
        if isinstance(humor_direction, np.ndarray):
            humor_direction = torch.from_numpy(humor_direction)
        self.humor_direction = humor_direction.float().to(self.device)
        self.humor_direction = self.humor_direction / self.humor_direction.norm()
        
    def _get_steering_hook(self, alpha: float, direction: torch.Tensor = None):
        """Create hook function for steering."""
        target_dir = direction if direction is not None else self.humor_direction
        
        def hook_fn(activation, hook):
            steering = alpha * target_dir
            return activation + steering.view(1, 1, -1)
        return hook_fn
    
    def _get_ablation_hook(self, direction: torch.Tensor = None):
        """Create hook function for ablation."""
        target_dir = direction if direction is not None else self.humor_direction
        
        def hook_fn(activation, hook):
            v = target_dir
            proj_coef = torch.einsum('bsd,d->bs', activation, v)
            projection = proj_coef.unsqueeze(-1) * v
            return activation - projection
        return hook_fn
    
    def steer_humor(self, prompt: str, alpha: float = 1.0, max_new_tokens: int = 50, temperature: float = 1.0) -> str:
        """Steer using the stored global humor direction."""
        return self.steer_direction(self.humor_direction, prompt, alpha, max_new_tokens, temperature)

    def ablate_humor(self, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0) -> str:
        """Ablate using the stored global humor direction."""
        return self.ablate_direction(self.humor_direction, prompt, max_new_tokens, temperature)

    def steer_direction(
        self,
        direction: torch.Tensor,
        prompt: str,
        alpha: float = 1.0, 
        max_new_tokens: int = 50, 
        temperature: float = 1.0
    ) -> str:
        """Generate text with steering applied along an arbitrary direction."""
        hook_name = f"blocks.{self.layer}.hook_resid_post"
        hook_fn = self._get_steering_hook(alpha, direction=direction)
        
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
    
    def ablate_direction(
        self,
        direction: torch.Tensor,
        prompt: str,
        max_new_tokens: int = 50, 
        temperature: float = 1.0
    ) -> str:
        """Generate text with arbitrary direction ablated."""
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
        hook_name = f"blocks.{self.layer}.hook_resid_post"
        ablation_hook = self._get_ablation_hook()
        
        all_ablated_acts = []

        def capture_hook(activation, hook):
            all_ablated_acts.append(activation.detach().cpu())
            return activation

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            tokens = self.model.to_tokens(batch_texts, prepend_bos=True)
            seq_lengths = (tokens != self.model.tokenizer.pad_token_id).sum(dim=1) - 1
            
            with self.model.hooks(fwd_hooks=[(hook_name, ablation_hook), (hook_name, capture_hook)]):
                self.model(tokens)
            
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
    humor_tokens: List[str] = None,
    serious_tokens: List[str] = None,
    intervention: Optional[HumorIntervention] = None,
    alpha: float = 0.0
) -> Dict[str, float]:
    """Compute logit difference between humor and serious tokens."""
    if humor_tokens is None:
        humor_tokens = ["haha", "lol", "funny", "joke", "hilarious", "laugh"]
    if serious_tokens is None:
        serious_tokens = ["however", "therefore", "thus", "indeed", "importantly", "specifically"]
    
    humor_ids = [model.to_tokens(t, prepend_bos=False)[0, 0].item() for t in humor_tokens]
    serious_ids = [model.to_tokens(t, prepend_bos=False)[0, 0].item() for t in serious_tokens]
    
    tokens = model.to_tokens(prompt, prepend_bos=True)
    
    if intervention and alpha != 0:
        hook_name = f"blocks.{intervention.layer}.hook_resid_post"
        hook_fn = intervention._get_steering_hook(alpha)
        with torch.no_grad():
            with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
                logits = model(tokens)
    else:
        with torch.no_grad():
            logits = model(tokens)
    
    final_logits = logits[0, -1, :]
    
    humor_logit = final_logits[humor_ids].mean().item()
    serious_logit = final_logits[serious_ids].mean().item()
    
    return {
        'humor_logit': humor_logit,
        'serious_logit': serious_logit,
        'logit_difference': humor_logit - serious_logit
    }


def evaluate_ablation_impact(
    probe,
    original_activations: np.ndarray,
    ablated_activations: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """Evaluate how ablation affects probe accuracy."""
    original_preds = probe.predict(original_activations)
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
    """Train linear probes for each humor subtype and compute cosine similarities."""
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
        
        n_samples = min(len(texts), len(non_humor_texts))
        combined_texts = texts[:n_samples] + non_humor_texts[:n_samples]
        labels = np.array([1] * n_samples + [0] * n_samples)
        
        activations = extract_activations(model, combined_texts, device, batch_size=32)
        X = activations[layer]
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=SEED, stratify=labels
        )
        
        result = train_linear_probe(X_train, y_train, X_test, y_test)
        
        subtype_directions[subtype] = result['normalized_weights']
        subtype_results[subtype] = {
            'accuracy': result['accuracy'],
            'f1': result['f1'],
            'n_samples': n_samples
        }
        print(f"  Accuracy: {result['accuracy']:.3f}, F1: {result['f1']:.3f}")
    
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
    """Evaluate cross-subtype generalization."""
    from sklearn.model_selection import train_test_split
    
    print("\n" + "="*60)
    print("Cross-Subtype Generalization Evaluation")
    print("="*60)
    
    dataset = load_dataset("CreativeLang/ColBERT_Humor_Detection")
    non_humor_texts = [ex['text'] for ex in dataset['train'] if not ex['humor']][:2000]
    
    subtypes = list(subtype_data.keys())
    n_subtypes = len(subtypes)
    
    subtype_activations = {}
    for subtype, data in subtype_data.items():
        print(f"Extracting activations for {subtype}...")
        texts = data['texts'][:500]
        activations = extract_activations(model, texts, device, batch_size=32)
        subtype_activations[subtype] = activations[layer]
    
    print("Extracting non-humor activations...")
    non_humor_activations = extract_activations(model, non_humor_texts[:500], device, batch_size=32)
    X_non_humor = non_humor_activations[layer]
    
    cross_accuracy = np.zeros((n_subtypes, n_subtypes))
    
    for i, train_subtype in enumerate(subtypes):
        X_pos = subtype_activations[train_subtype]
        n = min(len(X_pos), len(X_non_humor))
        
        X_train = np.vstack([X_pos[:n], X_non_humor[:n]])
        y_train = np.array([1]*n + [0]*n)
        
        idx = np.random.permutation(len(X_train))
        X_train, y_train = X_train[idx], y_train[idx]
        
        probe_result = train_linear_probe(
            X_train[:int(0.8*len(X_train))], y_train[:int(0.8*len(y_train))],
            X_train[int(0.8*len(X_train)):], y_train[int(0.8*len(y_train)):]
        )
        probe = probe_result['probe']
        
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
        'diagonal_mean': np.diag(cross_accuracy).mean(),
        'off_diagonal_mean': (cross_accuracy.sum() - np.trace(cross_accuracy)) / (n_subtypes * (n_subtypes - 1)) if n_subtypes > 1 else 0
    }


# =============================================================================
# Rank Analysis (PCA)

def analyze_rank(
    X: np.ndarray,
    y: np.ndarray,
    max_components: int = 100
) -> Dict:
    """Analyze the effective rank of humor representation using PCA."""
    n_components = min(max_components, X.shape[1], X.shape[0])

    pca = PCA(n_components=n_components, random_state=SEED)
    X_pca = pca.fit_transform(X)

    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    rank_90 = np.searchsorted(cumulative_var, 0.90) + 1
    rank_95 = np.searchsorted(cumulative_var, 0.95) + 1
    rank_99 = np.searchsorted(cumulative_var, 0.99) + 1

    accs_by_rank = []
    ranks_to_test = [1, 2, 3, 5, 10, 20, 50, 100]
    ranks_to_test = [r for r in ranks_to_test if r <= n_components]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for r in ranks_to_test:
        X_reduced = X_pca[:, :r]
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
    """Analyze how humor and non-humor classes separate in activation space."""
    humor_mask = y == 1
    X_humor = X[humor_mask]
    X_non_humor = X[~humor_mask]

    mean_humor = X_humor.mean(axis=0)
    mean_non_humor = X_non_humor.mean(axis=0)

    between_class = mean_humor - mean_non_humor
    between_norm = np.linalg.norm(between_class)

    var_humor = np.var(X_humor, axis=0).mean()
    var_non_humor = np.var(X_non_humor, axis=0).mean()
    within_var = (var_humor + var_non_humor) / 2

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
    """Plot probe accuracy across layers."""
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
    """Plot rank analysis results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    display_name = get_display_name()

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
    """Plot cosine similarity between probing methods across layers."""
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
    """Plot 2D PCA projection of activations."""
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
    
    Args:
        model_name: Model to use (e.g. "gpt2", "gemma-2-2b", "gemma-2-2b").
                    If None, uses the global MODEL_NAME.
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

    splits_a = load_unfun_dataset()
    
    train_texts = splits_a['train']['texts']
    train_labels = np.array(splits_a['train']['labels'])
    
    test_texts = splits_a['test']['texts']
    test_labels = np.array(splits_a['test']['labels'])

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

    train_activations = extract_activations(model, train_texts, device)
    test_activations = extract_activations(model, test_texts, device)

    # =========================================================================
    # Step 3: Linear Probing
    # =========================================================================
    print("\n" + "="*60)
    print("Step 3: Linear Probing")
    print("="*60)

    probe_results = probe_all_layers(
        train_activations, test_activations,
        train_labels, test_labels
    )

    print("\nProbe Results Summary:")
    print(probe_results.to_string(index=False))

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
            best_probe_result = layer_probe_result  # Save for later
        else:
            layer_probe_result = train_linear_probe(
                train_activations[layer_idx],
                train_labels,
                test_activations[layer_idx],
                test_labels
            )
        
        # Get the probe and use its decision function
        probe = layer_probe_result['probe']
        direction = torch.tensor(
            layer_probe_result['normalized_weights'], 
            dtype=torch.float32
        )

        # CRITICAL: Use probe's decision function on TEST data
        decisions = probe.decision_function(test_activations[layer_idx])
        humor_decisions = decisions[test_labels == 1]
        serious_decisions = decisions[test_labels == 0]

        mean_humor_proj = np.mean(humor_decisions)
        mean_serious_proj = np.mean(serious_decisions)
        
        needs_flip = mean_humor_proj > mean_serious_proj
        if needs_flip:
            print(f"  WARNING: Flipping direction (humor={mean_humor_proj:.3f} < serious={mean_serious_proj:.3f})")
            direction = -direction
        else:
            print(f"  OK: Direction correct (humor={mean_humor_proj:.3f} > serious={mean_serious_proj:.3f})")
        
        # Save this layer's direction
        direction_path = directions_dir / f"layer{layer_idx}.pt"
        torch.save(direction, direction_path)
        
        layer_directions[layer_idx] = {
            'direction': direction,
            'accuracy': layer_probe_result['accuracy'],
            'mean_humor_proj': mean_humor_proj if not needs_flip else -mean_humor_proj,
            'mean_serious_proj': mean_serious_proj if not needs_flip else -mean_serious_proj,
            'separation': abs(mean_humor_proj - mean_serious_proj),
            'needs_flip': needs_flip
        }
        
        print(f"  Saved to: {direction_path.name}")
        print(f"  Accuracy: {layer_probe_result['accuracy']:.4f}")
        print(f"  Separation: {abs(mean_humor_proj - mean_serious_proj):.4f}")
    
    # Save summary of all layer directions
    directions_summary = {
        str(layer): {
            'accuracy': float(info['accuracy']),
            'separation': float(info['separation']),
            'flipped': bool(info['needs_flip'])
        }
        for layer, info in layer_directions.items()
    }
    
    with open(directions_dir / "summary.json", 'w') as f:
        json.dump(directions_summary, f, indent=2)
    
    print(f"\nLayer-specific directions saved!")
    print(f"  Location: {directions_dir}/")
    print(f"  Summary: {directions_dir / 'summary.json'}")
    
    # Also save the best layer direction as "humor_direction.pt" for backward compatibility
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

    X_all = np.vstack([train_activations[best_layer_idx],
                       test_activations[best_layer_idx]])
    y_all = np.concatenate([train_labels, test_labels])

    rank_results = analyze_rank(X_all, y_all)

    print(f"\nRank Analysis at Layer {best_layer_idx}:")
    print(f"  Rank for 90% variance: {rank_results['rank_90']}")
    print(f"  Rank for 95% variance: {rank_results['rank_95']}")
    print(f"  Rank for 99% variance: {rank_results['rank_99']}")

    print("\nAccuracy vs. Rank:")
    print(rank_results['accuracy_by_rank'].to_string(index=False))

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
    
    splits_b = load_randomized_dataset()
    
    train_texts_b = splits_b['train']['texts']
    train_labels_b = np.array(splits_b['train']['labels'])
    test_texts_b = splits_b['test']['texts']
    test_labels_b = np.array(splits_b['test']['labels'])
    
    print("\nExtracting activations for Dataset B...")
    train_activations_b = extract_activations(model, train_texts_b, device)
    test_activations_b = extract_activations(model, test_texts_b, device)
    
    dataset_b_probe_result = train_linear_probe(
        train_activations_b[best_layer_idx],
        train_labels_b,
        test_activations_b[best_layer_idx],
        test_labels_b
    )
    
    print(f"\nDataset Comparison at Layer {best_layer_idx}:")
    print(f"  Dataset A (aligned pairs): {best_probe_result['accuracy']:.4f} accuracy")
    print(f"  Dataset B (randomized):    {dataset_b_probe_result['accuracy']:.4f} accuracy")
    
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