
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import pandas as pd
from transformer_lens import HookedTransformer

try:
    from sae_lens import SAE
except ImportError:
    print("Warning: sae-lens not installed. Please install with `pip install sae-lens`.")
    SAE = None

def load_sae_model(
    layer: int = 11,
    release: str = "gpt2-small-res-jb",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Load a pre-trained Sparse Autoencoder (SAE).
    
    Args:
        layer: Layer number in GPT-2 small (default: 11)
        release: Name of the SAE release (default: 'gpt2-small-res-jb')
        device: Device to load model on
        
    Returns:
        sae: The loaded SAE model
    """
    if SAE is None:
        raise ImportError("sae-lens is required. Install it using `pip install sae-lens`")

    sae_id = f"blocks.{layer}.hook_resid_post"
    print(f"Loading SAE: {release}, id: {sae_id}...")
    
    sae, _, _ = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device
    )
    
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
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # 1. Get model activations
        # We need the residual stream activations at the target layer
        if hasattr(sae.cfg, "hook_name"):
            hook_name = sae.cfg.hook_name
        elif hasattr(sae.cfg, "hook_point"):
            hook_name = sae.cfg.hook_point
        else:
            # Fallback for some versions or just use the one passed to loader if stored
            # Assuming GPT-2 Small Layer 11 Residual Post
            hook_name = "blocks.11.hook_resid_post"
            print(f"Warning: Could not find hook_name in sae.cfg. Using default: {hook_name}")
        
        # Tokenize
        tokens = model.to_tokens(batch_texts, prepend_bos=True)
        if tokens.shape[1] > max_length:
            tokens = tokens[:, :max_length]
            
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=[hook_name]
            )
            model_acts = cache[hook_name]  # (batch, seq, d_model)
            
            # 2. Run SAE
            # Encode: (batch, seq, d_model) -> (batch, seq, d_sae)
            feature_acts = sae.encode(model_acts)
            
            # 3. Aggregation: Max over sequence length
            # We want to know if a feature activated *anywhere* in the text
            # This is robust for variable length jokes
            max_acts, _ = feature_acts.max(dim=1)  # (batch, d_sae)
            
            all_acts.append(max_acts.cpu())
            
    return torch.cat(all_acts, dim=0)

def identify_humor_features(
    feature_acts: torch.Tensor,
    labels: List[int],
    top_k: int = 10
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
    
    # Difference
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
            'unfun_mean': unfun_mean[idx].item()
        })
        
    return pd.DataFrame(results)

def analyze_feature_interpretation(
    sae,
    feature_idx: int,
    k_tokens: int = 10
) -> List[Tuple[str, float]]:
    """
    Interpret a feature using the 'Logit Lens' technique.
    
    Look at the SAE decoder weights for this feature and project them
    into the vocabulary space to see which tokens they promote.
    """
    # Decoder weight: (d_sae, d_model) or similar 
    # Check shape: sae.W_dec might be (n_features, d_model)
    # sae-lens standard is usually sae.W_dec[feature_idx] -> (d_model,)
    
    feature_dir = sae.W_dec[feature_idx]
    
    # We rely on the SAE usually having access to the model's unembedding
    # But simpler is to assumes we just have the direction. 
    # To do full logit lens we need the base model's W_U.
    # Since we don't pass the base model here, we returns None or need to pass model.
    # Wait, better to pass model or assume user calls this with knowledge.
    
    # Let's adjust to take model as argument if we want tokens.
    # Or return the vector.
    pass

def analyze_features_with_model(
    model: HookedTransformer,
    sae,
    feature_indices: List[int],
    k: int = 10
) -> Dict[int, List[str]]:
    """
    Get top tokens promoted by each feature using model's unembedding matrix.
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
