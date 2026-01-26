#!/usr/bin/env python3
"""
Run SAE Experiment: Decomposing Humor with Sparse Autoencoders.

This script:
1. Loads GPT-2 Small and the Layer 11 SAE (gpt2-small-res-jb).
2. Extracts SAE features for Dataset A (Jokes vs Unfun).
3. Identifies top humor-correlated features.
4. Validates features via Steering and Ablation interventions.
5. Generates a report.
"""

import os
import json
import torch
import sys
import numpy as np
from pathlib import Path
from transformer_lens import HookedTransformer

# Add src to path
sys.path.append('src')

# Import modules
from experiment import (
    load_unfun_dataset,
    load_model,
    HumorIntervention,
    set_seed,
    SEED,
    RESULTS_DIR
)
from sae_analysis import (
    load_sae_model,
    extract_sae_features,
    identify_humor_features,
    analyze_features_with_model
)

def main():
    set_seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Model and SAE
    # Note: Previous experiments used Layer 7, but SAE is requested for Layer 11
    LAYER = 11
    print(f"\nPhase 1: Loading Models (Layer {LAYER})...")
    model, _ = load_model()
    sae = load_sae_model(layer=LAYER, device=device)
    
    # 2. Load Dataset
    print("\nPhase 2: Loading Data...")
    dataset = load_unfun_dataset(n_samples=2000)
    train_texts = dataset['train']['texts']
    train_labels = dataset['train']['labels']
    
    # 3. Feature Discovery
    print("\nPhase 3: Extracting SAE Features...")
    # Extract features for a subset to save time if needed, using 500 samples
    n_subset = 500
    subset_texts = train_texts[:n_subset]
    subset_labels = train_labels[:n_subset]
    
    feature_acts = extract_sae_features(model, sae, subset_texts, batch_size=16)
    print(f"  Feature activations shape: {feature_acts.shape}")
    
    # Identify top humor features
    print("  Identifying top humor features...")
    top_features_df = identify_humor_features(feature_acts, subset_labels, top_k=10)
    print("\nTop 10 Humor Features:")
    print(top_features_df)
    
    # Analyze interpretation (logit lens)
    top_indices = top_features_df['feature_idx'].tolist()
    interpretations = analyze_features_with_model(model, sae, top_indices, k=10)
    
    # 4. Causal Validation (Interventions)
    print("\nPhase 4: Causal Validation via Steering & Ablation")
    
    # Initialize intervention helper (humor_direction placeholder as we use features)
    # We can pass zeros or the SAE feature direction for init, but it expects a vector.
    # We will use the generic steer_direction method we added.
    dummy_direction = torch.zeros(768).to(device)
    intervention = HumorIntervention(model, dummy_direction, layer=LAYER)
    
    validation_results = []
    
    # Test top 3 features
    test_prompts = [
        "Why did the chicken cross the road?",
        "I told my friend a joke about",
        "The weather today is"
    ]
    
    for idx in top_indices[:3]:
        feature_dir = sae.W_dec[idx] # (d_model,)
        feature_tokens = interpretations[idx][:5]
        
        feature_result = {
            'feature_idx': idx,
            'top_tokens': feature_tokens,
            'steering': {},
            'ablation': {}
        }
        
        print(f"\nTesting Feature {idx} (Promotes: {feature_tokens})")
        
        # Steering
        for prompt in test_prompts:
            steered_text = intervention.steer_direction(
                feature_dir, prompt, alpha=50.0 # High alpha for SAE features usually needed?
                # Check sae-lens norm. Usually SAE features are unit norm or similar.
                # If they are unit norm, alpha=1.0 is small. 
                # Let's try explicit large value or maybe 20.
            )
            feature_result['steering'][prompt] = steered_text
            print(f"  Steer '{prompt}' -> {steered_text[-50:]}")
            
            # Ablation
            ablated_text = intervention.ablate_direction(feature_dir, prompt)
            feature_result['ablation'][prompt] = ablated_text
            
        validation_results.append(feature_result)
        
    # 5. Save Report
    output_path = RESULTS_DIR / "sae_features.json"
    report = {
        'top_features': top_features_df.to_dict(orient='records'),
        'interpretations': interpretations,
        'validation': validation_results
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"\nReport saved to {output_path}")

if __name__ == "__main__":
    main()
