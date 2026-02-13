#!/usr/bin/env python3
"""
Run the full humor experiment pipeline for multiple models.

Usage:
    python run_all_models.py                          # run both GPT-2 and Gemma-2-2B
    python run_all_models.py gpt2                     # run GPT-2 only
    python run_all_models.py gemma-2-2b               # run Gemma only
"""
import sys
import os

# Add src to path if it exists (for proper directory structure)
# Otherwise assume modules are in current directory (for Colab direct upload)
if os.path.exists('src'):
    sys.path.append('src')

from experiment import run_experiment
from intervention_tests import run_interventions

# Models to run (add/remove as needed)
DEFAULT_MODELS = ["gpt2", "gemma-2-2b"]

def main():
    if len(sys.argv) > 1:
        models = sys.argv[1:]
    else:
        models = DEFAULT_MODELS
    
    for model_name in models:
        print(f"\n{'#'*60}")
        print(f"# Running pipeline for: {model_name}")
        print(f"{'#'*60}\n")
        
        # Step 1: Probing + rank analysis
        run_experiment(model_name=model_name)
        
        # Step 2: Steering + ablation
        run_interventions(model_name=model_name)
        
        print(f"\n{'#'*60}")
        print(f"# DONE: {model_name}")
        print(f"{'#'*60}\n")

if __name__ == "__main__":
    main()