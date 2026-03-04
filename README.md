# Mechanistic Interpretability of Humor in Language Models

This repository contains the code, data, and results for our paper **"Mechanistic Interpretability of Humor in Language Models: From Correlation to Causation"** (COLM 2025).

We investigate whether humor is encoded as a linear direction in Gemma-2 2B's residual stream, establish causal necessity and sufficiency through ablation and steering experiments, and decompose the humor direction into sparse interpretable features using GemmaScope SAEs.

## Key Findings

- **Humor is highly linearly separable**: A linear probe achieves **99% accuracy** at layer 15 of Gemma-2 2B
- **The representation is causally active**: Layer-specific ablation across layers 15–19 collapses layer 20 probe accuracy from 92.5% to **48.75%** (below chance)
- **Steering produces dose-dependent control**: Adding the humor direction during generation shifts outputs toward humor predictably; effectiveness peaks at layers 10, 18, and 20–22
- **The direction rotates across layers**: 14 PCA components needed to capture 90% of cross-layer variance — humor is continuously re-encoded, not static
- **55% of the discriminative signal is genuine semantic content**: SAE analysis at layer 15 identifies 12/20 top features as Semantic Humor, encoding recognizable human humor cues (surprise, incongruity, punchline structure, laughter tokens)

## Quick Results

| Metric | Value |
|--------|-------|
| Best probe accuracy | 99.0% |
| Best layer (probing) | 15 (of 26) |
| Layer-specific ablation accuracy at layer 20 | 48.75% (below chance) |
| Shared-direction ablation accuracy at layer 20 | 85.0% |
| Peak steering layer | 20 (Δ = 1.79) |
| PCA components for 90% cross-layer variance | 14 |
| Semantic humor features at layer 15 | 12/20 (60%) |
| Semantic share of total Δf at layer 15 | 55% |

## Repository Structure

```
.
├── README.md
├── pyproject.toml
├── modal_gemma.py                  # Modal cloud setup for Gemma-2 2B
├── modal_runner.py                 # Modal job runner
├── datasets/
│   ├── dataset_a_paired.xlsx       # 1,000 aligned humor–serious pairs (unfun)
│   ├── dataset_b_randomized.xlsx   # 1,000 humor + 1,000 non-humor (unpaired)
│   ├── model_outputs.xlsx          # Raw model generation outputs
│   ├── samples.json                # Sample subset for inspection
│   └── README.md
├── src/
│   ├── experiment.py               # Linear probing across all 26 layers
│   ├── intervention_tests.py       # Ablation and steering experiments
│   ├── sae_analysis.py             # SAE feature extraction and filtering
│   ├── run_sae_experiment.py       # Full SAE pipeline runner (layers 15–20)
│   └── generate_sae_figures.py     # Per-layer diagnostic figure generation
└── gemma-2-2b/
    ├── config.json
    ├── probe_results.csv            # Per-layer accuracy, AUC, cosine similarity
    ├── probe_summary.json
    ├── rank_analysis.json           # Cross-layer PCA variance explained
    ├── dataset_comparison.json      # Dataset A vs B direction comparison
    ├── intervention_results.json
    ├── humor_direction.pt           # Saved humor direction vector
    ├── directions/
    │   ├── layer0.pt – layer25.pt   # Per-layer saved humor directions
    │   ├── best_layer15.pt
    │   └── summary.json
    ├── figures/
    │   ├── probe_accuracy_by_layer.png
    │   ├── pca_rank_analysis.png
    │   ├── direction_similarity.png
    │   ├── ablation_impact.png
    │   ├── downstream_probe_shared_abl15-19_probe20.png
    │   ├── downstream_probe_layerspecific_abl15-19_probe20.png
    │   ├── steering_by_layer_fixed.png
    │   ├── steering_by_layer_humor_fixed.png
    │   ├── steering_by_layer_control_fixed.png
    │   ├── steering_logit_diff.png
    │   └── steering_triangulation.png
    └── sae_results/
        └── layer_{15-20}/
            ├── gemma-2-2b_sae_features.json
            ├── gemma-2-2b_sae_complete_experiment.json
            └── figures/
                ├── 01_top_features_categorized.png
                ├── 02_signal_noise_scatter.png
                ├── 03_category_breakdown.png
                ├── 04_paired_activations.png
                ├── 05_selectivity_ratio.png
                └── 06_causal_validation.png
```

## Reproducing Results

### Environment Setup

```bash
uv venv
source .venv/bin/activate
uv sync
```

### Run Experiments

```bash
# 1. Linear probing across all 26 layers
python src/experiment.py

# 2. Ablation and steering experiments
python src/intervention_tests.py

# 3. SAE feature extraction and causal validation (layers 15–20)
python src/run_sae_experiment.py

# 4. Generate per-layer diagnostic figures
python src/generate_sae_figures.py --base-dir gemma-2-2b/sae_results
```

Experiments were run on Modal using `modal_gemma.py` and `modal_runner.py`. To reproduce on Modal:

```bash
modal run modal_runner.py
```

## Methodology

1. **Model**: Gemma-2 2B (26 layers, hidden dimension 2304, vocabulary 256,128), accessed via TransformerLens
2. **Datasets**: Dataset A — 1,000 aligned humor–serious pairs via the unfun library; Dataset B — 1,000 humor + 1,000 non-humor ColBERT samples (unpaired)
3. **Probing**: Logistic regression at each layer's residual stream (final token); pair-wise train/test splitting
4. **Ablation**: Orthogonal projection removes humor direction at layers 15–19; probed at layer 20; shared-direction and layer-specific variants
5. **Steering**: Scaled humor direction added during generation; logit difference over 200 humor-aligned vs. 200 serious-aligned tokens
6. **SAE analysis**: GemmaScope canonical SAEs (width 16k) at layers 15–20; max-pooling over token positions; top-200 candidates artifact-filtered; top-20 manually labeled; top-5 causally validated with α=30

## Citation

```bibtex
@inproceedings{jazra2025humor,
  title={Mechanistic Interpretability of Humor in Language Models: From Correlation to Causation},
  author={Jazra, Shana and Haddad, Cedric and Lu, Catherine},
  booktitle={Conference on Language Modeling (COLM)},
  year={2025}
}
```

## Related Work

- Tigges et al. (2023) — Linear Representations of Sentiment in LLMs
- Marks & Tegmark (2024) — The Geometry of Truth
- Horvitz et al. (2024) — Unfunning: Aligned Humor–Serious Pair Generation
- Annamoradnejad & Zoghi (2020) — ColBERT: Using BERT for Humor Detection
- Cunningham et al. (2023) — Sparse Autoencoders Find Monosemantic Features
- GemmaScope (2024) — Pre-trained SAEs for Gemma-2

## License

Research code for academic use.