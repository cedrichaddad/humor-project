"""
generate_sae_figures.py

Comprehensive visualization suite for SAE humor-feature analysis on Gemma-2-2B.
Produces publication-quality figures with principled feature categorization.

Categories
──────────
  • Semantic Humor  – features whose top tokens reflect genuine humor signals
                      (laughter words, surprise markers, rhetorical patterns)
  • Dataset Artifact – features that fire on code tokens, multilingual tokens,
                      or structural patterns unrelated to humor content
  • Other/Structural – features on frequent closed-class words, BOS/EOS,
                      punctuation, or ambiguous function words
"""

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════
DATA_PATH = Path("results/gemma-2-2b_sae_complete_experiment.json")
OUTPUT_DIR = Path("results/gemma-2-2b/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# Feature categorization  (based on manual inspection of top_tokens)
# ═══════════════════════════════════════════════════════════════════════════
# Each feature index → (category, short human-readable label)
FEATURE_CATEGORIES = {
    # ── Semantic Humor ────────────────────────────────────────────────
    7531:  ("Semantic Humor",    "Surprise punctuation (?!?)"),
    7343:  ("Semantic Humor",    "Laughter tokens (lol, haha)"),
    8665:  ("Semantic Humor",    "Hesitation / delivery (uh, um)"),
    12844: ("Semantic Humor",    "Exclamatory 'what' (WHAT?!)"),
    9303:  ("Semantic Humor",    "Profanity / shock (mierda, !?)"),
    15527: ("Semantic Humor",    "Double-meaning (\"mean\", meant)"),
    8064:  ("Semantic Humor",    "Punchline action (get rid, gets)"),
    4667:  ("Semantic Humor",    "Intensifiers (really, basically)"),
    12443: ("Semantic Humor",    "Question clusters (??, ?\")"),

    # ── Dataset Artifact ──────────────────────────────────────────────
    2748:  ("Dataset Artifact",  "Code: MigrationBuilder"),
    10663: ("Dataset Artifact",  "Code: principalTable, queryInterface"),
    4250:  ("Dataset Artifact",  "Arabic script tokens"),
    14701: ("Dataset Artifact",  "Malay/informal (alot, diatas)"),
    7224:  ("Dataset Artifact",  "Russian (данного) + informal"),
    9382:  ("Dataset Artifact",  "Code: ModelExpression, featureID"),

    # ── Other / Structural ────────────────────────────────────────────
    2048:  ("Other/Structural",  "BOS / sequence start"),
    4234:  ("Other/Structural",  "EOS / newline boundaries"),
    1275:  ("Other/Structural",  "Function words (it, in, the)"),
    2567:  ("Other/Structural",  "Pronouns / determiners (they, those)"),
    3408:  ("Other/Structural",  "Mixed (agua, increased, C)"),
}

# Color palette
CAT_COLORS = {
    "Semantic Humor":   "#2ca02c",   # green
    "Dataset Artifact": "#d62728",   # red
    "Other/Structural": "#7f7f7f",   # grey
}

# ═══════════════════════════════════════════════════════════════════════════
# Load & prepare data
# ═══════════════════════════════════════════════════════════════════════════
with open(DATA_PATH) as f:
    data = json.load(f)

df = pd.DataFrame(data["discovery"])

# Attach category + human label
df["category"] = df["feature_idx"].map(lambda x: FEATURE_CATEGORIES.get(x, ("Unknown", "?"))[0])
df["human_label"] = df["feature_idx"].map(lambda x: FEATURE_CATEGORIES.get(x, ("Unknown", str(x)))[1])
df["color"] = df["category"].map(CAT_COLORS)

# Handy short label for axes
df["bar_label"] = df.apply(
    lambda r: f"F{r['feature_idx']}: {r['human_label']}", axis=1
)

# Global style
sns.set_theme(style="whitegrid", font_scale=1.1)

print(f"Loaded {len(df)} features from {DATA_PATH}")
print(f"  Semantic Humor:   {(df['category'] == 'Semantic Humor').sum()}")
print(f"  Dataset Artifact: {(df['category'] == 'Dataset Artifact').sum()}")
print(f"  Other/Structural: {(df['category'] == 'Other/Structural').sum()}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Categorized Top-20 Feature Bar Chart
# ═══════════════════════════════════════════════════════════════════════════
def fig_top_features_bar():
    fig, ax = plt.subplots(figsize=(13, 10))
    df_s = df.sort_values("diff", ascending=True)

    bars = ax.barh(df_s["bar_label"], df_s["diff"], color=df_s["color"], edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Activation Difference  (Joke Mean − Non-Joke Mean)", fontsize=13)
    ax.set_title("Top-20 SAE Features Distinguishing Humor  ·  Gemma-2-2B Layer 15", fontsize=15, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Legend
    patches = [mpatches.Patch(color=c, label=l) for l, c in CAT_COLORS.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=11, framealpha=0.9)

    fig.tight_layout()
    out = OUTPUT_DIR / "01_top_features_categorized.png"
    fig.savefig(out, dpi=300)
    print(f"✓ Saved {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Signal vs Noise Scatter (joke_mean vs nonjoke_mean)
# ═══════════════════════════════════════════════════════════════════════════
def fig_signal_noise_scatter():
    fig, ax = plt.subplots(figsize=(10, 9))

    for cat, grp in df.groupby("category"):
        ax.scatter(
            grp["nonjoke_mean"], grp["joke_mean"],
            s=grp["diff"] * 30,
            c=CAT_COLORS[cat], label=cat,
            alpha=0.75, edgecolors="white", linewidth=0.8,
        )

    # y = x diagonal
    lo, hi = 0, max(df["joke_mean"].max(), df["nonjoke_mean"].max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.25, label="y = x (no difference)")

    # Annotate outliers
    for _, r in df.iterrows():
        if r["diff"] > 7 or r["nonjoke_mean"] > 20:
            ax.annotate(
                f"F{r['feature_idx']}",
                (r["nonjoke_mean"], r["joke_mean"]),
                textcoords="offset points", xytext=(8, -4),
                fontsize=8, color=r["color"], fontweight="bold",
            )

    ax.set_xlabel("Mean Activation on Non-Jokes  (noise baseline)", fontsize=12)
    ax.set_ylabel("Mean Activation on Jokes  (humor signal)", fontsize=12)
    ax.set_title("Signal vs. Noise: Feature Selectivity for Humor", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_xlim(left=-0.5)
    ax.set_ylim(bottom=-0.5)

    fig.tight_layout()
    out = OUTPUT_DIR / "02_signal_noise_scatter.png"
    fig.savefig(out, dpi=300)
    print(f"✓ Saved {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Category Breakdown (pie + stacked contribution)
# ═══════════════════════════════════════════════════════════════════════════
def fig_category_breakdown():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: pie chart of feature counts
    counts = df["category"].value_counts()
    colors = [CAT_COLORS[c] for c in counts.index]
    wedges, texts, autotexts = ax1.pie(
        counts, labels=counts.index, autopct="%1.0f%%",
        colors=colors, startangle=90,
        textprops={"fontsize": 11},
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax1.set_title("Feature Count by Category", fontsize=13, fontweight="bold")

    # Right: stacked bar showing total Δ contribution per category
    cat_diff = df.groupby("category")["diff"].sum().reindex(CAT_COLORS.keys())
    cat_diff_pct = cat_diff / cat_diff.sum() * 100
    bottom = 0
    for cat in CAT_COLORS:
        val = cat_diff_pct.get(cat, 0)
        ax2.bar("Total Δ Contribution", val, bottom=bottom, color=CAT_COLORS[cat], label=cat, width=0.5)
        if val > 5:
            ax2.text(0, bottom + val / 2, f"{val:.0f}%", ha="center", va="center", fontsize=12, fontweight="bold")
        bottom += val
    ax2.set_ylabel("% of Total Activation Difference", fontsize=12)
    ax2.set_title("Δ Contribution by Category", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=10)

    fig.suptitle("How Much of the 'Humor Signal' Is Real?", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = OUTPUT_DIR / "03_category_breakdown.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"✓ Saved {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Joke vs Non-Joke Mean Activation (grouped bar)
# ═══════════════════════════════════════════════════════════════════════════
def fig_paired_activations():
    fig, ax = plt.subplots(figsize=(14, 8))

    df_s = df.sort_values("diff", ascending=False)
    x = np.arange(len(df_s))
    w = 0.35

    bars_j = ax.bar(x - w / 2, df_s["joke_mean"], w, label="Jokes", color="#1f77b4", alpha=0.85)
    bars_n = ax.bar(x + w / 2, df_s["nonjoke_mean"], w, label="Non-Jokes", color="#ff7f0e", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"F{idx}" for idx in df_s["feature_idx"]],
        rotation=45, ha="right", fontsize=9,
    )

    # Color x-tick labels by category
    for tick, (_, row) in zip(ax.get_xticklabels(), df_s.iterrows()):
        tick.set_color(row["color"])
        tick.set_fontweight("bold")

    ax.set_ylabel("Mean Max Activation", fontsize=12)
    ax.set_title("Joke vs. Non-Joke Activations per Feature", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = OUTPUT_DIR / "04_paired_activations.png"
    fig.savefig(out, dpi=300)
    print(f"✓ Saved {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Selectivity Ratio  (joke_mean / nonjoke_mean)
# ═══════════════════════════════════════════════════════════════════════════
def fig_selectivity_ratio():
    fig, ax = plt.subplots(figsize=(13, 8))

    df_sel = df.copy()
    df_sel["selectivity"] = df_sel["joke_mean"] / df_sel["nonjoke_mean"].clip(lower=0.01)
    df_sel = df_sel.sort_values("selectivity", ascending=True)

    bars = ax.barh(
        df_sel["bar_label"], df_sel["selectivity"],
        color=df_sel["color"], edgecolor="white", linewidth=0.5,
    )

    ax.axvline(x=1.0, color="black", linestyle="--", alpha=0.4, label="Ratio = 1 (no selectivity)")
    ax.set_xlabel("Selectivity Ratio  (Joke Mean / Non-Joke Mean)", fontsize=12)
    ax.set_title("Feature Selectivity: How Humor-Specific Is Each Feature?", fontsize=15, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    patches = [mpatches.Patch(color=c, label=l) for l, c in CAT_COLORS.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=10, framealpha=0.9)

    fig.tight_layout()
    out = OUTPUT_DIR / "05_selectivity_ratio.png"
    fig.savefig(out, dpi=300)
    print(f"✓ Saved {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6: Causal Validation – Steering vs Ablation comparison
# ═══════════════════════════════════════════════════════════════════════════
def fig_causal_validation():
    validation = data.get("validation", [])
    if not validation:
        print("⚠  No validation data found, skipping causal figure.")
        return

    n_features = len(validation)
    prompts = list(validation[0]["steering"].keys())
    n_prompts = len(prompts)

    fig, axes = plt.subplots(n_features, 1, figsize=(14, 4 * n_features))
    if n_features == 1:
        axes = [axes]

    for ax, entry in zip(axes, validation):
        fidx = entry["feature_idx"]
        cat, label = FEATURE_CATEGORIES.get(fidx, ("Unknown", str(fidx)))

        rows = []
        for p in prompts:
            short_p = p[:35] + "…" if len(p) > 35 else p
            steered = entry["steering"].get(p, "")
            ablated = entry["ablation"].get(p, "")
            rows.append([short_p, textwrap.shorten(steered, 80, placeholder="…"), textwrap.shorten(ablated, 400, placeholder="…")])

        ax.axis("off")
        color = CAT_COLORS.get(cat, "#333")
        ax.set_title(
            f"Feature {fidx}  [{cat}]  –  {label}",
            fontsize=12, fontweight="bold", color=color, loc="left",
        )

        table = ax.table(
            cellText=rows,
            colLabels=["Prompt", "Steered (α=30)", "Ablated"],
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.8)

        # Header styling
        for j in range(3):
            table[0, j].set_facecolor("#e0e0e0")
            table[0, j].set_text_props(fontweight="bold")

    fig.suptitle(
        "Causal Validation: Steering & Ablation on Top-5 Features",
        fontsize=15, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out = OUTPUT_DIR / "06_causal_validation.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"✓ Saved {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Generate all figures
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\nGenerating figures → {OUTPUT_DIR}/\n")
    fig_top_features_bar()
    fig_signal_noise_scatter()
    fig_category_breakdown()
    fig_paired_activations()
    fig_selectivity_ratio()
    fig_causal_validation()
    print(f"\n✅ All figures saved to {OUTPUT_DIR}")