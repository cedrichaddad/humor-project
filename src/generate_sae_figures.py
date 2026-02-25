"""
generate_sae_figures.py

Produces all 6 publication-quality figures for each layer's SAE results.
Runs the full sweep (layers 15-20) in one call.

Figures produced per layer
──────────────────────────
  01_top_features_categorized  – horizontal bar, diff score, colour-coded by category
  02_signal_noise_scatter      – joke mean vs non-joke mean; points above y=x are selective
  03_category_breakdown        – pie (feature count) + stacked bar (delta contribution)
  04_paired_activations        – side-by-side joke/non-joke mean per feature
  05_selectivity_ratio         – joke_mean / nonjoke_mean, sorted ascending
  06_causal_validation         – steering and ablation outputs for top-5 features

Categories
──────────
  Semantic Humor   – top tokens reflect genuine humor signals
  Dataset Artifact – code/multilingual/pretraining residue
  Other/Structural – BOS/EOS, punctuation, high-freq function words
"""

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# ===========================================================================================
# Feature categories per layer
#
# Manually labelled from the actual artifact-filtered SAE output JSONs.
# Each entry: feature_idx -> (category, human-readable label)
#
# Layer 15 labels come from the original single-layer analysis.
# Layers 16-20 labels were assigned after running the full sweep so they
# reflect the filtered output (code/foreign tokens already removed).
# ===========================================================================================

FEATURE_CATEGORIES_BY_LAYER = {

    15: {
        7531:  ("Semantic Humor",   "Surprise punctuation (?!?)"),
        7343:  ("Semantic Humor",   "Laughter tokens (lol, haha)"),
        9303:  ("Semantic Humor",   "Profanity / shock (mierda, !?)"),
        15527: ("Semantic Humor",   'Double-meaning ("mean", meant)'),
        4667:  ("Semantic Humor",   "Intensifiers (really, basically)"),
        2796:  ("Semantic Humor",   "Modal absolutes (never, always, would)"),
        1790:  ("Semantic Humor",   "Punchline reaction (Oh, Ah, yes)"),
        7749:  ("Semantic Humor",   "Exclamatory 'what' (what, WHAT)"),
        12844: ("Semantic Humor",   "Exclamatory 'what' split neuron"),
        4923:  ("Semantic Humor",   "Emphatic direct address (YOU, You)"),
        4781:  ("Semantic Humor",   "Punchline emphasis (!!, ?!?)"),
        10751: ("Semantic Humor",   "Expectation subversion (actually, not)"),
        6846:  ("Semantic Humor",   "Taboo content (eating, sexual, drinking)"),
        11655: ("Dataset Artifact", "Code: fragmented brackets/quotes"),
        4234:  ("Other/Structural", "EOS / newline boundaries"),
        2048:  ("Other/Structural", "BOS / sequence start"),
        2567:  ("Other/Structural", "Pronouns / determiners (they, those)"),
        3408:  ("Other/Structural", "Mixed tokens (agua, increased, C)"),
        8146:  ("Other/Structural", "Setup/punchline line break"),
        12011: ("Other/Structural", "First/second person pronouns (I, my)"),
        1275:  ("Other/Structural", "Function words (it, in, the)"),
        12443: ("Other/Structural", "Question punctuation (??, ?')"),
        9382:  ("Dataset Artifact", "Mixed foreign/code tokens"),
        16047: ("Dataset Artifact", "Assembly/code artifacts"),
        3069:  ("Dataset Artifact", "Whitespace / CJK mixed"),
        9048:  ("Other/Structural", "Answer/Certainly structural"),
        4246:  ("Dataset Artifact", "Code: WriteTagHelper artifacts"),
        14701: ("Dataset Artifact", "Foreign informal (alot, diatas)"),
    },

    16: {
        # ── Semantic Humor ────────────────────────────────────────────────
        4789:  ("Semantic Humor",   "Joking / kidding (joke, Jokes, kidding)"),
        7526:  ("Semantic Humor",   "Surprise punctuation (?!?)"),
        10724: ("Semantic Humor",   "Exclamatory 'what happened' (WHAT)"),
        4822:  ("Semantic Humor",   'Double-meaning ("mean", meant)'),
        1634:  ("Semantic Humor",   "Self + modal (don, am, myself)"),
        5058:  ("Semantic Humor",   "Modal absolutes (can, never, had, would)"),
        3377:  ("Semantic Humor",   "Punchline reaction (Oh, oh, Which)"),
        9654:  ("Semantic Humor",   "Expectation subversion (not, always, actually)"),
        7144:  ("Semantic Humor",   "Exclamation + laughter (!!!!!! lol!!!!)"),
        1190:  ("Semantic Humor",   "Strong profanity (shit, fucker, FUCK)"),
        # ── Dataset Artifact ──────────────────────────────────────────────
        13328: ("Dataset Artifact", "tagHelperRunner / awaiter code artifacts"),
        11572: ("Dataset Artifact", 'Code-adjacent syntax ("):\\r, different{})'),
        7883:  ("Dataset Artifact", "ModelExpression / addGap code artifacts"),
        3566:  ("Dataset Artifact", "AccessorTable / InvalidProtocol artifacts"),
        11920: ("Dataset Artifact", "AttributeSet / AssemblyTitle artifacts"),
        10287: ("Dataset Artifact", "UserScript / JButton code artifacts"),
        # ── Other / Structural ────────────────────────────────────────────
        7403:  ("Other/Structural", "Answer / Well -- Q&A structural"),
        887:   ("Other/Structural", "EOS / newline boundaries"),
        4978:  ("Other/Structural", "Whitespace / indentation"),
        817:   ("Other/Structural", "Function words (we, the, it, in)"),
    },

    17: {
        # ── Semantic Humor ────────────────────────────────────────────────
        13570: ("Semantic Humor",   "Modal absolutes (can, have, didn, could)"),
        7616:  ("Semantic Humor",   "Seriously / Sorry / OK (apology markers)"),
        14000: ("Semantic Humor",   "Surprise punctuation (?!?)"),
        13068: ("Semantic Humor",   "Exclamatory 'what' (WHAT, What, what)"),
        15044: ("Semantic Humor",   "Hedging (Depends, Basically, nothing)"),
        760:   ("Semantic Humor",   "Laughter + profanity (lol, lmao, fucking)"),
        2918:  ("Semantic Humor",   "Internet meta-commentary (IIRC, IMHO, IMO)"),
        12421: ("Semantic Humor",   "Negation / does framing (not, does, Does)"),
        13800: ("Semantic Humor",   "Expectation subversion (not, always, actually)"),
        13868: ("Semantic Humor",   "Why framing (why, Why, Whyte)"),
        # ── Dataset Artifact ──────────────────────────────────────────────
        5373:  ("Dataset Artifact", "BOS / resourceCulture / LookAnd artifacts"),
        12602: ("Dataset Artifact", "Code-like closings (}).'\"])"),
        15935: ("Dataset Artifact", 'Code-adjacent syntax ("):\\r, different{})'),
        9931:  ("Dataset Artifact", "French medical / SequentialGroup artifacts"),
        4244:  ("Dataset Artifact", "ModelExpression / code mixed"),
        14533: ("Dataset Artifact", "UserScript / be ever not mixed"),
        # ── Other / Structural ────────────────────────────────────────────
        15890: ("Other/Structural", "EOS / sentence start (The, This, It)"),
        14695: ("Other/Structural", "Short action tokens (Do, Co, do)"),
        15778: ("Other/Structural", "Pronouns / I / you"),
        7437:  ("Other/Structural", "Self + guess (am, guess, II)"),
    },

    18: {
        # ── Semantic Humor ────────────────────────────────────────────────
        8532:  ("Semantic Humor",   "Modal verbs (can, have, will, could)"),
        8838:  ("Semantic Humor",   "Setup language (kind, happens, else, type)"),
        2528:  ("Semantic Humor",   "Punchline reaction (oh, Oh, Yep, yes, yeah)"),
        12288: ("Semantic Humor",   "Exclamatory 'what' (WHAT, What!)"),
        14403: ("Semantic Humor",   "Discourse pivots (Anyway, Anyways, Anyhow)"),
        9092:  ("Semantic Humor",   "Why / rhetorical framing (Why, why, would)"),
        13928: ("Semantic Humor",   "Explicit humor (humor, humorous, hilarious)"),
        # ── Dataset Artifact ──────────────────────────────────────────────
        8494:  ("Dataset Artifact", "requireNonNull / createState / steamcommunity"),
        1652:  ("Dataset Artifact", "Code-like closings ($. }()]. }}$.)"),
        3497:  ("Dataset Artifact", "denn / SourceChecksum / BoxDecoration artifacts"),
        8982:  ("Dataset Artifact", "UserScript / QName / Ikr mixed artifacts"),
        13457: ("Dataset Artifact", "Personensuche / UserScript / utafiti artifacts"),
        # ── Other / Structural ────────────────────────────────────────────
        1469:  ("Other/Structural", "Quantifiers (few, lot, different, large)"),
        3547:  ("Other/Structural", "Function words (we, it, in, the)"),
        8131:  ("Other/Structural", "Punctuation / ellipsis (/... !... :...)"),
        10330: ("Other/Structural", "Newlines / blockquote boundaries"),
        14676: ("Other/Structural", "Pronouns / determiners (the, it, you, this)"),
        14445: ("Other/Structural", "Articles (a, an, eine, seorang)"),
        3524:  ("Other/Structural", "Pronouns (we, they, it, you, he)"),
        5031:  ("Other/Structural", "Answer / Q&A structural"),
    },

    19: {
        # ── Semantic Humor ────────────────────────────────────────────────
        11028: ("Semantic Humor",   "Modal verbs (was, can, is, had, will)"),
        15731: ("Semantic Humor",   "Deserve / belong / need -- joke premise"),
        4634:  ("Semantic Humor",   "About what happened (kind, What, happened)"),
        7978:  ("Semantic Humor",   "Explicit humor (jokes, joke, humor, humour)"),
        1538:  ("Semantic Humor",   "Informal action (got, gotta, forgot, didn)"),
        1786:  ("Semantic Humor",   "Strong profanity (fucking, FUCKING, goddamn)"),
        14220: ("Semantic Humor",   "Setup framing (does, exactly, about, role)"),
        7914:  ("Semantic Humor",   "Expectation subversion (actually, still, not)"),
        12842: ("Semantic Humor",   "Informal register (been, gotta, gonna)"),
        440:   ("Semantic Humor",   "Rhetorical framing (would, not, pay, why)"),
        3471:  ("Semantic Humor",   "Setup language (kind, type, sorts, types)"),
        # ── Dataset Artifact ──────────────────────────────────────────────
        6694:  ("Dataset Artifact", "UserScript / initComponents / endblock artifacts"),
        # ── Other / Structural ────────────────────────────────────────────
        5406:  ("Other/Structural", "EOS / sentence start (The, This, It, There)"),
        1363:  ("Other/Structural", "Quantifiers (few, different, lot, large)"),
        13980: ("Other/Structural", "Pronouns (we, they, you, it, he)"),
        10399: ("Other/Structural", "Function words (we, it, no, if, the)"),
        3442:  ("Other/Structural", "Mixed punctuation (?!?, newlines)"),
        10937: ("Other/Structural", "EOS / Good / greetings structural"),
        7461:  ("Other/Structural", "Articles (a, an, eine, sebuah, seorang)"),
        5685:  ("Other/Structural", "Punctuation / ellipsis (/... !... :...)"),
    },

    20: {
        # ── Semantic Humor ────────────────────────────────────────────────
        10461: ("Semantic Humor",   "Modal verbs (is, can, has, was, will)"),
        4223:  ("Semantic Humor",   "Question framing (does, Does, did, Did)"),
        6385:  ("Semantic Humor",   "Punchline reception (hey, oh, Seriously, Oh)"),
        1761:  ("Semantic Humor",   "About / what happened framing"),
        11571: ("Semantic Humor",   "Situation language (happens, happened, kind)"),
        8112:  ("Semantic Humor",   "Setup language (kind, sort, type, types)"),
        1555:  ("Semantic Humor",   "Explicit humor (jokes, joke, humor, humour)"),
        10252: ("Semantic Humor",   "Informal register (been, gotta, gonna, BEEN)"),
        8535:  ("Semantic Humor",   "Expectation subversion (exactly, not, else)"),
        10873: ("Semantic Humor",   "Modal + self (am, don, have, dont, guess)"),
        3686:  ("Semantic Humor",   "Emphatic direct address (yourself, yourselves)"),
        259:   ("Semantic Humor",   "Physical / absurdist content (eating, food, shoes)"),
        # ── Dataset Artifact ──────────────────────────────────────────────
        140:   ("Dataset Artifact", "resourceCulture / createState / BASEPATH artifacts"),
        15074: ("Dataset Artifact", "AssemblyCulture / wir / code artifacts"),
        11795: ("Dataset Artifact", "verwijspagina / ujednoznacz / Forumite artifacts"),
        # ── Other / Structural ────────────────────────────────────────────
        1858:  ("Other/Structural", "EOS / sentence start (The, This, It, There)"),
        11092: ("Other/Structural", "Pronouns (they, you, we, he, it)"),
        9982:  ("Other/Structural", "Quantifiers (few, lot, couple, different)"),
        10414: ("Other/Structural", "Function words (we, there, it, if, they)"),
        3586:  ("Other/Structural", "Articles (a, an, eine, sebuah, another)"),
    },
}

CAT_COLORS = {
    "Semantic Humor":   "#2ca02c",
    "Dataset Artifact": "#d62728",
    "Other/Structural": "#7f7f7f",
    "Unknown":          "#9467bd",
}


# ===========================================================================================
# Helpers
# ===========================================================================================

def _safe_label(text: str) -> str:
    """Escape matplotlib special characters in label strings."""
    return text.replace("$", r"\$").replace("{", r"\{").replace("}", r"\}")


def _load_layer_data(layer, base_dir):
    # base_dir is expected to be  results/gemma-2-2b/sae_results/
    # Each layer lives in  base_dir / layer_N /
    # Prefer the complete experiment JSON (has both discovery + validation);
    # fall back to features-only JSON if validation hasn't been run yet
    path = base_dir / f"layer_{layer}" / "gemma-2-2b_sae_complete_experiment.json"
    if not path.exists():
        path = base_dir / f"layer_{layer}" / "gemma-2-2b_sae_features.json"

    if not path.exists():
        raise FileNotFoundError(f"No results for layer {layer} at {path}")

    with open(path) as f:
        raw = json.load(f)

    # Handle both complete experiment format {"discovery": [...], "validation": [...]}
    # and the features-only format (bare list)
    if isinstance(raw, dict) and "discovery" in raw:
        discovery  = raw["discovery"]
        validation = raw.get("validation", [])
    elif isinstance(raw, list):
        discovery  = raw
        validation = []
    else:
        discovery  = raw.get("discovery", raw)
        validation = raw.get("validation", [])

    # Join category labels onto the dataframe for colour-coding and display
    cats = FEATURE_CATEGORIES_BY_LAYER.get(layer, {})
    df = pd.DataFrame(discovery)
    df["category"]    = df["feature_idx"].map(lambda x: cats.get(x, ("Unknown", str(x)))[0])
    df["human_label"] = df["feature_idx"].map(lambda x: _safe_label(cats.get(x, ("Unknown", str(x)))[1]))
    df["color"]       = df["category"].map(CAT_COLORS).fillna("#999999")
    df["bar_label"]   = df.apply(lambda r: f"F{r['feature_idx']}: {r['human_label']}", axis=1)
    return df, validation


def _title(layer):
    # Consistent title prefix used across all figures
    return f"Gemma-2-2B  Layer {layer}"


def _patches():
    # Legend patches matching CAT_COLORS — shared across figures
    return [mpatches.Patch(color=c, label=l) for l, c in CAT_COLORS.items()]


# ===========================================================================================
# Figure functions
# ===========================================================================================

def fig_top_features_bar(df, out_dir, layer):
    # Fig 01: horizontal bar sorted by diff score, colour-coded by category
    # Shows at a glance which features carry the most humor signal and whether
    # those are genuine semantic features or pretraining artifacts
    fig, ax = plt.subplots(figsize=(13, 10))
    df_s = df.sort_values("diff", ascending=True)
    ax.barh(df_s["bar_label"], df_s["diff"], color=df_s["color"], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Activation Difference  (Joke Mean - Non-Joke Mean)", fontsize=13)
    ax.set_title(f"Top-20 SAE Features Distinguishing Humor  |  {_title(layer)}", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(handles=_patches(), loc="lower right", fontsize=11, framealpha=0.9)
    fig.tight_layout()
    out = out_dir / "01_top_features_categorized.png"
    fig.savefig(out, dpi=300)
    print(f"  saved {out.name}")
    plt.close(fig)


def fig_signal_noise_scatter(df, out_dir, layer):
    # Fig 02: scatter of joke_mean vs nonjoke_mean, bubble size = diff
    # Points above y=x line are selective for jokes; points on/below the line
    # activate equally on jokes and non-jokes (noise)
    fig, ax = plt.subplots(figsize=(10, 9))
    for cat, grp in df.groupby("category"):
        ax.scatter(grp["nonjoke_mean"], grp["joke_mean"],
                   s=grp["diff"] * 30, c=CAT_COLORS.get(cat, "#999"),
                   label=cat, alpha=0.75, edgecolors="white", linewidth=0.8)
    lo, hi = 0, max(df["joke_mean"].max(), df["nonjoke_mean"].max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.25, label="y = x")
    # Only annotate features with a strong diff or high base activation —
    # avoids clutter while labelling the most interesting points
    for _, r in df.iterrows():
        if r["diff"] > 7 or r["nonjoke_mean"] > 20:
            ax.annotate(f"F{r['feature_idx']}", (r["nonjoke_mean"], r["joke_mean"]),
                        textcoords="offset points", xytext=(8, -4),
                        fontsize=8, color=r["color"], fontweight="bold")
    ax.set_xlabel("Mean Activation on Non-Jokes", fontsize=12)
    ax.set_ylabel("Mean Activation on Jokes", fontsize=12)
    ax.set_title(f"Signal vs. Noise  |  {_title(layer)}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_xlim(left=-0.5)
    ax.set_ylim(bottom=-0.5)
    fig.tight_layout()
    out = out_dir / "02_signal_noise_scatter.png"
    fig.savefig(out, dpi=300)
    print(f"  saved {out.name}")
    plt.close(fig)


def fig_category_breakdown(df, out_dir, layer):
    # Fig 03: two panels side-by-side.
    # Left (pie): how many of the top-20 features fall into each category.
    # Right (stacked bar): what fraction of the total activation *difference*
    # each category accounts for — a feature count of 2 could contribute 60%
    # of the signal if those features have very large diffs.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    counts = df["category"].value_counts()
    colors = [CAT_COLORS.get(c, "#999") for c in counts.index]
    _, _, autotexts = ax1.pie(counts, labels=counts.index, autopct="%1.0f%%",
                               colors=colors, startangle=90, textprops={"fontsize": 11})
    for at in autotexts:
        at.set_fontweight("bold")
    ax1.set_title("Feature Count by Category", fontsize=13, fontweight="bold")

    cat_diff     = df.groupby("category")["diff"].sum().reindex(CAT_COLORS.keys())
    cat_diff_pct = cat_diff / cat_diff.sum() * 100
    bottom = 0
    for cat in CAT_COLORS:
        val = cat_diff_pct.get(cat, 0)
        ax2.bar("Total", val, bottom=bottom, color=CAT_COLORS[cat], label=cat, width=0.5)
        if val > 5:
            ax2.text(0, bottom + val / 2, f"{val:.0f}%", ha="center", va="center",
                     fontsize=12, fontweight="bold")
        bottom += val
    ax2.set_ylabel("% of Total Activation Difference", fontsize=12)
    ax2.set_title("Delta Contribution by Category", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=10)

    fig.suptitle(f"How Much of the Humor Signal Is Real?  |  {_title(layer)}",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = out_dir / "03_category_breakdown.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  saved {out.name}")
    plt.close(fig)


def fig_paired_activations(df, out_dir, layer):
    # Fig 04: grouped bar chart showing absolute joke vs non-joke mean
    # activation for every feature, sorted by diff descending.
    # Complements fig 02 by making the raw activation magnitudes visible
    # alongside the selectivity — a feature can have a large diff but still
    # fire at moderate absolute levels (or vice versa).
    fig, ax = plt.subplots(figsize=(14, 8))
    df_s = df.sort_values("diff", ascending=False)
    x = np.arange(len(df_s))
    w = 0.35
    ax.bar(x - w/2, df_s["joke_mean"],    w, label="Jokes",     color="#1f77b4", alpha=0.85)
    ax.bar(x + w/2, df_s["nonjoke_mean"], w, label="Non-Jokes", color="#ff7f0e", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"F{idx}" for idx in df_s["feature_idx"]], rotation=45, ha="right", fontsize=9)
    for tick, (_, row) in zip(ax.get_xticklabels(), df_s.iterrows()):
        tick.set_color(row["color"])
        tick.set_fontweight("bold")
    ax.set_ylabel("Mean Max Activation", fontsize=12)
    ax.set_title(f"Joke vs. Non-Joke Activations  |  {_title(layer)}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = out_dir / "04_paired_activations.png"
    fig.savefig(out, dpi=300)
    print(f"  saved {out.name}")
    plt.close(fig)


def fig_selectivity_ratio(df, out_dir, layer):
    # Fig 05: joke_mean / nonjoke_mean ratio per feature, sorted ascending.
    # Ratio > 1 means the feature fires more on jokes than non-jokes.
    # nonjoke_mean is clipped to 0.01 to avoid division-by-zero for features
    # that are essentially silent on non-joke texts.
    fig, ax = plt.subplots(figsize=(13, 8))
    df_sel = df.copy()
    df_sel["selectivity"] = df_sel["joke_mean"] / df_sel["nonjoke_mean"].clip(lower=0.01)
    df_sel = df_sel.sort_values("selectivity", ascending=True)
    ax.barh(df_sel["bar_label"], df_sel["selectivity"],
            color=df_sel["color"], edgecolor="white", linewidth=0.5)
    ax.axvline(x=1.0, color="black", linestyle="--", alpha=0.4, label="Ratio = 1")
    ax.set_xlabel("Selectivity Ratio  (Joke Mean / Non-Joke Mean)", fontsize=12)
    ax.set_title(f"Feature Selectivity  |  {_title(layer)}", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(handles=_patches(), loc="lower right", fontsize=10, framealpha=0.9)
    fig.tight_layout()
    out = out_dir / "05_selectivity_ratio.png"
    fig.savefig(out, dpi=300)
    print(f"  saved {out.name}")
    plt.close(fig)


def fig_causal_validation(validation, out_dir, layer, cats):
    # Fig 06: one subplot per feature (top-5), each showing a 3-row table.
    # Columns: prompt | steered completion (alpha=30) | ablated completion.
    # Steered text going humor-like and ablated text becoming neutral both
    # confirm the feature is causally involved in humor representation.
    if not validation:
        print(f"  (no validation data for layer {layer}, skipping fig 6)")
        return
    n_features = len(validation)
    prompts    = list(validation[0]["steering"].keys())
    fig, axes  = plt.subplots(n_features, 1, figsize=(14, 4 * n_features))
    if n_features == 1:
        axes = [axes]
    for ax, entry in zip(axes, validation):
        fidx      = entry["feature_idx"]
        cat, label = cats.get(fidx, ("Unknown", str(fidx)))
        rows = []
        for p in prompts:
            short_p = p[:35] + "..." if len(p) > 35 else p
            rows.append([short_p,
                         textwrap.shorten(entry["steering"].get(p, ""), 80, placeholder="..."),
                         textwrap.shorten(entry["ablation"].get(p, ""), 400, placeholder="...")])
        ax.axis("off")
        ax.set_title(f"Feature {fidx}  [{cat}]  -  {label}",
                     fontsize=12, fontweight="bold",
                     color=CAT_COLORS.get(cat, "#333"), loc="left")
        tbl = ax.table(cellText=rows,
                       colLabels=["Prompt", "Steered (alpha=30)", "Ablated"],
                       loc="center", cellLoc="left")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.8)
        for j in range(3):
            tbl[0, j].set_facecolor("#e0e0e0")
            tbl[0, j].set_text_props(fontweight="bold")
    fig.suptitle(f"Causal Validation  |  {_title(layer)}", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = out_dir / "06_causal_validation.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  saved {out.name}")
    plt.close(fig)


# ===========================================================================================
# Entry point
# ===========================================================================================

def generate_figures_for_layer(layer, base_dir):
    out_dir = base_dir / f"layer_{layer}" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Layer {layer}  ->  {out_dir}")
    print(f"{'='*55}")

    sns.set_theme(style="whitegrid", font_scale=1.1)
    # Disable LaTeX math text parsing — labels contain $, {}, ^ etc.
    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.default"] = "regular"
    df, validation = _load_layer_data(layer, base_dir)
    cats = FEATURE_CATEGORIES_BY_LAYER.get(layer, {})

    n_unknown = (df["category"] == "Unknown").sum()
    if n_unknown:
        print(f"  WARNING: {n_unknown} features have Unknown category")

    fig_top_features_bar(df, out_dir, layer)
    fig_signal_noise_scatter(df, out_dir, layer)
    fig_category_breakdown(df, out_dir, layer)
    fig_paired_activations(df, out_dir, layer)
    fig_selectivity_ratio(df, out_dir, layer)
    fig_causal_validation(validation, out_dir, layer, cats)
    print(f"  All figures for layer {layer} done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default="results/gemma-2-2b/sae_results")
    args = parser.parse_args()

    base = Path(args.base_dir)
    for layer in [15, 16, 17, 18, 19, 20]:
        try:
            generate_figures_for_layer(layer, base)
        except FileNotFoundError as e:
            print(f"  Skipping layer {layer}: {e}")