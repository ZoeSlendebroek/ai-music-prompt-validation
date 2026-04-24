#!/usr/bin/env python3
"""
3_visualise.py

Generates publication-quality figures for the prompt-following validation study.

Figures produced:
  fig1_direction_heatmap.png     — heatmap: did each system follow direction per axis?
  fig2_effect_sizes.png          — Cliff's delta per axis × system, all genres
  fig3_tempo_accuracy.png        — absolute tempo accuracy (BPM target vs output)
  fig4_feature_distributions.png — violin plots: LOW vs HIGH per axis per system
  fig5_radar.png                 — radar chart: overall prompt-following score per system
  fig6_per_genre.png             — faceted bar chart: directional accuracy by genre

Inputs:
    data/results/prompt_following_summary.csv
    data/results/track_level_results.csv
    data/features/all_tracks_features.csv

Usage:
    python src/3_visualise.py
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# ── publication style ────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":      300,
    "savefig.dpi":     300,
    "font.family":     "serif",
    "font.size":       10,
    "axes.labelsize":  11,
    "axes.titlesize":  12,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "legend.frameon":  False,
})

SYSTEM_COLORS = {
    "suno":  "#2196F3",   # blue
    "lyria": "#4CAF50",   # green
    "udio":  "#FF5722",   # orange
}
SYSTEM_LABELS = {"suno": "Suno", "lyria": "Lyria", "udio": "Udio"}

AXIS_LABELS = {
    "tempo":      "Tempo",
    "density":    "Density",
    "texture":    "Texture\n(harmonic)",
    "structure":  "Structure\n(self-sim)",
    "distortion": "Distortion\n(flatness)",
}

FEATURE_LABELS = {
    "tempo":                "Tempo (BPM)",
    "onset_density":        "Onset Density (events/s)",
    "harmonic_ratio":       "Harmonic Ratio",
    "self_similarity_mean": "Self-Similarity",
    "spectral_flatness_mean":"Spectral Flatness",
}


def load_data(project_root):
    res_dir  = project_root / "data" / "results"
    feat_csv = project_root / "data" / "features" / "all_tracks_features.csv"
    summary  = pd.read_csv(res_dir / "prompt_following_summary.csv")
    tracks   = pd.read_csv(res_dir / "track_level_results.csv")
    features = pd.read_csv(feat_csv)
    return summary, tracks, features


# ── Figure 1: Direction heatmap ──────────────────────────────────
def fig1_direction_heatmap(summary, out_dir):
    """
    Grid: rows = axis × genre, columns = system.
    Green = direction correct, Red = wrong, with Cliff's delta as annotation.
    """
    systems   = sorted(summary["system"].unique())
    axis_types= sorted([a for a in summary["axis_type"].unique() if a != "combined"])
    genres    = sorted(summary["genre"].unique())

    # create row labels
    rows = [(g, a) for g in genres for a in axis_types
            if not summary[(summary["genre"]==g) & (summary["axis_type"]==a)].empty]

    n_rows = len(rows)
    n_cols = len(systems)

    fig, ax = plt.subplots(figsize=(3.5 + n_cols * 1.8, 1.2 + n_rows * 0.7))

    green = "#4CAF50"
    red   = "#EF5350"
    grey  = "#EEEEEE"

    for ri, (genre, axis) in enumerate(rows):
        for ci, sys in enumerate(systems):
            sub = summary[(summary["system"]==sys) &
                          (summary["genre"]==genre) &
                          (summary["axis_type"]==axis)]
            if sub.empty:
                color = grey
                label = "—"
            else:
                r     = sub.iloc[0]
                color = green if r["direction_correct"] else red
                cd    = r["cliffs_delta"]
                sig   = "*" if r["significant_p05"] else ""
                label = f"{cd:+.2f}{sig}"

            rect = mpatches.FancyBboxPatch(
                (ci + 0.05, n_rows - ri - 0.95), 0.9, 0.85,
                boxstyle="round,pad=0.05",
                facecolor=color, edgecolor="white", linewidth=1.5,
                transform=ax.transData,
            )
            ax.add_patch(rect)
            ax.text(ci + 0.5, n_rows - ri - 0.52, label,
                    ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white")

    # axis labels
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([i + 0.5 for i in range(n_cols)])
    ax.set_xticklabels([SYSTEM_LABELS.get(s, s) for s in systems], fontsize=11)
    ax.set_yticks([n_rows - i - 0.52 for i in range(n_rows)])
    ax.set_yticklabels(
        [f"{g.capitalize()} — {AXIS_LABELS.get(a, a)}" for g, a in rows],
        fontsize=9
    )
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # legend
    green_p = mpatches.Patch(color=green, label="Direction correct")
    red_p   = mpatches.Patch(color=red,   label="Direction wrong")
    ax.legend(handles=[green_p, red_p], loc="lower right",
              bbox_to_anchor=(1, -0.08), ncol=2, fontsize=9)

    ax.set_title("Prompt-Following Direction Accuracy\n(Cliff's delta shown; * p<.05)",
                 fontsize=12, fontweight="bold", pad=18)

    fig.tight_layout()
    fig.savefig(out_dir / "fig1_direction_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ fig1_direction_heatmap.png")


# ── Figure 2: Effect sizes (Cliff's delta) ───────────────────────
def fig2_effect_sizes(summary, out_dir):
    """
    Grouped bar chart: Cliff's delta per axis, grouped by system.
    Dashed reference lines at |d| = 0.147 (small), 0.330 (medium), 0.474 (large).
    """
    axis_types = sorted([a for a in summary["axis_type"].unique() if a != "combined"])
    systems    = sorted(summary["system"].unique())
    genres     = sorted(summary["genre"].unique())

    n_genres = len(genres)
    fig, axes = plt.subplots(1, n_genres, figsize=(6 * n_genres, 5), sharey=True)
    if n_genres == 1:
        axes = [axes]

    for ax, genre in zip(axes, genres):
        sub_g  = summary[summary["genre"] == genre]
        x      = np.arange(len(axis_types))
        width  = 0.25
        n_sys  = len(systems)
        offsets= np.linspace(-(n_sys-1)/2, (n_sys-1)/2, n_sys) * width

        for i, sys in enumerate(systems):
            vals = []
            for a in axis_types:
                r = sub_g[(sub_g["system"]==sys) & (sub_g["axis_type"]==a)]
                vals.append(float(r["cliffs_delta"].iloc[0]) if not r.empty else 0.0)
            bars = ax.bar(x + offsets[i], vals,
                          width=width * 0.9,
                          color=SYSTEM_COLORS.get(sys, "grey"),
                          label=SYSTEM_LABELS.get(sys, sys),
                          alpha=0.85, zorder=3)

        # reference lines
        for level, ls, label in [(0.147, ":", "small"),
                                  (0.330, "--", "medium"),
                                  (0.474, "-.", "large")]:
            ax.axhline( level, color="grey", linewidth=0.8, linestyle=ls, zorder=1)
            ax.axhline(-level, color="grey", linewidth=0.8, linestyle=ls, zorder=1)

        ax.axhline(0, color="black", linewidth=0.8, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([AXIS_LABELS.get(a, a) for a in axis_types], fontsize=9)
        ax.set_title(genre.capitalize(), fontsize=12, fontweight="bold")
        ax.set_ylim(-1.1, 1.1)
        ax.set_ylabel("Cliff's delta (effect size)" if ax == axes[0] else "")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.suptitle("Effect Size of Prompt Following per Axis and System\n"
                 "(positive = high prompt produced higher value than low prompt)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_effect_sizes.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ fig2_effect_sizes.png")


# ── Figure 3: Tempo accuracy ─────────────────────────────────────
def fig3_tempo_accuracy(features, out_dir):
    """
    Scatter + violin: measured BPM vs target BPM (80 and 120) for tempo prompts.
    One panel per system.
    """
    tempo_df = features[features["axis_type"] == "tempo"].copy()
    if tempo_df.empty:
        print("  [skip] fig3 — no tempo data")
        return

    systems  = sorted(tempo_df["system"].unique())
    targets  = {"low": 80, "high": 120}

    fig, axes = plt.subplots(1, len(systems), figsize=(4.5 * len(systems), 4.5),
                              sharey=True)
    if len(systems) == 1:
        axes = [axes]

    for ax, sys in zip(axes, systems):
        sub = tempo_df[tempo_df["system"] == sys]
        color = SYSTEM_COLORS.get(sys, "grey")

        for direction, target_bpm in targets.items():
            vals = sub[sub["axis_direction"] == direction]["tempo"].dropna().values
            if len(vals) == 0:
                continue
            label = f"Prompt: {target_bpm} BPM\n(n={len(vals)})"
            x_pos = 0 if direction == "low" else 1
            parts = ax.violinplot([vals], positions=[x_pos],
                                  showmeans=True, showmedians=False,
                                  widths=0.6)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.5)
            for partname in ["cmeans", "cbars", "cmins", "cmaxes"]:
                if partname in parts:
                    parts[partname].set_color(color)

            ax.axhline(target_bpm, color=color, linestyle="--",
                       linewidth=1.2, alpha=0.7)
            ax.scatter([x_pos] * len(vals), vals,
                       color=color, alpha=0.6, s=30, zorder=5)
            ax.text(x_pos, target_bpm + 2, f"Target: {target_bpm}",
                    ha="center", fontsize=8, color=color)

        # mean absolute error
        for x_pos, direction, target_bpm in [(0,"low",80),(1,"high",120)]:
            vals = sub[sub["axis_direction"] == direction]["tempo"].dropna().values
            if len(vals):
                mae = np.mean(np.abs(vals - target_bpm))
                ax.text(x_pos, ax.get_ylim()[0] + 3 if ax.get_ylim()[0] > 0 else 50,
                        f"MAE={mae:.1f}", ha="center", fontsize=8, style="italic")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Low (80 BPM)", "High (120 BPM)"])
        ax.set_title(SYSTEM_LABELS.get(sys, sys), fontweight="bold")
        ax.set_ylabel("Measured Tempo (BPM)" if sys == systems[0] else "")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Tempo Prompt Accuracy: Target vs Measured BPM",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_tempo_accuracy.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ fig3_tempo_accuracy.png")


# ── Figure 4: Feature distributions per axis ─────────────────────
def fig4_feature_distributions(features, out_dir):
    """
    For each axis, violin plot of the primary feature split by LOW vs HIGH,
    faceted by system. One figure per genre.
    """
    AXIS_FEATURE = {
        "tempo":      "tempo",
        "density":    "onset_density",
        "texture":    "harmonic_ratio",
        "structure":  "self_similarity_mean",
        "distortion": "spectral_flatness_mean",
    }
    DIR_LABELS = {
        "low": "LOW", "high": "HIGH",
        "percussive": "PERCUSSIVE", "melodic": "MELODIC",
        "loop": "LOOP", "varied": "VARIED",
    }

    for genre in sorted(features["genre"].unique()):
        feat_g    = features[features["genre"] == genre]
        axis_list = sorted([a for a in feat_g["axis_type"].unique()
                            if a in AXIS_FEATURE and a != "combined"])
        systems   = sorted(feat_g["system"].unique())

        n_axes = len(axis_list)
        n_sys  = len(systems)
        fig, axes = plt.subplots(n_axes, n_sys,
                                  figsize=(4 * n_sys, 3.5 * n_axes),
                                  squeeze=False)

        for ri, axis_type in enumerate(axis_list):
            feat_col = AXIS_FEATURE[axis_type]
            sub_a    = feat_g[feat_g["axis_type"] == axis_type]
            directions = sorted(sub_a["axis_direction"].unique())

            for ci, sys in enumerate(systems):
                ax     = axes[ri][ci]
                sub_as = sub_a[sub_a["system"] == sys]
                color  = SYSTEM_COLORS.get(sys, "grey")

                vals_list  = []
                dir_labels = []
                for d in directions:
                    v = sub_as[sub_as["axis_direction"] == d][feat_col].dropna().values
                    if len(v):
                        vals_list.append(v)
                        dir_labels.append(DIR_LABELS.get(d, d))

                if not vals_list:
                    ax.set_visible(False)
                    continue

                pos = list(range(len(vals_list)))
                parts = ax.violinplot(vals_list, positions=pos,
                                      showmeans=True, showmedians=False,
                                      widths=0.6)
                for pc in parts["bodies"]:
                    pc.set_facecolor(color)
                    pc.set_alpha(0.5)
                for partname in ["cmeans","cbars","cmins","cmaxes"]:
                    if partname in parts:
                        parts[partname].set_color(color)

                for pi, v in enumerate(vals_list):
                    ax.scatter([pi]*len(v), v, color=color, alpha=0.5, s=25, zorder=5)

                ax.set_xticks(pos)
                ax.set_xticklabels(dir_labels, fontsize=9)
                ax.set_ylabel(FEATURE_LABELS.get(feat_col, feat_col), fontsize=8)
                ax.grid(axis="y", alpha=0.3)

                if ri == 0:
                    ax.set_title(SYSTEM_LABELS.get(sys, sys), fontweight="bold", fontsize=11)
                if ci == 0:
                    ax.set_ylabel(f"{AXIS_LABELS.get(axis_type,axis_type)}\n"
                                  f"{FEATURE_LABELS.get(feat_col,feat_col)}", fontsize=9)

        fig.suptitle(f"Feature Distributions: LOW vs HIGH Prompts — {genre.capitalize()}",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        fname = f"fig4_distributions_{genre}.png"
        fig.savefig(out_dir / fname, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ {fname}")


# ── Figure 5: Radar chart ────────────────────────────────────────
def fig5_radar(summary, out_dir):
    """
    Radar chart: one spoke per axis, value = |Cliff's delta| (0–1).
    One polygon per system, averaged across genres.
    """
    axis_types = sorted([a for a in summary["axis_type"].unique() if a != "combined"])
    systems    = sorted(summary["system"].unique())
    N          = len(axis_types)
    angles     = [n / float(N) * 2 * np.pi for n in range(N)]
    angles    += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6),
                            subplot_kw=dict(projection="polar"))

    for sys in systems:
        vals = []
        for a in axis_types:
            sub = summary[(summary["system"]==sys) & (summary["axis_type"]==a)]
            if sub.empty:
                vals.append(0.0)
            else:
                vals.append(float(sub["cliffs_delta"].abs().mean()))
        vals += vals[:1]
        color = SYSTEM_COLORS.get(sys, "grey")
        ax.plot(angles, vals, color=color, linewidth=2,
                label=SYSTEM_LABELS.get(sys, sys))
        ax.fill(angles, vals, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([AXIS_LABELS.get(a, a) for a in axis_types], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2\n(small)", "0.4\n(medium)", "0.6\n(large)", "0.8"],
                        fontsize=7, color="grey")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)
    ax.set_title("Prompt-Following Strength per Axis\n(|Cliff's delta|, averaged across genres)",
                 fontsize=11, fontweight="bold", pad=20)

    fig.tight_layout()
    fig.savefig(out_dir / "fig5_radar.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ fig5_radar.png")


# ── Figure 6: Directional accuracy bar chart ─────────────────────
def fig6_accuracy_bars(summary, out_dir):
    """
    Grouped bar chart: % direction correct per system, split by genre and axis.
    Simple and clear for a presentation.
    """
    systems = sorted(summary["system"].unique())
    genres  = sorted(summary["genre"].unique())
    axis_types = sorted([a for a in summary["axis_type"].unique() if a != "combined"])

    fig, axes = plt.subplots(1, len(genres), figsize=(5.5 * len(genres), 4.5),
                              sharey=True)
    if len(genres) == 1:
        axes = [axes]

    for ax, genre in zip(axes, genres):
        sub_g  = summary[summary["genre"] == genre]
        x      = np.arange(len(axis_types))
        width  = 0.25
        n_sys  = len(systems)
        offsets= np.linspace(-(n_sys-1)/2, (n_sys-1)/2, n_sys) * width

        for i, sys in enumerate(systems):
            vals = []
            for a in axis_types:
                r = sub_g[(sub_g["system"]==sys) & (sub_g["axis_type"]==a)]
                vals.append(1.0 if (not r.empty and r.iloc[0]["direction_correct"]) else 0.0)

            ax.bar(x + offsets[i], [v*100 for v in vals],
                   width=width*0.9,
                   color=SYSTEM_COLORS.get(sys, "grey"),
                   label=SYSTEM_LABELS.get(sys, sys),
                   alpha=0.85)

        ax.axhline(50, color="grey", linestyle="--", linewidth=0.8, label="Chance (50%)")
        ax.axhline(100, color="lightgrey", linestyle=":", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([AXIS_LABELS.get(a, a) for a in axis_types], fontsize=9)
        ax.set_ylim(0, 115)
        ax.set_title(genre.capitalize(), fontsize=12, fontweight="bold")
        ax.set_ylabel("Direction Correct (%)" if ax == axes[0] else "")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Directional Accuracy: Did the System Follow the Prompt Direction?",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig6_accuracy_bars.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ fig6_accuracy_bars.png")


# ── main ────────────────────────────────────────────────────────
def main():
    project_root = Path(__file__).resolve().parent.parent
    out_dir      = project_root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PROMPT VALIDATION — VISUALISATIONS")
    print("=" * 70)

    summary, tracks, features = load_data(project_root)

    fig1_direction_heatmap(summary, out_dir)
    fig2_effect_sizes(summary, out_dir)
    fig3_tempo_accuracy(features, out_dir)
    fig4_feature_distributions(features, out_dir)
    fig5_radar(summary, out_dir)
    fig6_accuracy_bars(summary, out_dir)

    print(f"\n✓ All figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
