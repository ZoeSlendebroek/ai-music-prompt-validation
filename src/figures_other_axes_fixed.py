#!/usr/bin/env python3
"""
figures_prompt_validation.py  — FIXED version

Replace the make_other_axes_figure function in your original script with this one,
or run this file standalone.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
    "savefig.dpi":      300,
})

GREEN = "#43A047"
RED   = "#E53935"
AMBER = "#FB8C00"
DARK  = "#212121"
MID   = "#757575"

SYS_COLOR = {"Suno": "#1565C0", "Lyria": "#2E7D32", "Udio": "#BF360C"}
SYSTEMS   = ["Suno", "Lyria", "Udio"]

OTHER = [
    (
        "Afrobeats", "Density",
        "Onset Density (events/sec)",
        "Counts how many note/drum events occur per second. More = busier.",
        "sparse percussion, few drum hits",
        "dense polyrhythmic percussion",
        5.13, 5.97,
        4.80, 6.47,
        6.07, 6.87,
        None,
    ),
    (
        "Afrobeats", "Texture",
        "Harmonic Ratio",
        "Fraction of audio energy that is melodic/tonal.  0 = all drums,  1 = all melody.",
        "drum-forward, percussion-heavy, minimal melody",
        "melodic synth leads, harmonic content, understated percussion",
        0.588, 0.587,
        0.475, 0.715,
        0.391, 0.759,
        None,
    ),
    (
        "Afrobeats", "Structure",
        "Self-Similarity",
        "How much each moment resembles other moments.  Higher = more repetitive/loopy.",
        "loop-based, ostinato, same pattern throughout",
        "varied arrangement, evolving sections, changing texture",
        0.139, 0.129,
        0.118, 0.117,
        0.148, 0.100,
        "* 30s clips limit reliability of this measure",
    ),
    (
        "Metal", "Density",
        "Onset Density (events/sec)",
        "Counts how many note/drum events occur per second. More = busier.",
        "sparse drumming, groove-focused",
        "blast beat drumming, extreme percussion density",
        4.43, 5.70,
        2.90, 6.17,
        6.57, 5.93,
        None,
    ),
    (
        "Metal", "Distortion",
        "Spectral Flatness",
        "How evenly spread energy is across frequencies.  Clean = low (tonal). Distorted = high (noisy).",
        "clean guitar tone, no distortion",
        "heavy saturation, fuzz, noisy texture",
        0.0366, 0.0283,
        0.0337, 0.0397,
        0.0147, 0.0245,
        None,
    ),
]


def make_other_axes_figure(out_path):
    n = len(OTHER)

    # tall figure — extra height per panel for the prompt boxes below x-axis
    fig, all_axes = plt.subplots(n, 1, figsize=(15, 5.5 * n), facecolor="white")
    if n == 1:
        all_axes = [all_axes]

    fig.suptitle(
        "Prompt-Following Validation — Non-Tempo Axes\n"
        "What was asked (grey boxes) vs what was measured (coloured bars), "
        "and how much spread did each system produce?",
        fontsize=13, fontweight="bold", y=1.002, color=DARK
    )

    for ai, (ax, row_data) in enumerate(zip(all_axes, OTHER)):
        (genre, axis, feat_name, feat_expl,
         prompt_lo, prompt_hi,
         slo, shi, llo, lhi, ulo, uhi,
         caveat) = row_data

        vals = {
            "Suno":  (slo, shi),
            "Lyria": (llo, lhi),
            "Udio":  (ulo, uhi),
        }

        ax.set_facecolor("#FAFAFA")
        for spine in ax.spines.values():
            spine.set_color("#E0E0E0")

        # ── title: genre + axis + feature — no overlap with prompt boxes ──
        ax.set_title(
            f"{genre}  —  {axis}\nMeasured by: {feat_name}  ({feat_expl.split(chr(10))[0]})",
            fontsize=11, fontweight="bold",
            loc="left", pad=10, color=DARK
        )

        # ── compute observable range ──────────────────────────────
        all_lo  = [slo, llo, ulo]
        all_hi  = [shi, lhi, uhi]
        obs_min = min(all_lo + all_hi)
        obs_max = max(all_lo + all_hi)
        obs_rng = obs_max - obs_min if obs_max != obs_min else 1e-9

        y_sys = {"Suno": 0, "Lyria": 1, "Udio": 2}
        bar_h = 0.55

        for sys in SYSTEMS:
            vlo, vhi  = vals[sys]
            y         = y_sys[sys]
            col       = SYS_COLOR[sys]
            correct   = vhi > vlo
            spread_pct= abs(vhi - vlo) / obs_rng * 100

            # full observable range background bar
            ax.barh(y, obs_rng, left=obs_min, height=bar_h,
                    color="#EEEEEE", edgecolor="#CCCCCC",
                    linewidth=0.8, zorder=1)

            # actual spread bar
            bar_lo = min(vlo, vhi)
            bar_w  = abs(vhi - vlo)
            ec     = GREEN if correct else RED
            ax.barh(y, bar_w, left=bar_lo, height=bar_h,
                    color=col, alpha=0.75,
                    edgecolor=ec, linewidth=2.5, zorder=3)

            # dots
            ax.scatter(vlo, y, color=col, s=100, zorder=5,
                       edgecolors="white", linewidth=1.5)
            ax.scatter(vhi, y, color=col, s=140, zorder=5,
                       marker="D", edgecolors="white", linewidth=1.5)

            # value labels
            def fmt(v):
                if abs(v) >= 10:   return f"{v:.1f}"
                if abs(v) >= 1:    return f"{v:.2f}"
                if abs(v) >= 0.01: return f"{v:.3f}"
                return f"{v:.4f}"

            offset = obs_rng * 0.02
            ax.text(min(vlo, vhi) - offset, y, fmt(vlo),
                    ha="right", va="center", fontsize=9,
                    color=col, fontweight="bold")
            ax.text(max(vlo, vhi) + offset, y, fmt(vhi),
                    ha="left", va="center", fontsize=9,
                    color=col, fontweight="bold")

            # PASS/FAIL + spread on far right
            verdict   = "PASS" if correct else "FAIL"
            vrd_color = GREEN if correct else RED
            ax.text(obs_max + obs_rng * 0.38, y,
                    f"{verdict}   spread: {spread_pct:.0f}% of range",
                    va="center", fontsize=9.5,
                    fontweight="bold", color=vrd_color)

        # ── y axis: system names ──────────────────────────────────
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(SYSTEMS, fontsize=11, fontweight="bold")
        for tick, sys in zip(ax.get_yticklabels(), SYSTEMS):
            tick.set_color(SYS_COLOR[sys])

        ax.set_ylim(-0.9, 2.6)
        ax.set_xlim(obs_min - obs_rng * 0.3,
                    obs_max + obs_rng * 1.2)
        ax.set_xlabel(feat_name, fontsize=9.5, color=MID, labelpad=8)
        ax.grid(axis="x", alpha=0.3, zorder=0)

        # ── prompt boxes BELOW the x-axis ────────────────────────
        # use ax.annotate with clip_on=False to place text below
        ax.text(
            obs_min - obs_rng * 0.28,
            -0.75,
            f"LOW prompt:\n\"{prompt_lo}\"",
            ha="left", va="top", fontsize=8,
            color="#424242", style="italic",
            bbox=dict(boxstyle="round,pad=0.35",
                      facecolor="#EEEEEE",
                      edgecolor="#BDBDBD",
                      linewidth=0.8),
            clip_on=False
        )

        ax.text(
            obs_max + obs_rng * 0.02,
            -0.75,
            f"HIGH prompt:\n\"{prompt_hi}\"",
            ha="left", va="top", fontsize=8,
            color="#424242", style="italic",
            bbox=dict(boxstyle="round,pad=0.35",
                      facecolor="#EEEEEE",
                      edgecolor="#BDBDBD",
                      linewidth=0.8),
            clip_on=False
        )

        # caveat below prompt boxes
        if caveat:
            ax.text(
                (obs_min + obs_max) / 2, -0.88,
                caveat,
                ha="center", va="top", fontsize=8,
                color=AMBER, style="italic",
                clip_on=False
            )

    # ── shared legend ─────────────────────────────────────────────
    lo_dot  = plt.Line2D([0],[0], marker="o", color="grey",
                          markersize=8, linestyle="None",
                          label="LOW prompt output (circle)")
    hi_dot  = plt.Line2D([0],[0], marker="D", color="grey",
                          markersize=8, linestyle="None",
                          label="HIGH prompt output (diamond)")
    full_rng= mpatches.Patch(color="#EEEEEE", edgecolor="#CCCCCC",
                              label="Full observable range (all systems combined)")
    grn_bar = mpatches.Patch(color=GREEN, alpha=0.7,
                              label="PASS — correct direction")
    red_bar = mpatches.Patch(color=RED,   alpha=0.7,
                              label="FAIL — wrong direction")

    fig.legend(handles=[lo_dot, hi_dot, full_rng, grn_bar, red_bar],
               loc="lower center",
               bbox_to_anchor=(0.5, -0.01),
               ncol=5, fontsize=9, frameon=True,
               edgecolor="#E0E0E0")

    fig.text(
        0.5, -0.03,
        "Spread = fraction of the observable range each system used.  "
        "Low spread = both conditions produced similar output = homogenization.\n"
        "n=1 per condition — descriptive pilot observations only.",
        ha="center", fontsize=8.5, color=MID, style="italic"
    )

    fig.tight_layout(h_pad=4.5)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    make_other_axes_figure(out_dir / "fig_other_axes_compliance.png")
    print("Done.")
