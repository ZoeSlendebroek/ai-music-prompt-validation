#!/usr/bin/env python3
"""
figures_prompt_validation.py

Generates two publication figures for the prompt-following validation study.

Figure 1: Tempo axis — target vs actual with range compression
Figure 2: All other axes — table showing what was asked vs what was measured,
          with spread bars showing homogenization

Run from your project root:
    conda activate aijam
    python src/figures_prompt_validation.py

Outputs saved to: figures/
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
    "savefig.dpi":      300,
})

GREEN  = "#43A047"
RED    = "#E53935"
AMBER  = "#FB8C00"
DARK   = "#212121"
MID    = "#757575"
LGREY  = "#F0F0F0"

SYS_COLOR  = {"Suno": "#1565C0", "Lyria": "#2E7D32", "Udio": "#BF360C"}
SYSTEMS    = ["Suno", "Lyria", "Udio"]

# ════════════════════════════════════════════════════════════════
#  DATA
# ════════════════════════════════════════════════════════════════

# Tempo data: (genre, system, target_low, target_high, measured_low, measured_high)
TEMPO = [
    ("Afrobeats", "Suno",   80, 120,  99.4, 117.5),
    ("Afrobeats", "Lyria",  80, 120, 107.7, 123.0),
    ("Afrobeats", "Udio",   80, 120,  99.4, 129.2),
    ("Metal",     "Suno",   75, 180, 143.6, 143.6),
    ("Metal",     "Lyria",  75, 180, 152.0,  92.3),
    ("Metal",     "Udio",   75, 180, 129.2,  99.4),
]

# Other axes data
# (genre, axis, feature_name, feature_explanation,
#  prompt_low, prompt_high,
#  suno_lo, suno_hi, lyria_lo, lyria_hi, udio_lo, udio_hi,
#  caveat)
OTHER = [
    (
        "Afrobeats", "Density",
        "Onset Density (events/sec)",
        "Counts how many note/drum events occur per second. More = busier.",
        "sparse percussion,\nfew drum hits",
        "dense polyrhythmic\npercussion",
        5.13, 5.97,   # Suno
        4.80, 6.47,   # Lyria
        6.07, 6.87,   # Udio
        None,
    ),
    (
        "Afrobeats", "Texture",
        "Harmonic Ratio",
        "Fraction of audio energy that is melodic/tonal.\n0 = all drums,  1 = all melody.",
        "drum-forward,\npercussion-heavy,\nminimal melody",
        "melodic synth leads,\nharmonic content,\nunderstated percussion",
        0.588, 0.587,  # Suno
        0.475, 0.715,  # Lyria
        0.391, 0.759,  # Udio
        None,
    ),
    (
        "Afrobeats", "Structure",
        "Self-Similarity",
        "How much each moment resembles other moments.\nHigher = more repetitive/loopy.",
        "loop-based,\nostinato,\nsame pattern throughout",
        "varied arrangement,\nevolving sections,\nchanging texture",
        0.139, 0.129,  # Suno
        0.118, 0.117,  # Lyria
        0.148, 0.100,  # Udio
        "* 30s clips limit reliability of this measure",
    ),
    (
        "Metal", "Density",
        "Onset Density (events/sec)",
        "Counts how many note/drum events occur per second. More = busier.",
        "sparse drumming,\ngroove-focused",
        "blast beat drumming,\nextreme percussion density",
        4.43, 5.70,   # Suno
        2.90, 6.17,   # Lyria
        6.57, 5.93,   # Udio
        None,
    ),
    (
        "Metal", "Distortion",
        "Spectral Flatness",
        "How evenly spread energy is across frequencies.\nClean guitar = low (tonal peaks). Distorted = high (noise-like).",
        "clean guitar tone,\nno distortion",
        "heavy saturation,\nfuzz, noisy texture",
        0.0366, 0.0283,  # Suno
        0.0337, 0.0397,  # Lyria
        0.0147, 0.0245,  # Udio
        None,
    ),
]


# ════════════════════════════════════════════════════════════════
#  FIGURE 1: TEMPO
# ════════════════════════════════════════════════════════════════

def make_tempo_figure(out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="white")
    fig.suptitle(
        "Tempo Prompt Compliance: Target vs Measured BPM",
        fontsize=14, fontweight="bold", y=1.01, color=DARK
    )

    genres = ["Afrobeats", "Metal"]

    for gi, (ax, genre) in enumerate(zip(axes, genres)):
        rows = [r for r in TEMPO if r[0] == genre]
        tgt_lo = rows[0][2]
        tgt_hi = rows[0][3]

        ax.set_facecolor("#FAFAFA")
        ax.set_title(genre, fontsize=13, fontweight="bold", pad=12)

        # ── target shaded zone ───────────────────────────────────
        ax.axhspan(tgt_lo, tgt_hi, color="#E0E0E0", alpha=0.5, zorder=0,
                   label=f"Requested range\n({tgt_lo}–{tgt_hi} BPM)")
        ax.axhline(tgt_lo, color="#9E9E9E", linestyle="--", lw=1.5, zorder=1)
        ax.axhline(tgt_hi, color="#9E9E9E", linestyle="--", lw=1.5, zorder=1)

        # label targets on right
        ax.text(3.55, tgt_lo, f"Target LOW\n{tgt_lo} BPM",
                va="center", fontsize=8, color="#9E9E9E")
        ax.text(3.55, tgt_hi, f"Target HIGH\n{tgt_hi} BPM",
                va="center", fontsize=8, color="#9E9E9E")

        # ── plot each system ─────────────────────────────────────
        x_pos = np.arange(len(SYSTEMS))
        bar_w = 0.35

        for si, sys in enumerate(SYSTEMS):
            row = next(r for r in rows if r[1] == sys)
            _, _, tlo, thi, mlo, mhi = row
            correct = mhi > mlo
            col     = SYS_COLOR[sys]
            ec      = GREEN if correct else RED

            # low bar
            ax.bar(si - bar_w / 2, mlo, width=bar_w,
                   color=col, alpha=0.45,
                   edgecolor=ec, linewidth=2, zorder=3)
            # high bar
            ax.bar(si + bar_w / 2, mhi, width=bar_w,
                   color=col, alpha=0.9,
                   edgecolor=ec, linewidth=2, zorder=3)

            # value labels on bars
            ax.text(si - bar_w / 2, mlo + 2, f"{mlo:.0f}",
                    ha="center", fontsize=8.5, color=col, fontweight="bold")
            ax.text(si + bar_w / 2, mhi + 2, f"{mhi:.0f}",
                    ha="center", fontsize=8.5, color=col, fontweight="bold")

            # range ratio annotation
            rr = (mhi - mlo) / (thi - tlo)
            rr_col = GREEN if rr >= 0.6 else (AMBER if rr >= 0.0 else RED)
            ax.text(si, max(mlo, mhi) + (thi - tlo) * 0.08,
                    f"range ratio\n{rr:.2f}",
                    ha="center", fontsize=8, color=rr_col, fontweight="bold")

            # arrow connecting low to high bar
            arr_col = GREEN if correct else RED
            ax.annotate(
                "", xy=(si + bar_w / 2, mhi),
                xytext=(si - bar_w / 2, mlo),
                arrowprops=dict(arrowstyle="->", color=arr_col, lw=2),
                zorder=5
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(SYSTEMS, fontsize=11, fontweight="bold")
        for tick, sys in zip(ax.get_xticklabels(), SYSTEMS):
            tick.set_color(SYS_COLOR[sys])

        ax.set_ylabel("Measured Tempo (BPM)", fontsize=10)
        ax.set_xlim(-0.7, len(SYSTEMS) - 0.3 + 0.8)   # room for target labels
        ax.set_ylim(
            max(0, min(tgt_lo, min(r[4] for r in rows)) - 15),
            max(tgt_hi, max(r[5] for r in rows)) + (tgt_hi - tgt_lo) * 0.35
        )
        ax.grid(axis="y", alpha=0.3, zorder=0)

        # legend only on first panel
        if gi == 0:
            lo_p = mpatches.Patch(color="grey", alpha=0.45, label="LOW prompt output")
            hi_p = mpatches.Patch(color="grey", alpha=0.9,  label="HIGH prompt output")
            gr_p = mpatches.Patch(color=GREEN,  alpha=0.8,  label="Correct direction")
            rd_p = mpatches.Patch(color=RED,    alpha=0.8,  label="Wrong direction")
            ax.legend(handles=[lo_p, hi_p, gr_p, rd_p],
                      fontsize=8.5, loc="upper left", frameon=True)

    fig.text(
        0.5, -0.04,
        "Range ratio = (measured HIGH - measured LOW) / (target HIGH - target LOW)\n"
        "1.0 = perfect compliance   |   0.5 = half the requested spread produced   |   <0 = reversed\n"
        "n=1 per condition — descriptive pilot observations only.",
        ha="center", fontsize=8.5, color=MID, style="italic"
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved -> {out_path}")


# ════════════════════════════════════════════════════════════════
#  FIGURE 2: OTHER AXES
# ════════════════════════════════════════════════════════════════

def make_other_axes_figure(out_path):
    n   = len(OTHER)
    fig = plt.figure(figsize=(16, 4.2 * n + 1.5), facecolor="white")

    fig.suptitle(
        "Prompt-Following Validation — Non-Tempo Axes\n"
        "What was asked vs what was measured, and how much spread did each system produce?",
        fontsize=13, fontweight="bold", y=1.005, color=DARK
    )

    # One subplot per axis
    all_axes = []
    for i in range(n):
        ax = fig.add_subplot(n, 1, i + 1)
        all_axes.append(ax)

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

        # ── title ────────────────────────────────────────────────
        ax.set_title(
            f"{genre}  —  {axis}    "
            f"[Measured: {feat_name}]",
            fontsize=11, fontweight="bold", loc="left",
            color=DARK, pad=8
        )

        # ── compute range stats ──────────────────────────────────
        all_lo  = [slo, llo, ulo]
        all_hi  = [shi, lhi, uhi]
        obs_min = min(all_lo + all_hi)
        obs_max = max(all_lo + all_hi)
        obs_rng = obs_max - obs_min if obs_max != obs_min else 1e-9

        # ── horizontal bar chart showing spread ──────────────────
        # y positions: 0=Suno, 1=Lyria, 2=Udio (bottom to top)
        y_sys = {"Suno": 0, "Lyria": 1, "Udio": 2}
        bar_h = 0.55

        for sys in SYSTEMS:
            vlo, vhi  = vals[sys]
            y         = y_sys[sys]
            col       = SYS_COLOR[sys]
            correct   = vhi > vlo
            spread_pct= abs(vhi - vlo) / obs_rng * 100

            # background full-range bar (shows what's possible)
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

            # dots for low and high
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

            offset = obs_rng * 0.015
            ax.text(vlo - offset, y, fmt(vlo),
                    ha="right", va="center", fontsize=8.5,
                    color=col, fontweight="bold")
            ax.text(vhi + offset, y, fmt(vhi),
                    ha="left", va="center", fontsize=8.5,
                    color=col, fontweight="bold")

            # spread % and PASS/FAIL on right
            verdict   = "PASS" if correct else "FAIL"
            vrd_color = GREEN if correct else RED
            ax.text(obs_max + obs_rng * 0.35, y,
                    f"{verdict}   spread: {spread_pct:.0f}% of range",
                    va="center", fontsize=9,
                    fontweight="bold", color=vrd_color)

        # ── y axis labels = system names ─────────────────────────
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(SYSTEMS, fontsize=10, fontweight="bold")
        for tick, sys in zip(ax.get_yticklabels(), SYSTEMS):
            tick.set_color(SYS_COLOR[sys])

        ax.set_ylim(-0.5, 2.8)
        ax.set_xlim(obs_min - obs_rng * 0.25,
                    obs_max + obs_rng * 1.05)
        ax.set_xlabel(f"{feat_name}   ({feat_expl.split(chr(10))[0]})",
                      fontsize=9, color=MID)
        ax.grid(axis="x", alpha=0.3, zorder=0)

        # ── what was asked annotations ───────────────────────────
        # LOW label above left side, HIGH label above right side
        ax.text(obs_min - obs_rng * 0.22, 2.72,
                f"LOW prompt: {prompt_lo}",
                ha="left", va="bottom", fontsize=8,
                color="#616161", style="italic",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="#EEEEEE", edgecolor="#CCCCCC",
                          linewidth=0.8, alpha=0.9))

        ax.text(obs_max + obs_rng * 0.02, 2.72,
                f"HIGH prompt: {prompt_hi}",
                ha="left", va="bottom", fontsize=8,
                color="#616161", style="italic",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="#EEEEEE", edgecolor="#CCCCCC",
                          linewidth=0.8, alpha=0.9))

        # ── caveat ───────────────────────────────────────────────
        if caveat:
            ax.text(0.5, -0.22, caveat,
                    transform=ax.transAxes,
                    ha="center", fontsize=8,
                    color=AMBER, style="italic")

    # ── shared legend ────────────────────────────────────────────
    lo_dot  = plt.Line2D([0],[0], marker="o", color="grey",
                          markersize=8, linestyle="None",
                          label="LOW prompt output (circle)")
    hi_dot  = plt.Line2D([0],[0], marker="D", color="grey",
                          markersize=8, linestyle="None",
                          label="HIGH prompt output (diamond)")
    full_rng= mpatches.Patch(color="#EEEEEE", edgecolor="#CCCCCC",
                              label="Full observable range (all systems)")
    grn_bar = mpatches.Patch(color=GREEN, alpha=0.7,
                              label="PASS — correct direction")
    red_bar = mpatches.Patch(color=RED,   alpha=0.7,
                              label="FAIL — wrong direction")

    fig.legend(handles=[lo_dot, hi_dot, full_rng, grn_bar, red_bar],
               loc="lower center",
               bbox_to_anchor=(0.5, -0.025),
               ncol=5, fontsize=9, frameon=True,
               edgecolor="#E0E0E0")

    fig.text(
        0.5, -0.055,
        "Spread = how much of the observable range each system used.\n"
        "Low spread = both conditions produced similar output = homogenization.\n"
        "n=1 per condition — descriptive pilot observations only.",
        ha="center", fontsize=8.5, color=MID, style="italic"
    )

    fig.tight_layout(h_pad=3.5)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved -> {out_path}")


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    make_tempo_figure(out_dir / "fig_tempo_compliance.png")
    make_other_axes_figure(out_dir / "fig_other_axes_compliance.png")

    print("\nDone. Check your figures/ folder.")
