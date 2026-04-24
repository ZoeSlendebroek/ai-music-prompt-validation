#!/usr/bin/env python3
"""
table_all_results.py

Generates one clean colour-coded table of ALL prompt-following results.

Run:
    conda activate aijam
    python src/table_all_results.py

Output: figures/table_all_results.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── colours ──────────────────────────────────────────────────────
GREEN  = "#2E7D32"
GREEN2 = "#66BB6A"   # lighter green for partial pass
RED    = "#C62828"
RED2   = "#EF5350"   # lighter red
AMBER  = "#E65100"
DARK   = "#212121"
MID    = "#555555"
LGREY  = "#F5F5F5"
DGREY  = "#E0E0E0"
WHITE  = "#FFFFFF"

SYS_COLOR = {
    "Suno":  "#1565C0",
    "Lyria": "#2E7D32",
    "Udio":  "#BF360C",
}

# ════════════════════════════════════════════════════════════════
#  ALL RESULTS
#  One row per (genre, axis, system)
#  Columns: genre, axis, system,
#           what_asked_low, measured_low,
#           what_asked_high, measured_high,
#           feature, unit,
#           direction_correct, abs_diff, range_ratio,
#           note
# ════════════════════════════════════════════════════════════════

ROWS = [
    # ── AFROBEATS TEMPO ─────────────────────────────────────────
    # direction correct = measured_high > measured_low
    # range_ratio = (measured_high - measured_low) / (target_high - target_low)
    # abs_error = average of |measured_low - target_low| and |measured_high - target_high|
    {
        "genre": "Afrobeats", "axis": "Tempo",
        "system": "Suno",
        "asked_low": "80 BPM", "measured_low": "99.4 BPM",
        "asked_high": "120 BPM", "measured_high": "117.5 BPM",
        "feature": "Tempo (BPM)", "unit": "beat tracker",
        "direction": True,
        "abs_error_low": 19.4, "abs_error_high": 2.5,
        "range_ratio": (117.5 - 99.4) / (120 - 80),   # 0.45
        "note": "Direction correct but LOW too fast. Only 45% of requested spread.",
    },
    {
        "genre": "Afrobeats", "axis": "Tempo",
        "system": "Lyria",
        "asked_low": "80 BPM", "measured_low": "107.7 BPM",
        "asked_high": "120 BPM", "measured_high": "123.0 BPM",
        "feature": "Tempo (BPM)", "unit": "beat tracker",
        "direction": True,
        "abs_error_low": 27.7, "abs_error_high": 3.0,
        "range_ratio": (123.0 - 107.7) / (120 - 80),  # 0.38
        "note": "Direction correct. LOW badly off target. 38% of requested spread.",
    },
    {
        "genre": "Afrobeats", "axis": "Tempo",
        "system": "Udio",
        "asked_low": "80 BPM", "measured_low": "99.4 BPM",
        "asked_high": "120 BPM", "measured_high": "129.2 BPM",
        "feature": "Tempo (BPM)", "unit": "beat tracker",
        "direction": True,
        "abs_error_low": 19.4, "abs_error_high": 9.2,
        "range_ratio": (129.2 - 99.4) / (120 - 80),   # 0.75
        "note": "Best range compliance (75%). Both errors moderate.",
    },
    # ── AFROBEATS DENSITY ────────────────────────────────────────
    {
        "genre": "Afrobeats", "axis": "Density",
        "system": "Suno",
        "asked_low": "sparse percussion", "measured_low": "5.13 ev/s",
        "asked_high": "dense polyrhythmic", "measured_high": "5.97 ev/s",
        "feature": "Onset Density", "unit": "events / second",
        "direction": True,
        "abs_error_low": None, "abs_error_high": None,
        "range_ratio": (5.97 - 5.13) / (6.87 - 4.80),  # rel to full range
        "note": "Correct direction. Small absolute difference (0.84 ev/s).",
    },
    {
        "genre": "Afrobeats", "axis": "Density",
        "system": "Lyria",
        "asked_low": "sparse percussion", "measured_low": "4.80 ev/s",
        "asked_high": "dense polyrhythmic", "measured_high": "6.47 ev/s",
        "feature": "Onset Density", "unit": "events / second",
        "direction": True,
        "abs_error_low": None, "abs_error_high": None,
        "range_ratio": (6.47 - 4.80) / (6.87 - 4.80),
        "note": "Clearest density contrast. Largest spread (1.67 ev/s).",
    },
    {
        "genre": "Afrobeats", "axis": "Density",
        "system": "Udio",
        "asked_low": "sparse percussion", "measured_low": "6.07 ev/s",
        "asked_high": "dense polyrhythmic", "measured_high": "6.87 ev/s",
        "feature": "Onset Density", "unit": "events / second",
        "direction": True,
        "abs_error_low": None, "abs_error_high": None,
        "range_ratio": (6.87 - 6.07) / (6.87 - 4.80),
        "note": "Correct direction. LOW already dense — limited headroom.",
    },
    # ── AFROBEATS TEXTURE ────────────────────────────────────────
    {
        "genre": "Afrobeats", "axis": "Texture",
        "system": "Suno",
        "asked_low": "drum-forward,\npercussion-heavy", "measured_low": "0.588",
        "asked_high": "melodic synth,\nharmonic", "measured_high": "0.587",
        "feature": "Harmonic Ratio", "unit": "0=all drums  1=all melody",
        "direction": False,
        "abs_error_low": None, "abs_error_high": None,
        "range_ratio": abs(0.587 - 0.588) / (0.759 - 0.391),
        "note": "FAIL. Virtually identical outputs. Suno did not differentiate on texture.",
    },
    {
        "genre": "Afrobeats", "axis": "Texture",
        "system": "Lyria",
        "asked_low": "drum-forward,\npercussion-heavy", "measured_low": "0.475",
        "asked_high": "melodic synth,\nharmonic", "measured_high": "0.715",
        "feature": "Harmonic Ratio", "unit": "0=all drums  1=all melody",
        "direction": True,
        "abs_error_low": None, "abs_error_high": None,
        "range_ratio": (0.715 - 0.475) / (0.759 - 0.391),
        "note": "Clear contrast. 65% of observable range.",
    },
    {
        "genre": "Afrobeats", "axis": "Texture",
        "system": "Udio",
        "asked_low": "drum-forward,\npercussion-heavy", "measured_low": "0.391",
        "asked_high": "melodic synth,\nharmonic", "measured_high": "0.759",
        "feature": "Harmonic Ratio", "unit": "0=all drums  1=all melody",
        "direction": True,
        "abs_error_low": None, "abs_error_high": None,
        "range_ratio": (0.759 - 0.391) / (0.759 - 0.391),
        "note": "Strongest texture contrast. Full observable range.",
    },
    # ── AFROBEATS STRUCTURE ──────────────────────────────────────
    {
        "genre": "Afrobeats", "axis": "Structure",
        "system": "Suno",
        "asked_low": "loop-based,\nostinato", "measured_low": "0.139",
        "asked_high": "varied,\nevolving", "measured_high": "0.129",
        "feature": "Self-Similarity", "unit": "higher = more repetitive",
        "direction": True,
        "abs_error_low": None, "abs_error_high": None,
        "range_ratio": abs(0.139 - 0.129) / (0.148 - 0.100),
        "note": "Correct direction but tiny diff (0.010). 30s clips limit reliability.",
    },
    {
        "genre": "Afrobeats", "axis": "Structure",
        "system": "Lyria",
        "asked_low": "loop-based,\nostinato", "measured_low": "0.118",
        "asked_high": "varied,\nevolving", "measured_high": "0.117",
        "feature": "Self-Similarity", "unit": "higher = more repetitive",
        "direction": True,
        "abs_error_low": None, "abs_error_high": None,
        "range_ratio": abs(0.118 - 0.117) / (0.148 - 0.100),
        "note": "Correct direction but negligible diff (0.001). Near-zero spread.",
    },
    {
        "genre": "Afrobeats", "axis": "Structure",
        "system": "Udio",
        "asked_low": "loop-based,\nostinato", "measured_low": "0.148",
        "asked_high": "varied,\nevolving", "measured_high": "0.100",
        "feature": "Self-Similarity", "unit": "higher = more repetitive",
        "direction": True,
        "abs_error_low": None, "abs_error_high": None,
        "range_ratio": abs(0.148 - 0.100) / (0.148 - 0.100),
        "note": "Largest diff among systems (0.048) but still small in absolute terms.",
    },
    # ── METAL TEMPO ──────────────────────────────────────────────
    {
        "genre": "Metal", "axis": "Tempo",
        "system": "Suno",
        "asked_low": "75 BPM (doom)", "measured_low": "143.6 BPM",
        "asked_high": "180 BPM (thrash)", "measured_high": "143.6 BPM",
        "feature": "Tempo (BPM)", "unit": "beat tracker",
        "direction": False,
        "abs_error_low": 68.6, "abs_error_high": 36.4,
        "range_ratio": 0.0,
        "note": "FAIL. Identical output for both prompts. Stuck at ~144 BPM regardless.",
    },
    {
        "genre": "Metal", "axis": "Tempo",
        "system": "Lyria",
        "asked_low": "75 BPM (doom)", "measured_low": "152.0 BPM",
        "asked_high": "180 BPM (thrash)", "measured_high": "92.3 BPM",
        "feature": "Tempo (BPM)", "unit": "beat tracker",
        "direction": False,
        "abs_error_low": 77.0, "abs_error_high": 87.7,
        "range_ratio": (92.3 - 152.0) / (180 - 75),  # negative = reversed
        "note": "FAIL + REVERSED. Doom faster than thrash. Subgenre overrides BPM instruction.",
    },
    {
        "genre": "Metal", "axis": "Tempo",
        "system": "Udio",
        "asked_low": "75 BPM (doom)", "measured_low": "129.2 BPM",
        "asked_high": "180 BPM (thrash)", "measured_high": "99.4 BPM",
        "feature": "Tempo (BPM)", "unit": "beat tracker",
        "direction": False,
        "abs_error_low": 54.2, "abs_error_high": 80.6,
        "range_ratio": (99.4 - 129.2) / (180 - 75),  # negative = reversed
        "note": "FAIL + REVERSED. Same pattern as Lyria — subgenre label dominates.",
    },
    # ── METAL DENSITY ────────────────────────────────────────────
    {
        "genre": "Metal", "axis": "Density",
        "system": "Suno",
        "asked_low": "sparse drumming,\ngroove-focused", "measured_low": "4.43 ev/s",
        "asked_high": "blast beats,\nextreme density", "measured_high": "5.70 ev/s",
        "feature": "Onset Density", "unit": "events / second",
        "direction": True,
        "abs_error_low": None, "abs_error_high": None,
        "range_ratio": (5.70 - 4.43) / (6.17 - 2.90),
        "note": "Correct direction. Moderate contrast.",
    },
    {
        "genre": "Metal", "axis": "Density",
        "system": "Lyria",
        "asked_low": "sparse drumming,\ngroove-focused", "measured_low": "2.90 ev/s",
        "asked_high": "blast beats,\nextreme density", "measured_high": "6.17 ev/s",
        "feature": "Onset Density", "unit": "events / second",
        "direction": True,
        "abs_error_low": None, "abs_error_high": None,
        "range_ratio": (6.17 - 2.90) / (6.17 - 2.90),
        "note": "Strongest density contrast across all systems and genres.",
    },
    {
        "genre": "Metal", "axis": "Density",
        "system": "Udio",
        "asked_low": "sparse drumming,\ngroove-focused", "measured_low": "6.57 ev/s",
        "asked_high": "blast beats,\nextreme density", "measured_high": "5.93 ev/s",
        "feature": "Onset Density", "unit": "events / second",
        "direction": False,
        "abs_error_low": None, "abs_error_high": None,
        "range_ratio": (5.93 - 6.57) / (6.17 - 2.90),
        "note": "FAIL + REVERSED. Sparse prompt produced denser output than blast beats prompt.",
    },
    # ── METAL DISTORTION ─────────────────────────────────────────
    {
        "genre": "Metal", "axis": "Distortion",
        "system": "Suno",
        "asked_low": "clean guitar,\nno distortion", "measured_low": "0.0366",
        "asked_high": "heavy saturation,\nfuzz", "measured_high": "0.0283",
        "feature": "Spectral Flatness", "unit": "higher = noisier / more distorted",
        "direction": False,
        "abs_error_low": None, "abs_error_high": None,
        "range_ratio": (0.0283 - 0.0366) / (0.0397 - 0.0147),
        "note": "FAIL + REVERSED. Clean prompt produced more distorted output.",
    },
    {
        "genre": "Metal", "axis": "Distortion",
        "system": "Lyria",
        "asked_low": "clean guitar,\nno distortion", "measured_low": "0.0337",
        "asked_high": "heavy saturation,\nfuzz", "measured_high": "0.0397",
        "feature": "Spectral Flatness", "unit": "higher = noisier / more distorted",
        "direction": True,
        "abs_error_low": None, "abs_error_high": None,
        "range_ratio": (0.0397 - 0.0337) / (0.0397 - 0.0147),
        "note": "Correct direction. Small absolute difference (0.006).",
    },
    {
        "genre": "Metal", "axis": "Distortion",
        "system": "Udio",
        "asked_low": "clean guitar,\nno distortion", "measured_low": "0.0147",
        "asked_high": "heavy saturation,\nfuzz", "measured_high": "0.0245",
        "feature": "Spectral Flatness", "unit": "higher = noisier / more distorted",
        "direction": True,
        "abs_error_low": None, "abs_error_high": None,
        "range_ratio": (0.0245 - 0.0147) / (0.0397 - 0.0147),
        "note": "Correct direction. Largest absolute spread for distortion.",
    },
]

# ════════════════════════════════════════════════════════════════
#  BUILD FIGURE
# ════════════════════════════════════════════════════════════════

n_rows = len(ROWS)

# column definitions: (header, width_ratio)
COLS = [
    ("Genre",             0.055),
    ("Axis",              0.055),
    ("System",            0.055),
    ("Asked LOW",         0.100),
    ("Measured\nLOW",     0.080),
    ("Asked HIGH",        0.100),
    ("Measured\nHIGH",    0.080),
    ("Feature\n(unit)",   0.090),
    ("Direction\ncorrect",0.060),
    ("BPM error\n(LOW / HIGH)", 0.075),
    ("Range\nratio",      0.060),
    ("Notes",             0.190),
]

col_names  = [c[0] for c in COLS]
col_widths = [c[1] for c in COLS]
col_x      = np.cumsum([0] + col_widths[:-1])   # left edge of each col

ROW_H   = 0.042    # height of each data row in figure coords
HDR_H   = 0.048    # header row height
PAD_TOP = 0.06
PAD_BOT = 0.06
PAD_LR  = 0.01

fig_h = (n_rows * ROW_H + HDR_H + PAD_TOP + PAD_BOT) * 26
fig   = plt.figure(figsize=(22, fig_h), facecolor="white")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size":   9,
})

fig.text(0.5, 0.995,
         "Prompt-Following Validation — All Results",
         ha="center", va="top",
         fontsize=16, fontweight="bold", color=DARK)
fig.text(0.5, 0.988,
         "n=1 per condition — descriptive pilot observations. "
         "Range ratio: 1.0 = full requested spread produced; "
         "<0.5 = compressed (homogenized); <0 = reversed.",
         ha="center", va="top", fontsize=10, color=MID, style="italic")

# coordinate system: figure coords 0–1
# rows go from top down
y_hdr = 1.0 - PAD_TOP / fig_h * 26   # normalised

def fig_y(row_i):
    """Top of data row i in figure coords."""
    return y_hdr - HDR_H / fig_h * 26 - row_i * ROW_H / fig_h * 26

def draw_rect(x, y_top, w, h, fc, ec="#CCCCCC", lw=0.6, alpha=1.0, zorder=1):
    r = plt.Rectangle((x, y_top - h), w, h,
                       transform=fig.transFigure,
                       facecolor=fc, edgecolor=ec,
                       linewidth=lw, clip_on=False,
                       alpha=alpha, zorder=zorder)
    fig.add_artist(r)

def draw_text(x, y_top, h, text, ha="center", va="center",
              fontsize=9, bold=False, color=DARK, style="normal",
              zorder=2):
    fig.text(x, y_top - h / 2,
             text, ha=ha, va="center",
             fontsize=fontsize,
             fontweight="bold" if bold else "normal",
             color=color, style=style,
             transform=fig.transFigure,
             clip_on=False, zorder=zorder)

# ── header row ───────────────────────────────────────────────────
for ci, (name, w) in enumerate(zip(col_names, col_widths)):
    x = PAD_LR + col_x[ci]
    draw_rect(x, y_hdr, w - 0.003, HDR_H / fig_h * 26,
              fc="#1A237E", ec=WHITE, lw=1.0)
    draw_text(x + w / 2, y_hdr, HDR_H / fig_h * 26,
              name, bold=True, color=WHITE, fontsize=9)

# ── data rows ────────────────────────────────────────────────────
prev_genre = None
prev_axis  = None

for ri, row in enumerate(ROWS):
    yt   = fig_y(ri)
    rh   = ROW_H / fig_h * 26
    alt  = "#F8F8F8" if ri % 2 == 0 else WHITE

    genre  = row["genre"]
    axis   = row["axis"]
    system = row["system"]
    direct = row["direction"]
    rr     = row["range_ratio"]

    # cell background colour logic
    def cell_bg(col_idx):
        """Return background colour for a given column."""
        if col_idx in (8,):   # direction
            return GREEN if direct else RED
        if col_idx in (10,):  # range ratio
            if rr is None:    return alt
            if rr >= 0.6:     return "#C8E6C9"   # light green
            if rr >= 0.0:     return "#FFF9C4"   # light yellow
            return "#FFCDD2"                       # light red
        return alt

    for ci, (name, w) in enumerate(zip(col_names, col_widths)):
        x  = PAD_LR + col_x[ci]
        bg = cell_bg(ci)
        draw_rect(x, yt, w - 0.003, rh, fc=bg)

    # ── cell content ─────────────────────────────────────────────
    def tx(ci, text, ha="center", bold=False, color=DARK,
           fontsize=9, style="normal"):
        x = PAD_LR + col_x[ci] + col_widths[ci] / 2
        draw_text(x, yt, rh, text,
                  ha=ha, bold=bold, color=color,
                  fontsize=fontsize, style=style)

    # genre (only show when changes)
    tx(0, genre if genre != prev_genre else "",
       bold=True, color="#37474F")

    # axis (only show when changes within genre)
    tx(1, axis if (axis != prev_axis or genre != prev_genre) else "",
       bold=True, color="#455A64")

    # system
    tx(2, system, bold=True, color=SYS_COLOR[system])

    # asked low / high
    tx(3, row["asked_low"],  fontsize=8, style="italic", color="#424242")
    tx(5, row["asked_high"], fontsize=8, style="italic", color="#424242")

    # measured low / high
    tx(4, row["measured_low"],  bold=True, color=SYS_COLOR[system])
    tx(6, row["measured_high"], bold=True, color=SYS_COLOR[system])

    # feature
    feat_txt = f"{row['feature']}\n({row['unit']})"
    tx(7, feat_txt, fontsize=7.5, color=MID)

    # direction
    verdict = "PASS" if direct else "FAIL"
    tx(8, verdict, bold=True,
       color=WHITE, fontsize=10)

    # BPM error
    elo = row["abs_error_low"]
    ehi = row["abs_error_high"]
    if elo is not None:
        err_txt = f"{elo:.1f} / {ehi:.1f} BPM"
        err_col = GREEN if max(elo, ehi) < 15 else (AMBER if max(elo, ehi) < 40 else RED)
        tx(9, err_txt, bold=True,
           color=err_col if direct else RED, fontsize=8.5)
    else:
        tx(9, "N/A", color="#9E9E9E", fontsize=8.5)

    # range ratio
    if rr is not None:
        rr_txt = f"{rr:.2f}"
        if rr < 0:
            rr_txt = f"{rr:.2f}\n(reversed)"
        rr_col = GREEN if rr >= 0.6 else (AMBER if rr >= 0.0 else RED)
        tx(10, rr_txt, bold=True, color=rr_col, fontsize=9)
    else:
        tx(10, "—", color=MID)

    # notes
    x_note = PAD_LR + col_x[11] + 0.006
    draw_text(x_note, yt, rh,
              row["note"],
              ha="left", fontsize=7.5,
              color="#424242", style="italic")

    # horizontal divider
    draw_rect(PAD_LR, yt, sum(col_widths) - 0.003, 0.0008,
              fc="#CCCCCC", ec="none", lw=0)

    prev_genre = genre
    prev_axis  = axis

# ── genre group dividers ─────────────────────────────────────────
# thicker line between Afrobeats and Metal blocks
afrobeats_end = sum(1 for r in ROWS if r["genre"] == "Afrobeats")
y_divider = fig_y(afrobeats_end)
draw_rect(PAD_LR, y_divider + ROW_H / fig_h * 26 * 0.05,
          sum(col_widths) - 0.003, 0.003,
          fc="#1A237E", ec="none", lw=0, zorder=5)

# genre labels on left margin
y_ab_mid = (y_hdr - HDR_H / fig_h * 26 + y_divider) / 2
y_me_mid = (y_divider + fig_y(n_rows)) / 2

fig.text(0.003, y_ab_mid, "AFROBEATS",
         ha="center", va="center",
         fontsize=11, fontweight="bold",
         color="#1A237E", rotation=90,
         transform=fig.transFigure)
fig.text(0.003, y_me_mid, "METAL",
         ha="center", va="center",
         fontsize=11, fontweight="bold",
         color="#B71C1C", rotation=90,
         transform=fig.transFigure)

# ── legend ───────────────────────────────────────────────────────
y_leg = fig_y(n_rows) - 0.015

fig.text(PAD_LR, y_leg,
         "Direction:  PASS = measured HIGH > measured LOW (correct direction)    "
         "FAIL = wrong direction or no change\n"
         "Range ratio:  >= 0.6 = good compliance (green)    "
         "0.0 – 0.59 = compressed output / homogenized (yellow)    "
         "< 0 = reversed (red)\n"
         "BPM error:  absolute difference between measured and target BPM    "
         "Structure axis: self-similarity unreliable at 30s — treat as indicative only",
         ha="left", va="top",
         fontsize=8.5, color=MID, style="italic",
         transform=fig.transFigure)

# ── save ─────────────────────────────────────────────────────────
out_dir = Path(__file__).resolve().parent.parent / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
out     = out_dir / "table_all_results.png"

fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
