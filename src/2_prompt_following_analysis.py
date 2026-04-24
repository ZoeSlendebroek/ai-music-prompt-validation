#!/usr/bin/env python3
"""
2_prompt_following_analysis.py

Analyses whether and how well AI music systems follow prompt instructions.

For each axis (tempo, density, texture, distortion, structure):
  - Compares LOW vs HIGH prompt outputs using Mann-Whitney U test
  - Computes effect size (Cliff's delta)
  - Calculates directional accuracy (did output go the right way?)
  - Runs across all systems and genres

Inputs:
    data/features/all_tracks_features.csv

Outputs:
    data/results/prompt_following_summary.csv   — one row per axis × system × genre
    data/results/track_level_results.csv        — one row per track with validation score
    data/results/directional_accuracy.csv       — did each system follow direction?

Usage:
    python src/2_prompt_following_analysis.py
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# ── validation config ────────────────────────────────────────────
# For each axis_type, which feature to measure and in which direction
# (high prompt → higher feature value, low prompt → lower feature value)
AXIS_CONFIG = {
    "tempo": {
        "feature":   "tempo",
        "low_dir":   "low",
        "high_dir":  "high",
        "label":     "Tempo (BPM)",
        "unit":      "BPM",
        "expected_low":  85,    # ~80 BPM target
        "expected_high": 115,   # ~120 BPM target
    },
    "density": {
        "feature":   "onset_density",
        "low_dir":   "low",
        "high_dir":  "high",
        "label":     "Onset Density (events/s)",
        "unit":      "onsets/s",
        "expected_low":  None,
        "expected_high": None,
    },
    "texture": {
        "feature":   "harmonic_ratio",
        "low_dir":   "percussive",   # percussive = low harmonic
        "high_dir":  "melodic",      # melodic    = high harmonic
        "label":     "Harmonic Ratio",
        "unit":      "",
        "expected_low":  None,
        "expected_high": None,
    },
    "structure": {
        "feature":   "self_similarity_mean",
        "low_dir":   "varied",   # varied = low self-similarity
        "high_dir":  "loop",     # loop   = high self-similarity
        "label":     "Self-Similarity (loop vs varied)",
        "unit":      "",
        "expected_low":  None,
        "expected_high": None,
    },
    "distortion": {
        "feature":   "spectral_flatness_mean",
        "low_dir":   "low",    # clean   = low flatness
        "high_dir":  "high",   # saturated = high flatness
        "label":     "Spectral Flatness (distortion proxy)",
        "unit":      "",
        "expected_low":  None,
        "expected_high": None,
    },
}

def cliffs_delta(a, b):
    """Cliff's delta: P(a>b) - P(a<b). Range [-1,1]."""
    a, b   = np.asarray(a), np.asarray(b)
    greater = np.sum([np.sum(ai > b) for ai in a])
    less    = np.sum([np.sum(ai < b) for ai in a])
    return (greater - less) / (len(a) * len(b))

def interpret_cliffs(d):
    ad = abs(d)
    if ad < 0.147:  return "negligible"
    if ad < 0.330:  return "small"
    if ad < 0.474:  return "medium"
    return "large"

def analyse_axis(df_axis, axis_type, system, genre):
    """
    Compare low vs high prompt outputs for one axis × system × genre.
    Returns a result dict.
    """
    cfg = AXIS_CONFIG[axis_type]
    feat     = cfg["feature"]
    low_dir  = cfg["low_dir"]
    high_dir = cfg["high_dir"]

    low_vals  = df_axis[df_axis["axis_direction"] == low_dir][feat].dropna().values
    high_vals = df_axis[df_axis["axis_direction"] == high_dir][feat].dropna().values

    if len(low_vals) == 0 or len(high_vals) == 0:
        return None

    # Mann-Whitney U (two-sided)
    mw_stat, mw_p = stats.mannwhitneyu(low_vals, high_vals, alternative="two-sided")

    # Cliff's delta (positive = high > low, as expected)
    cd = cliffs_delta(high_vals, low_vals)

    # Directional accuracy: did high prompt produce higher values than low?
    direction_correct = np.mean(high_vals) > np.mean(low_vals)

    # For tempo: also check absolute accuracy against target BPM
    abs_error_low  = None
    abs_error_high = None
    if cfg["expected_low"] is not None:
        abs_error_low  = float(np.mean(np.abs(low_vals  - cfg["expected_low"])))
        abs_error_high = float(np.mean(np.abs(high_vals - cfg["expected_high"])))

    return {
        "system":             system,
        "genre":              genre,
        "axis_type":          axis_type,
        "feature":            feat,
        "n_low":              len(low_vals),
        "n_high":             len(high_vals),
        "mean_low":           float(np.mean(low_vals)),
        "mean_high":          float(np.mean(high_vals)),
        "std_low":            float(np.std(low_vals)),
        "std_high":           float(np.std(high_vals)),
        "mean_diff":          float(np.mean(high_vals) - np.mean(low_vals)),
        "direction_correct":  direction_correct,
        "cliffs_delta":       float(cd),
        "effect_size_label":  interpret_cliffs(cd),
        "mw_statistic":       float(mw_stat),
        "mw_pvalue":          float(mw_p),
        "significant_p05":    mw_p < 0.05,
        "abs_error_low_bpm":  abs_error_low,
        "abs_error_high_bpm": abs_error_high,
    }


def main():
    project_root = Path(__file__).resolve().parent.parent
    feat_csv     = project_root / "data" / "features" / "all_tracks_features.csv"
    out_dir      = project_root / "data" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PROMPT VALIDATION — FOLLOWING ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(feat_csv)
    print(f"Loaded {len(df)} tracks")

    systems = df["system"].unique()
    genres  = df["genre"].unique()

    summary_rows = []

    for system in sorted(systems):
        for genre in sorted(genres):
            df_sg = df[(df["system"] == system) & (df["genre"] == genre)]
            if df_sg.empty:
                continue

            print(f"\n── {system.upper()} / {genre} ──")

            # get axis types present for this genre
            axis_types = df_sg["axis_type"].unique()
            axis_types = [a for a in axis_types if a in AXIS_CONFIG and a != "combined"]

            for axis_type in sorted(axis_types):
                df_ax = df_sg[df_sg["axis_type"] == axis_type]
                result = analyse_axis(df_ax, axis_type, system, genre)
                if result is None:
                    print(f"  [{axis_type}] insufficient data")
                    continue

                summary_rows.append(result)

                # print summary
                tick = "✓" if result["direction_correct"] else "✗"
                sig  = "p<.05" if result["significant_p05"] else "n.s."
                print(f"  [{axis_type:12s}] {tick} direction={result['direction_correct']} | "
                      f"Δ={result['mean_diff']:+.3f} | "
                      f"cliff's d={result['cliffs_delta']:+.3f} ({result['effect_size_label']}) | "
                      f"{sig}")
                if result["abs_error_low_bpm"] is not None:
                    print(f"               tempo error: LOW={result['abs_error_low_bpm']:.1f} BPM, "
                          f"HIGH={result['abs_error_high_bpm']:.1f} BPM")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "prompt_following_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n{'='*70}")
    print(f"Saved summary → {summary_csv}")

    # ── overall directional accuracy per system ──────────────────
    print(f"\n{'='*70}")
    print("DIRECTIONAL ACCURACY BY SYSTEM")
    print("=" * 70)
    acc = summary_df.groupby("system")["direction_correct"].agg(["sum","count","mean"])
    acc.columns = ["correct", "total", "accuracy"]
    acc["accuracy_pct"] = (acc["accuracy"] * 100).round(1)
    print(acc.to_string())

    acc_csv = out_dir / "directional_accuracy.csv"
    acc.reset_index().to_csv(acc_csv, index=False)

    # ── track-level: score each track against its target ────────
    track_rows = []
    for _, row in df.iterrows():
        axis_type = row["axis_type"]
        if axis_type not in AXIS_CONFIG or axis_type == "combined":
            continue
        cfg  = AXIS_CONFIG[axis_type]
        feat = cfg["feature"]
        val  = row.get(feat, np.nan)

        # find the counterpart mean from summary
        match = summary_df[
            (summary_df["system"]    == row["system"]) &
            (summary_df["genre"]     == row["genre"])  &
            (summary_df["axis_type"] == axis_type)
        ]
        if match.empty:
            continue
        m = match.iloc[0]

        # is this track on the correct side of the midpoint?
        midpoint = (m["mean_low"] + m["mean_high"]) / 2
        is_low   = row["axis_direction"] in [cfg["low_dir"]]
        if is_low:
            on_correct_side = val < midpoint
        else:
            on_correct_side = val > midpoint

        track_rows.append({
            "filename":        row["filename"],
            "system":          row["system"],
            "genre":           row["genre"],
            "prompt_id":       row["prompt_id"],
            "axis_type":       axis_type,
            "axis_direction":  row["axis_direction"],
            "feature":         feat,
            "feature_value":   float(val) if not np.isnan(val) else None,
            "midpoint":        float(midpoint),
            "on_correct_side": on_correct_side,
        })

    track_df = pd.DataFrame(track_rows)
    track_csv = out_dir / "track_level_results.csv"
    track_df.to_csv(track_csv, index=False)
    print(f"\nSaved track-level results → {track_csv}")

    # ── print final verdict ──────────────────────────────────────
    print(f"\n{'='*70}")
    print("OVERALL VERDICT")
    print("=" * 70)
    for system in sorted(systems):
        sub = summary_df[summary_df["system"] == system]
        n_correct = sub["direction_correct"].sum()
        n_total   = len(sub)
        n_sig     = sub["significant_p05"].sum()
        large_fx  = (sub["effect_size_label"].isin(["large","medium"])).sum()
        print(f"\n{system.upper()}:")
        print(f"  Direction correct:    {n_correct}/{n_total} ({n_correct/n_total*100:.0f}%)")
        print(f"  Statistically sig:    {n_sig}/{n_total}")
        print(f"  Medium/large effect:  {large_fx}/{n_total}")


if __name__ == "__main__":
    main()
