#!/usr/bin/env python3
"""
0_run_pipeline.py

Master runner for the prompt-following validation study.
Runs all three steps in order.

Usage:
    python src/0_run_pipeline.py
"""

import sys
import subprocess
from pathlib import Path

print("""
================================================================================
PROMPT-FOLLOWING VALIDATION PIPELINE
AI Music Generation Systems — Afrobeats & Metal
================================================================================
Step 1: Extract 67 MIR features from all tracks (30s middle crop)
Step 2: Analyse prompt-following (direction accuracy, effect sizes, significance)
Step 3: Generate figures
================================================================================
""")

src = Path(__file__).resolve().parent

steps = [
    ("1_extract_features.py",       "Feature Extraction"),
    ("2_prompt_following_analysis.py","Prompt-Following Analysis"),
    ("3_visualise.py",              "Visualisation"),
]

results = []
for script, name in steps:
    print(f"\n{'='*70}")
    print(f"RUNNING: {name}")
    print("="*70)
    ret = subprocess.run([sys.executable, str(src / script)], capture_output=False)
    success = ret.returncode == 0
    results.append((name, success))
    if not success:
        print(f"\n⚠ {name} failed — check errors above before continuing.")
        break

print(f"\n{'='*70}")
print("PIPELINE SUMMARY")
print("="*70)
for name, ok in results:
    print(f"  {'✓' if ok else '✗'}  {name}")

if all(ok for _, ok in results):
    print("""
All steps complete. Outputs:
  data/features/all_tracks_features.csv
  data/results/prompt_following_summary.csv
  data/results/directional_accuracy.csv
  data/results/track_level_results.csv
  figures/fig1_direction_heatmap.png
  figures/fig2_effect_sizes.png
  figures/fig3_tempo_accuracy.png
  figures/fig4_distributions_afrobeats.png
  figures/fig4_distributions_metal.png
  figures/fig5_radar.png
  figures/fig6_accuracy_bars.png
""")
