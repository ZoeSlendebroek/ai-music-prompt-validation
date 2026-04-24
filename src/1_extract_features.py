#!/usr/bin/env python3
"""
1_extract_features.py

Extract 67 MIR features from all generated tracks across systems and genres.
All audio is cropped/padded to exactly 30 seconds (middle crop) for comparability.

Inputs:
    data/audio/{system}/{genre}/*.{mp3,wav,m4a,flac}
    systems: suno, lyria, udio
    genres:  afrobeats, metal

Outputs:
    data/features/all_tracks_features.csv

Usage:
    python src/1_extract_features.py
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import librosa

# ── prompt metadata ─────────────────────────────────────────────
# Maps prompt_id → (genre, axis_varied, axis_type, axis_direction)
# axis_type:      tempo | density | texture | distortion | combined
# axis_direction: low | high | percussive | melodic | loop | varied | combined_low | combined_high

PROMPT_META = {
    # AFROBEATS
    "A01": ("afrobeats", "Tempo LOW",      "tempo",     "low"),
    "A02": ("afrobeats", "Tempo HIGH",     "tempo",     "high"),
    "A03": ("afrobeats", "Density LOW",    "density",   "low"),
    "A04": ("afrobeats", "Density HIGH",   "density",   "high"),
    "A05": ("afrobeats", "Texture PERCUSSIVE", "texture", "percussive"),
    "A06": ("afrobeats", "Texture MELODIC",    "texture", "melodic"),
    "A07": ("afrobeats", "Structure LOOP",     "structure","loop"),
    "A08": ("afrobeats", "Structure VARIED",   "structure","varied"),
    "A09": ("afrobeats", "Combined LOW",   "combined",  "combined_low"),
    "A10": ("afrobeats", "Combined HIGH",  "combined",  "combined_high"),
    # METAL
    "M01": ("metal", "Tempo LOW (doom)",         "tempo",      "low"),
    "M02": ("metal", "Tempo HIGH (thrash)",       "tempo",      "high"),
    "M03": ("metal", "Distortion LOW (clean)",    "distortion", "low"),
    "M04": ("metal", "Distortion HIGH (saturated)","distortion","high"),
    "M05": ("metal", "Density LOW (groove)",      "density",    "low"),
    "M06": ("metal", "Density HIGH (blast beats)","density",    "high"),
    "M07": ("metal", "Combined: doom+clean+sparse","combined",  "combined_low"),
    "M08": ("metal", "Combined: thrash+saturated+dense","combined","combined_high"),
    "M09": ("metal", "Combined: moderate+saturated+groove","combined","combined_mid1"),
    "M10": ("metal", "Combined: moderate+clean+sparse",   "combined","combined_mid2"),
}

# For each axis type, which MIR feature is the PRIMARY validation target
# and what direction should it move (low prompt → lower value)
VALIDATION_TARGETS = {
    # axis_type → {direction: (feature, expected_direction)}
    "tempo":      {"low": ("tempo", "low"),       "high": ("tempo", "high")},
    "density":    {"low": ("onset_density", "low"), "high": ("onset_density", "high")},
    "texture":    {"percussive": ("harmonic_ratio", "low"),   # low harmonic = percussive
                   "melodic":    ("harmonic_ratio", "high")},
    "structure":  {"loop":   ("self_similarity_mean", "high"),
                   "varied": ("self_similarity_mean", "low")},
    "distortion": {"low":  ("spectral_flatness_mean", "low"),
                   "high": ("spectral_flatness_mean", "high")},
}

# ── feature extractor (identical to homogenisation pipeline) ────
class FeatureExtractor:
    def __init__(self, sr=22050, n_mfcc=13):
        self.sr     = sr
        self.n_mfcc = n_mfcc

    def load_30s(self, path):
        y, sr = librosa.load(str(path), sr=self.sr)
        target = int(30 * sr)
        if len(y) > target:
            mid   = len(y) // 2
            start = max(0, mid - target // 2)
            end   = start + target
            if end > len(y):
                end   = len(y)
                start = end - target
            y = y[start:end]
        elif len(y) < target:
            pad   = target - len(y)
            y     = np.pad(y, (pad // 2, pad - pad // 2), mode="constant")
        return y, sr

    def extract(self, path):
        y, sr = self.load_30s(path)
        f = {}
        f.update(self._spectral(y, sr))
        f.update(self._timbral(y, sr))
        f.update(self._rhythmic(y, sr))
        f.update(self._harmonic(y, sr))
        f.update(self._structural(y, sr))
        f.update(self._dynamic(y, sr))
        f.update(self._mfcc(y, sr))
        return f

    def _spectral(self, y, sr):
        sc  = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        bw  = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        ro  = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        fl  = librosa.feature.spectral_flatness(y=y)[0]
        con = librosa.feature.spectral_contrast(y=y, sr=sr)
        return {
            "spectral_centroid_mean": float(np.mean(sc)),
            "spectral_centroid_std":  float(np.std(sc)),
            "spectral_centroid_var":  float(np.var(sc)),
            "spectral_bandwidth_mean":float(np.mean(bw)),
            "spectral_bandwidth_std": float(np.std(bw)),
            "spectral_rolloff_mean":  float(np.mean(ro)),
            "spectral_rolloff_std":   float(np.std(ro)),
            "spectral_flatness_mean": float(np.mean(fl)),
            "spectral_flatness_std":  float(np.std(fl)),
            "spectral_contrast_mean": float(np.mean(con)),
            "spectral_contrast_std":  float(np.std(con)),
        }

    def _timbral(self, y, sr):
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        yh, yp   = librosa.effects.hpss(y)
        h_en = float(np.sum(yh**2))
        p_en = float(np.sum(yp**2))
        tot  = h_en + p_en
        return {
            "zero_crossing_rate_mean": float(np.mean(zcr)),
            "zero_crossing_rate_std":  float(np.std(zcr)),
            "harmonic_ratio":   h_en / tot if tot > 0 else 0.0,
            "percussive_ratio": p_en / tot if tot > 0 else 0.0,
        }

    def _rhythmic(self, y, sr):
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo    = float(tempo)
        oe       = librosa.onset.onset_strength(y=y, sr=sr)
        of       = librosa.onset.onset_detect(onset_envelope=oe, sr=sr)
        if len(of) > 1:
            ot  = librosa.frames_to_time(of, sr=sr)
            ioi = np.diff(ot)
            od  = len(of) / (len(y) / sr)
        else:
            ioi = np.array([0.0])
            od  = 0.0
        ioi_m = float(np.mean(ioi))
        ioi_s = float(np.std(ioi))
        return {
            "tempo":               tempo,
            "onset_density":       float(od),
            "onset_strength_mean": float(np.mean(oe)),
            "onset_strength_std":  float(np.std(oe)),
            "ioi_mean":            ioi_m,
            "ioi_std":             ioi_s,
            "ioi_cv":              ioi_s / ioi_m if ioi_m > 0 else 0.0,
        }

    def _harmonic(self, y, sr):
        ch  = librosa.feature.chroma_stft(y=y, sr=sr)
        cq  = librosa.feature.chroma_cqt(y=y, sr=sr)
        tn  = librosa.feature.tonnetz(y=y, sr=sr)
        return {
            "chroma_stft_mean": float(np.mean(ch)),
            "chroma_stft_std":  float(np.std(ch)),
            "chroma_stft_var":  float(np.var(ch)),
            "chroma_cqt_mean":  float(np.mean(cq)),
            "chroma_cqt_std":   float(np.std(cq)),
            "tonnetz_mean":     float(np.mean(tn)),
            "tonnetz_std":      float(np.std(tn)),
        }

    def _structural(self, y, sr):
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        _, beats = librosa.beat.beat_track(y=y, sr=sr)
        ms   = librosa.util.sync(mfcc, beats) if beats is not None and len(beats) >= 2 else mfcc
        R    = librosa.segment.recurrence_matrix(ms, mode="affinity")
        diag = float(np.mean(np.diag(R))) if R.shape[0] > 0 else 0.0
        return {
            "repetition_score":      float(np.mean(R) - diag),
            "self_similarity_mean":  float(np.mean(R)) if R.size else 0.0,
            "self_similarity_std":   float(np.std(R))  if R.size else 0.0,
        }

    def _dynamic(self, y, sr):
        rms  = librosa.feature.rms(y=y)[0]
        # FIXED: ref=1.0 gives absolute dB (not normalised to track max)
        db   = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=1.0)
        msq  = float(np.mean(y**2))
        return {
            "rms_mean":        float(np.mean(rms)),
            "rms_std":         float(np.std(rms)),
            "rms_var":         float(np.var(rms)),
            "dynamic_range_db":float(np.max(db) - np.min(db)) if db.size else 0.0,
            "crest_factor":    float(np.max(np.abs(y)) / np.sqrt(msq)) if msq > 0 else 0.0,
        }

    def _mfcc(self, y, sr):
        mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        d1    = librosa.feature.delta(mfcc)
        d2    = librosa.feature.delta(mfcc, order=2)
        out   = {}
        for i in range(self.n_mfcc):
            out[f"mfcc_{i}_mean"] = float(np.mean(mfcc[i]))
            out[f"mfcc_{i}_std"]  = float(np.std(mfcc[i]))
        out["mfcc_delta_mean"]  = float(np.mean(d1))
        out["mfcc_delta_std"]   = float(np.std(d1))
        out["mfcc_delta2_mean"] = float(np.mean(d2))
        out["mfcc_delta2_std"]  = float(np.std(d2))
        return out


# ── main ────────────────────────────────────────────────────────
def main():
    project_root = Path(__file__).resolve().parent.parent
    audio_root   = project_root / "data" / "audio"
    out_csv      = project_root / "data" / "features" / "all_tracks_features.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    extractor = FeatureExtractor()
    exts      = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"}
    rows      = []

    systems = ["suno", "lyria", "udio"]
    genres  = ["afrobeats", "metal"]

    print("=" * 70)
    print("PROMPT VALIDATION — FEATURE EXTRACTION")
    print("=" * 70)

    for system in systems:
        for genre in genres:
            folder = audio_root / system / genre
            if not folder.exists():
                print(f"  [SKIP] {folder} not found")
                continue

            files = sorted([f for f in folder.iterdir() if f.suffix.lower() in exts])
            print(f"\n[{system.upper()} / {genre}] — {len(files)} files")

            for f in files:
                # parse prompt_id from filename e.g. suno_A01.mp3 → A01
                stem = f.stem                         # e.g. "suno_A01"
                parts = stem.split("_")
                # find the part that looks like a prompt id (letter + 2 digits)
                pid = None
                for p in parts:
                    if len(p) == 3 and p[0].isalpha() and p[1:].isdigit():
                        pid = p.upper()
                        break

                if pid is None or pid not in PROMPT_META:
                    print(f"  [WARN] Cannot parse prompt ID from '{f.name}' — skipping")
                    continue

                g, axis, axis_type, axis_dir = PROMPT_META[pid]
                print(f"  {f.name:<35} → {pid} ({axis})", end=" ... ")

                try:
                    feats = extractor.extract(f)
                    feats["filename"]      = f.name
                    feats["system"]        = system
                    feats["genre"]         = genre
                    feats["prompt_id"]     = pid
                    feats["axis_varied"]   = axis
                    feats["axis_type"]     = axis_type
                    feats["axis_direction"]= axis_dir
                    rows.append(feats)
                    print("ok")
                except Exception as e:
                    print(f"ERROR: {e}")

    df = pd.DataFrame(rows)

    # put metadata cols first
    meta_cols = ["filename","system","genre","prompt_id",
                 "axis_varied","axis_type","axis_direction"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + feat_cols]

    df.to_csv(out_csv, index=False)
    print(f"\n{'='*70}")
    print(f"Saved {len(df)} tracks → {out_csv}")
    print(f"Systems: {df['system'].value_counts().to_dict()}")
    print(f"Genres:  {df['genre'].value_counts().to_dict()}")
    print(f"Features per track: {len(feat_cols)}")


if __name__ == "__main__":
    main()
