# =============================================================================
# Program      : 09_e2_feature_bias_probe.py
# Version      : 1.0
# GPU Required : NO
# Description  : Experiment 2 -- Feature-Space Bias Probe. Tests whether
#                 musical tradition is linearly decodable directly from
#                 phi(t) -- the handcrafted audio feature representation
#                 -- BEFORE any learned model is involved, and whether a
#                 tonic-normalized chroma variant changes the answer
#                 versus standard (untransposed) chroma.
#
#                 Reuses the 7 per-source crop caches already produced and
#                 cached to Google Drive by Program 08 (08_crops_<source>.pt) --
#                 no Drive re-staging, no GPU. Takes the FIRST
#                 of each track's 3 stored crops (this probe doesn't need
#                 multiple views the way E1's contrastive training did).
#
#                 phi(t) here = MFCC + chroma + spectral contrast +
#                 Tonnetz (mean+std pooled), computed via librosa. VGGish
#                 is deliberately OMITTED from this pass -- adding another
#                 external pretrained-model dependency carries real risk
#                 given how much trouble torchaudio's own API surface
#                 caused in Program 08. This is a stated limitation, not a
#                 silent omission: results should be read as "decodable
#                 from handcrafted acoustic features," not the full
#                 originally-planned phi(t).
#
#                 Tonic estimation is ALGORITHMIC (argmax of the per-track
#                 chroma energy summed over time), not the real annotated
#                 tonic Saraga ships for Hindustani (.ctonic.txt) -- that
#                 file was never staged during Program 08's audio-only
#                 extraction. Applying the same simple estimator uniformly
#                 across all 5 classes (4 traditions + dominant pool) is a
#                 deliberate choice for comparability, not an assumption
#                 that "tonic" is a native concept for the dominant pool.
#
#                 CLASS LABELS (5-way, not the original 4 traditions --
#                 the dominant pool is included as its own class since the
#                 audit's whole point is contrasting traditions against
#                 it): fma+jamendo -> Dominant, carnatic -> Carnatic,
#                 hindustani -> Hindustani, jingju_base+jacrc -> Jingju,
#                 arab_andalusian -> ArabAndalusian.
#
#                 SPLIT CAVEAT: track-level random split, stratified by
#                 class, NOT artist-disjoint -- the crop caches carry no
#                 artist metadata (stripped for simplicity in Program 08).
#                 Some accuracy inflation from artist leakage is possible,
#                 especially for the small-count traditions. Flagged here
#                 rather than silently assumed away.
#
# INPUT:
#                  Google Drive (from Program 08):
#                    08_crops_fma.pt, 08_crops_carnatic.pt,
#                    08_crops_hindustani.pt, 08_crops_jingju_base.pt,
#                    08_crops_jacrc.pt, 08_crops_arab_andalusian.pt,
#                    08_crops_jamendo.pt
#
# STEPS:
#                  Step 1  Ensure PROJECT_DIR exists
#                  Step 2  Copy the 7 per-source crop caches
#                  Step 3  Take crop index 0 per track; compute phi(t) in
#                          both variants (standard chroma, tonic-
#                          normalized chroma) via librosa
#                  Step 4  Stratified train/test split (80/20, seed=42)
#                  Step 5  Fit a multinomial logistic-regression probe per
#                          variant; report accuracy, majority-class
#                          baseline, per-class precision/recall/F1,
#                          macro-F1
#                  Step 6  Save results JSON + a per-class F1 comparison
#                          figure (standard vs tonic-normalized), copy
#                          both to Google Drive
#
# OUTPUT FILES : under PROJECT_DIR
#                  09_feature_bias_probe_results.json
#                      Per-variant: accuracy, majority_baseline,
#                      per-class precision/recall/F1, macro_f1,
#                      class_counts (train+test)
#                  fig_09_per_class_f1_comparison.png
#                      Grouped bar chart, per-tradition F1, standard vs
#                      tonic-normalized chroma variants
#
# Hyperparameters : SAMPLE_RATE=22050 (matches Program 08's crops),
#                    N_MFCC=13, TEST_SIZE=0.2, RANDOM_SEED=42,
#                    LOGISTIC_MAX_ITER=2000.
# Dependencies : torch, librosa, scikit-learn, pandas, tqdm, matplotlib
#
# Change Log   :
#   v1.0  2026-09-17  Initial version
# Running time:
# !pip install torch librosa scikit-learn pandas tqdm matplotlib
# =============================================================================

!pip install torch librosa scikit-learn pandas tqdm matplotlib

import os
import time
import json
from datetime import datetime

import numpy as np
import pandas as pd
import librosa
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

from google.colab import drive
drive.mount('/content/drive')

RUN_START_TIME = time.time()
print(f"[{datetime.now().strftime('%H:%M:%S')}] Program 09_e2_feature_bias_probe.py started.")

# --- Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/sounding-fair/"

SAMPLE_RATE = 22050
N_MFCC = 13
TEST_SIZE = 0.2
RANDOM_SEED = 42
LOGISTIC_MAX_ITER = 2000

LOCAL_ROOT = "/content/local_data_e2"
os.makedirs(LOCAL_ROOT, exist_ok=True)

# Source -> class label mapping (5-way: 4 traditions + dominant pool)
SOURCE_TO_CLASS = {
    "fma": "Dominant",
    "jamendo": "Dominant",
    "carnatic": "Carnatic",
    "hindustani": "Hindustani",
    "jingju_base": "Jingju",
    "jacrc": "Jingju",
    "arab_andalusian": "ArabAndalusian",
}
SOURCES = list(SOURCE_TO_CLASS.keys())

# --- Begin Google Drive Helper Functions ---
def ensure_project_dir():
    """Create PROJECT_DIR on Google Drive if it doesn't already exist."""
    os.makedirs(PROJECT_DIR, exist_ok=True)
    print(f"[Drive] Project directory ready: {PROJECT_DIR}")


def copy_to_drive(local_filepath, remote_filename):
    """Copy a local file to the Google Drive project directory."""
    dest = os.path.join(PROJECT_DIR, remote_filename)
    total_size = os.path.getsize(local_filepath)
    chunk_size = 1024 * 1024 * 8  # 8 MB
    with open(local_filepath, "rb") as fsrc, open(dest, "wb") as fdst, tqdm(
        total=total_size, unit="B", unit_scale=True, unit_divisor=1024,
        desc=f"[Drive] Copying {remote_filename}",
    ) as pbar:
        while True:
            chunk = fsrc.read(chunk_size)
            if not chunk:
                break
            fdst.write(chunk)
            pbar.update(len(chunk))


def copy_from_drive(remote_filename, local_filepath, verbose=True, skip_if_exists=True):
    """Copy a file from the Google Drive project directory to local disk.
    Skips the copy entirely if a local copy already exists AND its
    size matches the source's (a size mismatch means re-copy)."""
    src = os.path.join(PROJECT_DIR, remote_filename)

    if skip_if_exists and os.path.exists(local_filepath):
        if not os.path.exists(src):
            src_size = None
        else:
            src_size = os.path.getsize(src)
        local_size = os.path.getsize(local_filepath)
        if src_size is None or local_size == src_size:
            print(f"[Drive] {local_filepath} already exists locally "
                  f"({local_size / 1e6:.1f} MB), skipping copy.")
            return True
        print(f"[Drive] {local_filepath} exists locally but size differs from "
              f"source ({local_size} vs {src_size} bytes) -- re-copying.")

    if not os.path.exists(src):
        print(f"[Drive] {remote_filename} not found in {PROJECT_DIR}")
        return False

    total_size = os.path.getsize(src)
    chunk_size = 1024 * 1024 * 8  # 8 MB
    with open(src, "rb") as fsrc, open(local_filepath, "wb") as fdst, tqdm(
        total=total_size, unit="B", unit_scale=True, unit_divisor=1024,
        desc=f"[Drive] {remote_filename}", disable=not verbose,
    ) as pbar:
        while True:
            chunk = fsrc.read(chunk_size)
            if not chunk:
                break
            fdst.write(chunk)
            pbar.update(len(chunk))
    return True
# --- End Google Drive Helper Functions ---


def compute_phi(waveform_np, sr, tonic_normalize=False):
    """MFCC + chroma + spectral contrast + Tonnetz, mean+std pooled.
    If tonic_normalize, chroma (and the Tonnetz derived from it) is
    circularly shifted so the track's own most-energetic pitch class
    becomes bin 0 -- an algorithmic tonic estimate, not a real annotation."""
    mfcc = librosa.feature.mfcc(y=waveform_np, sr=sr, n_mfcc=N_MFCC)
    chroma = librosa.feature.chroma_stft(y=waveform_np, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=waveform_np, sr=sr)

    if tonic_normalize:
        tonic_bin = int(np.argmax(chroma.sum(axis=1)))
        chroma = np.roll(chroma, -tonic_bin, axis=0)

    tonnetz = librosa.feature.tonnetz(chroma=chroma, sr=sr)

    def pool(feat):
        return np.concatenate([feat.mean(axis=1), feat.std(axis=1)])

    return np.concatenate([pool(mfcc), pool(chroma), pool(contrast), pool(tonnetz)])


def evaluate_variant(X, y, variant_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=LOGISTIC_MAX_ITER, multi_class="multinomial")
    clf.fit(X_train_s, y_train)
    preds = clf.predict(X_test_s)

    accuracy = accuracy_score(y_test, preds)
    class_counts_test = pd.Series(y_test).value_counts()
    majority_baseline = class_counts_test.max() / len(y_test)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    macro_f1 = report["macro avg"]["f1-score"]

    print(f"[PROBE:{variant_name}] accuracy={accuracy:.4f}  majority_baseline={majority_baseline:.4f}  "
          f"macro_f1={macro_f1:.4f}")
    per_class = {
        cls: {"precision": report[cls]["precision"], "recall": report[cls]["recall"],
              "f1": report[cls]["f1-score"], "support": report[cls]["support"]}
        for cls in report if cls not in ("accuracy", "macro avg", "weighted avg")
    }
    for cls, m in per_class.items():
        print(f"    {cls:16s} precision={m['precision']:.3f}  recall={m['recall']:.3f}  "
              f"f1={m['f1']:.3f}  n={int(m['support'])}")

    return {
        "accuracy": accuracy,
        "majority_baseline": majority_baseline,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "class_counts_train": pd.Series(y_train).value_counts().to_dict(),
        "class_counts_test": class_counts_test.to_dict(),
    }


def main():
    ensure_project_dir()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 2: copying per-source crop caches...")
    per_track_waveforms = []
    labels = []
    for name in SOURCES:
        local_path = os.path.join(LOCAL_ROOT, f"crops_{name}.pt")
        if not copy_from_drive(f"08_crops_{name}.pt", local_path, skip_if_exists=True):
            print(f"[WARNING] Could not fetch 08_crops_{name}.pt -- skipping this source entirely.")
            continue
        data = torch.load(local_path, map_location="cpu")
        crops = data["crops"]  # [N, n_crops, SEGMENT_LENGTH] int16
        cls = SOURCE_TO_CLASS[name]
        for i in range(crops.shape[0]):
            wf = (crops[i, 0].float() / 32767.0).numpy()  # crop index 0 only
            per_track_waveforms.append(wf)
            labels.append(cls)
        os.remove(local_path)  # small enough overall not to need per-source disk cleanup elsewhere,
                                # but no reason to keep the raw crop file around post-extraction
        print(f"[SOURCE] {name} ({cls}): {crops.shape[0]} tracks added.")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 3: computing phi(t) "
          f"(standard + tonic-normalized) for {len(per_track_waveforms)} tracks...")
    X_standard, X_tonic = [], []
    for wf in tqdm(per_track_waveforms, desc="[PHI] extracting features"):
        X_standard.append(compute_phi(wf, SAMPLE_RATE, tonic_normalize=False))
        X_tonic.append(compute_phi(wf, SAMPLE_RATE, tonic_normalize=True))
    X_standard = np.stack(X_standard)
    X_tonic = np.stack(X_tonic)
    y = np.array(labels)
    print(f"[PHI] Feature matrix shape: {X_standard.shape} (standard), {X_tonic.shape} (tonic-normalized)")
    print(f"[PHI] Class distribution: {pd.Series(y).value_counts().to_dict()}")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 4-5: fitting linear probes...")
    results = {
        "standard": evaluate_variant(X_standard, y, "standard"),
        "tonic_normalized": evaluate_variant(X_tonic, y, "tonic_normalized"),
    }

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 6: saving results and figure...")
    local_results_path = os.path.join(LOCAL_ROOT, "09_feature_bias_probe_results.json")
    elapsed_seconds = time.time() - RUN_START_TIME
    output = {
        "program": "09_e2_feature_bias_probe.py",
        "phi_components": "MFCC + chroma + spectral_contrast + Tonnetz (VGGish omitted, see header)",
        "n_tracks_total": len(per_track_waveforms),
        "class_labels": sorted(set(labels)),
        "split_caveat": "track-level random split, stratified by class, NOT artist-disjoint",
        "results": results,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "elapsed_hms": time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds)),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(local_results_path, "w") as f:
        json.dump(output, f, indent=2)

    # --- Figure: per-class F1, standard vs tonic-normalized ---
    classes = sorted(results["standard"]["per_class"].keys())
    std_f1 = [results["standard"]["per_class"][c]["f1"] for c in classes]
    tonic_f1 = [results["tonic_normalized"]["per_class"][c]["f1"] for c in classes]

    x = np.arange(len(classes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, std_f1, width, label="Standard chroma")
    ax.bar(x + width / 2, tonic_f1, width, label="Tonic-normalized chroma")
    ax.set_ylabel("F1 score")
    ax.set_title("E2: Per-class F1, standard vs tonic-normalized phi(t)")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=20)
    ax.set_ylim(0, 1.0)
    ax.legend()
    fig.tight_layout()
    local_fig_path = os.path.join(LOCAL_ROOT, "fig_09_per_class_f1_comparison.png")
    fig.savefig(local_fig_path, dpi=300)
    plt.close(fig)

    copy_to_drive(local_results_path, "09_feature_bias_probe_results.json")
    copy_to_drive(local_fig_path, "fig_09_per_class_f1_comparison.png")

    return output


if __name__ == "__main__":
    summary = main()
    elapsed = time.time() - RUN_START_TIME
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Program 09_e2_feature_bias_probe.py finished.")
    print(f"[TOTAL RUNNING TIME] {time.strftime('%H:%M:%S', time.gmtime(elapsed))} ({elapsed:.1f}s)")