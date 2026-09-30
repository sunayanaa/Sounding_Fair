# =============================================================================
# Program      : 10_e2b_cfe_cfcc_probe.py
# Version      : 2.1
# GPU Required : NO
# Description  : Experiment 2b -- follow-up to E2, testing whether the
#                 user's own published Cent Filterbank Energy (CFE) and
#                 Cent Filterbank Cepstral Coefficient (CFCC) features
#                 (Sankaran, Krishnaraj & Murthy, CMMR 2015; Krishnaraj,
#                 Sankaran & Murthy, IIT Madras 2016) show a STRONGER
#                 cultural-specificity signal than generic 12-bin chroma,
#                 when given the REAL annotated tonic Saraga ships per
#                 composition -- not the argmax-chroma proxy E2 used.
#
#                 E2's chroma-based tonic-normalization gain for Carnatic
#                 (+0.068 F1) and Hindustani (+0.051 F1) was flagged as a
#                 likely FLOOR, not a ceiling, because it used (a) 12-bin
#                 chroma, too coarse to capture gamaka (the continuous
#                 pitch movement carrying much of a raga's identity), and
#                 (b) an inferred proxy tonic (argmax of a track's own
#                 chroma energy), not a verified frequency. This program
#                 removes both limitations for Carnatic/Hindustani: real
#                 per-composition tonic (Saraga's .ctonic.txt, confirmed
#                 present for both traditions) feeding CFE/CFCC's
#                 continuous cent-scale (1200*log2(f/tonic)), 80-filter
#                 representation.
#
#                 CORRECTION TO EARLIER CATALOGING: Saraga Carnatic DOES
#                 ship .ctonic.txt per composition (confirmed directly by
#                 the user against 'Chaitra Sairam - Anandaamrutakarshini
#                 .ctonic.txt' = 197.133121 Hz) -- earlier cataloging only
#                 confirmed this for Hindustani. Coverage across all 249
#                 Carnatic compositions is checked here empirically, not
#                 assumed from one example.
#
#                 SCOPE: 3-way classification only -- Carnatic, Hindustani,
#                 and a Dominant-pool sample (reused from Program 08's
#                 already-cached FMA crops on Google Drive, no restaging).
#                 CFE/CFCC need SOME tonic to normalize by; Dominant-pool
#                 tracks get one via librosa.pyin pitch-histogram peak -- a
#                 real Hz estimate (unlike E2's pitch-CLASS-only proxy), but
#                 still not a genuine tonic, since that concept doesn't
#                 apply to most Western pop/electronic content. This exists
#                 only to give CFE/CFCC a contrast class to be compared
#                 against, not to claim the Dominant class has meaningful
#                 tonic structure. Jingju and Arab-Andalusian are excluded
#                 from this specific comparison -- they aren't built around
#                 a continuous drone tonic the way Carnatic/Hindustani are,
#                 which is exactly why E2's proxy-tonic approach helped
#                 Carnatic/Hindustani but hurt Arab-Andalusian.
#
#                 All three feature variants (chroma-phi(t), CFE, CFCC)
#                 are computed on the SAME 4-second crop per track, for a
#                 fair comparison -- not reusing E2's old numbers, which
#                 were a different task (5-way) on a different track set.
#
# INPUT:
#                  Google Drive:
#                    datasets/Saraga/saraga1.5_carnatic.zip
#                    datasets/Saraga/saraga1.5_hindustani.zip
#                  Google Drive (from Program 08, no restaging):
#                    08_crops_fma.pt
#                  Local files (upload alongside this script in Colab):
#                    cfe.py, cfcc.py (user's own published implementations,
#                    used verbatim, not reconstructed)
#
# STEPS:
#                  Step 1  Mount Drive, ensure PROJECT_DIR exists
#                  Step 2  Stage Carnatic: mix-only mp3 + .ctonic.txt
#                          (extending Program 02's selective-extraction
#                          predicate to also keep .ctonic.txt)
#                  Step 3  Stage Hindustani: full extraction (already
#                          includes .ctonic.txt, confirmed 108/108 in
#                          Program 03) + .ctonic.txt
#                  Step 4  Copy 08_crops_fma.pt from Google Drive, sample 300
#                          tracks, estimate a proxy tonic per track via
#                          librosa.pyin
#                  Step 5  For each track (all 3 classes): extract a
#                          4-second crop, compute chroma-phi(t), CFE
#                          (mean+std pooled), CFCC (mean+std pooled)
#                  Step 6  5-fold stratified cross-validation per feature
#                          variant (not a single 80/20 split -- every
#                          track serves as a held-out test example
#                          exactly once, giving out-of-fold predictions
#                          for the full 197/107/300-track dataset instead
#                          of a 40/21/60 held-out slice)
#                          per feature variant; report accuracy, macro-F1,
#                          per-class precision/recall/F1; paired bootstrap
#                          significance test (chroma_phi vs CFE, chroma_phi
#                          vs CFCC, both traditions)
#                  Step 6c Narrow follow-up: plain 12-bin chroma alone
#                          (sliced from the cached chroma_phi vector, no
#                          audio reprocessing) vs CFE vs CFCC -- isolates
#                          whether Step 6b's result was about feature-
#                          family breadth (chroma_phi combines 4 families;
#                          CFE/CFCC are single-family) rather than cent-
#                          scale/real-tonic resolution itself
#                  Step 7  Save results JSON + a 3-way per-class F1
#                          comparison figure, copy both to Google Drive
#
# OUTPUT FILES : under PROJECT_DIR
#                  10_cfe_cfcc_probe_results.json
#                  fig_10_per_class_f1_comparison.png
#
# Hyperparameters : SAMPLE_RATE=22050, SEGMENT_SECONDS=4, N_MFCC=13,
#                    CFE/CFCC params = author defaults (frame_ms=100,
#                    hop_ms=10, n_filters=80, cent_min=-1200, cent_max=6000,
#                    n_ceps=13), DOMINANT_SAMPLE_SIZE=300,
#                    PYIN_FMIN=80.0, PYIN_FMAX=300.0, PYIN_DEFAULT_TONIC=150.0,
#                    N_FOLDS=5, RANDOM_SEED=42, LOGISTIC_MAX_ITER=2000.
# Dependencies : torch, librosa, scikit-learn, pandas, tqdm, matplotlib,
#                 scipy, zipfile (stdlib); cfe.py and
#                 cfcc.py must be present in the working directory.
#
# Change Log   :
#   v1.0  2026-09-23  Initial version
#   v1.1  2026-09-23  Added a paired bootstrap significance test (10,000
#                      iterations, Bonferroni-corrected for 2 comparisons)
#                      for Carnatic's F1 regression under CFE/CFCC vs
#                      chroma phi(t), following the first real run.
#                      Verifies the pairing assumption (same test split
#                      across variants) with an assert rather than just
#                      relying on "same y/seed/stratify should match"
#                      reasoning. Also caches the extracted feature
#                      matrices to Google Drive (10_features_cache.pt) so this and
#                      any further follow-up analysis doesn't require
#                      redoing the ~25-minute extraction each time.
#   v1.2  2026-09-23  Added the matching Hindustani bootstrap tests
#                      (chroma vs CFE, chroma vs CFCC) -- the Carnatic
#                      run showed no significant difference, so whether
#                      Hindustani's larger observed gain (+0.056 to
#                      +0.061 F1) is itself significant needed the same
#                      check, not just an assumption from the raw numbers
#                      looking more consistent. Bonferroni correction
#                      updated to n=4 across all four comparisons together
#                      (2 traditions x 2 feature types), not 2 separately
#                      per tradition -- more conservative for Carnatic
#                      (doesn't change its already-null conclusion) and
#                      the statistically correct family size now that
#                      Hindustani's tests are part of the same analysis.
#   v1.3  2026-09-23  Fixed a torch.load crash on the features cache and
#                      the FMA crops load: PyTorch 2.6+ defaults
#                      weights_only=True, which blocks unpickling numpy
#                      arrays (the features cache stores X_chroma/X_cfe/
#                      X_cfcc/y as plain numpy arrays, not torch tensors)
#                      as an unlisted "unsafe global." Both loads are our
#                      own self-created Google Drive cache, not untrusted input,
#                      so weights_only=False is added explicitly.
#   v2.0  2026-09-23  Replaced the single 80/20 train/test split with
#                      5-fold stratified cross-validation, following the
#                      first bootstrap run: neither Carnatic nor
#                      Hindustani reached significance, and Hindustani's
#                      CI was actually WIDER than Carnatic's despite a
#                      more consistent-looking raw gain -- a direct
#                      consequence of the single split only holding out
#                      21 Hindustani tracks. Out-of-fold predictions now
#                      cover all 197/107 tracks for the paired bootstrap,
#                      a more efficient use of the same small annotated-
#                      tonic corpora, not new data. NOTE: this invalidates
#                      any previously cached 10_features_cache.pt's
#                      DOWNSTREAM results (the cached FEATURES are still
#                      valid and reused as before -- only the evaluation
#                      method changed, not the inputs to it).
#   v2.1  2026-09-23  v2.0's 5-fold CV run showed chroma_phi significantly
#                      OUTPERFORMS CFCC for both traditions -- the
#                      opposite direction from what smaller-sample runs
#                      suggested. Added a narrow follow-up (Step 6c):
#                      plain 12-bin chroma alone vs CFE vs CFCC, sliced
#                      directly from the cached chroma_phi vector (no
#                      audio reprocessing), to isolate whether that
#                      result was really about feature-FAMILY breadth
#                      (chroma_phi = MFCC+chroma+contrast+Tonnetz, 4
#                      families; CFE/CFCC are single-family) rather than
#                      cent-scale/real-tonic resolution being unhelpful.
# Running time: 
# !pip install torch librosa scikit-learn pandas tqdm matplotlib scipy
# =============================================================================

!pip install torch librosa scikit-learn pandas tqdm matplotlib scipy

import os
import sys
import time
import json
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import librosa
import torch
import torchaudio
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

from google.colab import drive
drive.mount('/content/drive')

# cfe.py / cfcc.py must be uploaded alongside this script (Colab: use the
# file upload panel, or place them in /content/ before running).
sys.path.insert(0, "/content")
try:
    from cfe import extract_cfe
    from cfcc import extract_cfcc
except ImportError:
    print("[ERROR] cfe.py and/or cfcc.py not found in /content/. "
          "Upload both files (unmodified from the user's own implementation) "
          "before running this script.")
    sys.exit(1)

RUN_START_TIME = time.time()
print(f"[{datetime.now().strftime('%H:%M:%S')}] Program 10_e2b_cfe_cfcc_probe.py started.")

# --- Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/sounding-fair/"

SAMPLE_RATE = 22050
SEGMENT_SECONDS = 4
SEGMENT_LENGTH = SAMPLE_RATE * SEGMENT_SECONDS
N_MFCC = 13
DOMINANT_SAMPLE_SIZE = 300
PYIN_FMIN = 80.0
PYIN_FMAX = 300.0
PYIN_DEFAULT_TONIC = 150.0
N_FOLDS = 5
RANDOM_SEED = 42
LOGISTIC_MAX_ITER = 2000

LOCAL_ROOT = "/content/local_data_e2b"
os.makedirs(LOCAL_ROOT, exist_ok=True)

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


def copy_from_drive_with_progress(src, dst):
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
        print(f"[COPY] {dst} already present locally, skipping copy.")
        return
    total_size = os.path.getsize(src)
    chunk_size = 1024 * 1024 * 8
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst, tqdm(
        total=total_size, unit="B", unit_scale=True, unit_divisor=1024,
        desc=f"[COPY] {os.path.basename(src)}",
    ) as pbar:
        while True:
            chunk = fsrc.read(chunk_size)
            if not chunk:
                break
            fdst.write(chunk)
            pbar.update(len(chunk))


def is_junk(name):
    if "__MACOSX" in name or name.endswith("/"):
        return True
    basename = os.path.basename(name)
    return basename == ".DS_Store" or basename.startswith("._")


def extract_zip_selective(zip_path, extract_dir, member_predicate, expected_min_files=None):
    if expected_min_files is not None and os.path.isdir(extract_dir):
        existing = sum(len(files) for _, _, files in os.walk(extract_dir))
        if existing >= expected_min_files:
            print(f"[EXTRACT] {extract_dir} already has {existing} files, skipping.")
            return
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        wanted = [m for m in zf.namelist() if member_predicate(m)]
        for member in tqdm(wanted, desc=f"[EXTRACT] {os.path.basename(zip_path)}"):
            target = os.path.join(extract_dir, member)
            if os.path.exists(target):
                continue
            zf.extract(member, extract_dir)


def resolve_mix_path(comp_dir, comp_name):
    """Same dual-pattern check established in Program 02: some concerts
    ship the mix as '<name>.mp3.mp3', others as plain '<name>.mp3'."""
    double_ext = os.path.join(comp_dir, f"{comp_name}.mp3.mp3")
    if os.path.isfile(double_ext):
        return double_ext
    single_ext = os.path.join(comp_dir, f"{comp_name}.mp3")
    if os.path.isfile(single_ext):
        return single_ext
    return None


def read_tonic_file(comp_dir, comp_name):
    tonic_path = os.path.join(comp_dir, f"{comp_name}.ctonic.txt")
    if not os.path.isfile(tonic_path):
        return None
    try:
        with open(tonic_path, "r") as f:
            return float(f.read().strip())
    except (ValueError, OSError):
        return None


def stage_carnatic_with_tonic():
    root = os.path.join(LOCAL_ROOT, "carnatic")
    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, "saraga1.5_carnatic.zip")
    copy_from_drive_with_progress("/content/drive/MyDrive/datasets/Saraga/saraga1.5_carnatic.zip", zip_path)

    def is_wanted(name):
        if is_junk(name):
            return False
        basename = os.path.basename(name)
        if basename.endswith(".json") or basename.endswith(".ctonic.txt"):
            return True
        if basename.endswith(".mp3") and "multitrack" not in basename:
            return True
        return False

    extract_dir = os.path.join(root, "extracted")
    extract_zip_selective(zip_path, extract_dir, is_wanted, expected_min_files=249)

    saraga_root = os.path.join(extract_dir, "saraga1.5_carnatic")
    if not os.path.isdir(saraga_root):
        saraga_root = extract_dir

    pairs = []
    n_json = n_with_tonic = 0
    for dirpath, _, filenames in os.walk(saraga_root):
        for fn in filenames:
            if fn.endswith(".json"):
                n_json += 1
                comp_name = os.path.splitext(fn)[0]
                mix_path = resolve_mix_path(dirpath, comp_name)
                tonic = read_tonic_file(dirpath, comp_name)
                if mix_path is not None and tonic is not None:
                    pairs.append((mix_path, tonic))
                    n_with_tonic += 1
    print(f"[CARNATIC] {n_with_tonic}/{n_json} compositions have both audio and a real tonic "
          f"(coverage check -- earlier cataloging never confirmed this for Carnatic).")
    return pairs


def stage_hindustani_with_tonic():
    root = os.path.join(LOCAL_ROOT, "hindustani")
    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, "saraga1.5_hindustani.zip")
    copy_from_drive_with_progress("/content/drive/MyDrive/datasets/Saraga/saraga1.5_hindustani.zip", zip_path)
    extract_dir = os.path.join(root, "extracted")
    extract_zip_selective(zip_path, extract_dir, lambda n: not is_junk(n), expected_min_files=108)

    saraga_root = os.path.join(extract_dir, "saraga1.5_hindustani")
    if not os.path.isdir(saraga_root):
        saraga_root = extract_dir

    pairs = []
    n_json = n_with_tonic = 0
    for dirpath, _, filenames in os.walk(saraga_root):
        for fn in filenames:
            if fn.endswith(".json"):
                n_json += 1
                comp_name = os.path.splitext(fn)[0]
                mix_path = resolve_mix_path(dirpath, comp_name)
                tonic = read_tonic_file(dirpath, comp_name)
                if mix_path is not None and tonic is not None:
                    pairs.append((mix_path, tonic))
                    n_with_tonic += 1
    print(f"[HINDUSTANI] {n_with_tonic}/{n_json} recordings have both audio and a real tonic "
          f"(expected 108/108, per Program 03's earlier confirmation).")
    return pairs


def load_crop_from_path(path):
    """Same middle-third heuristic used throughout this project -- avoids
    likely silence/fade at the very start or end, bounded read regardless
    of the file's true length."""
    try:
        guess_sr = 44100
        orig_segment_guess = guess_sr * SEGMENT_SECONDS
        offset_guess = guess_sr * 20
        wf = None
        for frame_offset in (offset_guess, 0):
            try:
                w, s = torchaudio.load(path, frame_offset=frame_offset, num_frames=orig_segment_guess)
            except Exception:
                continue
            if w.shape[1] > 0:
                wf, sr = w, s
                break
        if wf is None:
            return None
        if wf.shape[0] > 1:
            wf = wf.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            wf = torchaudio.functional.resample(wf, sr, SAMPLE_RATE)
        if wf.shape[1] < SEGMENT_LENGTH:
            wf = torch.nn.functional.pad(wf, (0, SEGMENT_LENGTH - wf.shape[1]))
        elif wf.shape[1] > SEGMENT_LENGTH:
            wf = wf[:, :SEGMENT_LENGTH]
        return wf.squeeze(0).numpy()
    except Exception:
        return None


def estimate_proxy_tonic(waveform_np):
    """Real-Hz tonic estimate for the Dominant class via a pitch-histogram
    peak (librosa.pyin) -- an actual frequency, unlike E2's pitch-CLASS-
    only chroma-argmax proxy, but still not a genuine tonic in the
    musicological sense used for Carnatic/Hindustani."""
    try:
        f0, voiced_flag, _ = librosa.pyin(
            waveform_np, fmin=PYIN_FMIN, fmax=PYIN_FMAX, sr=SAMPLE_RATE)
        voiced_f0 = f0[voiced_flag]
        voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]
        if len(voiced_f0) == 0:
            return PYIN_DEFAULT_TONIC
        hist, edges = np.histogram(voiced_f0, bins=30)
        peak_idx = int(np.argmax(hist))
        return float((edges[peak_idx] + edges[peak_idx + 1]) / 2)
    except Exception:
        return PYIN_DEFAULT_TONIC


def compute_chroma_phi(waveform_np, sr):
    """Same construction as Program 09's compute_phi(standard variant) --
    MFCC + chroma + spectral contrast + Tonnetz, mean+std pooled."""
    mfcc = librosa.feature.mfcc(y=waveform_np, sr=sr, n_mfcc=N_MFCC)
    chroma = librosa.feature.chroma_stft(y=waveform_np, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=waveform_np, sr=sr)
    tonnetz = librosa.feature.tonnetz(chroma=chroma, sr=sr)

    def pool(feat):
        return np.concatenate([feat.mean(axis=1), feat.std(axis=1)])

    return np.concatenate([pool(mfcc), pool(chroma), pool(contrast), pool(tonnetz)])


def compute_cfe_pooled(waveform_np, sr, tonic):
    cfe = extract_cfe(waveform_np, sr, tonic=tonic)  # (n_frames, 80), author defaults
    return np.concatenate([cfe.mean(axis=0), cfe.std(axis=0)])  # 160-dim


def compute_cfcc_pooled(waveform_np, sr, tonic):
    cfcc = extract_cfcc(waveform_np, sr, tonic=tonic)  # (n_frames, 13), author defaults
    return np.concatenate([cfcc.mean(axis=0), cfcc.std(axis=0)])  # 26-dim


def evaluate_variant(X, y, variant_name, n_folds=N_FOLDS):
    """Stratified k-fold cross-validation instead of a single 80/20 split.
    Every track serves as a held-out test example exactly once across
    the n_folds folds (predictions are always made by a model that never
    saw that track during training), giving out-of-fold predictions for
    the FULL dataset -- 197 Carnatic / 107 Hindustani tracks instead of
    the 40/21 a single split held out. This directly addresses the wide,
    inconclusive confidence intervals the single-split version produced:
    more efficient use of the same small annotated-tonic corpora, not
    new data. Fold assignment depends only on (len(X), y, random_state,
    shuffle) -- not on X's values -- so calling this with the same y
    across chroma_phi/CFE/CFCC guarantees identical fold splits, keeping
    the downstream paired bootstrap valid."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    oof_preds = np.empty(len(y), dtype=object)
    fold_accuracies = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=LOGISTIC_MAX_ITER)
        clf.fit(X_train_s, y_train)
        fold_preds = clf.predict(X_test_s)
        oof_preds[test_idx] = fold_preds
        fold_accuracies.append(float(accuracy_score(y_test, fold_preds)))

    accuracy = accuracy_score(y, oof_preds)
    class_counts = pd.Series(y).value_counts()
    majority_baseline = class_counts.max() / len(y)
    report = classification_report(y, oof_preds, output_dict=True, zero_division=0)
    macro_f1 = report["macro avg"]["f1-score"]

    print(f"[PROBE-CV:{variant_name}] accuracy={accuracy:.4f} "
          f"(per-fold range {min(fold_accuracies):.3f}-{max(fold_accuracies):.3f})  "
          f"majority_baseline={majority_baseline:.4f}  macro_f1={macro_f1:.4f}")
    per_class = {
        cls: {"precision": report[cls]["precision"], "recall": report[cls]["recall"],
              "f1": report[cls]["f1-score"], "support": report[cls]["support"]}
        for cls in report if cls not in ("accuracy", "macro avg", "weighted avg")
    }
    for cls, m in per_class.items():
        print(f"    {cls:12s} precision={m['precision']:.3f}  recall={m['recall']:.3f}  "
              f"f1={m['f1']:.3f}  n={int(m['support'])}")

    metrics = {
        "accuracy": accuracy, "majority_baseline": majority_baseline, "macro_f1": macro_f1,
        "per_class": per_class, "fold_accuracies": fold_accuracies, "n_folds": n_folds,
        "class_counts": class_counts.to_dict(),
    }
    # Out-of-fold y/preds returned separately (not folded into metrics) so
    # the per-track arrays needed for the paired bootstrap below don't get
    # written into the saved results JSON. y itself is returned too
    # (identical to the input, but explicit) so the caller can verify all
    # three variants' out-of-fold arrays line up with the same underlying
    # track order before treating the comparison as paired.
    return metrics, y, oof_preds


def one_vs_rest_f1(y_true, y_pred, target_class):
    tp = np.sum((y_true == target_class) & (y_pred == target_class))
    fp = np.sum((y_true != target_class) & (y_pred == target_class))
    fn = np.sum((y_true == target_class) & (y_pred != target_class))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def paired_bootstrap_f1_diff(y_test, preds_a, preds_b, target_class,
                              n_iterations=10000, seed=RANDOM_SEED, n_comparisons=2):
    """Paired bootstrap over test-set tracks (with replacement), for one
    one-vs-rest class F1 comparison between two prediction sets that share
    the SAME underlying test set (same y, same split). Not artist-
    clustered -- the crop caches this whole project's later stages rely on
    carry no artist metadata, same limitation already flagged in Program
    09. n_comparisons sets the Bonferroni correction (default 2, for the
    chroma-vs-CFE and chroma-vs-CFCC comparisons run together below)."""
    rng = np.random.RandomState(seed)
    n = len(y_test)
    observed_a = one_vs_rest_f1(y_test, preds_a, target_class)
    observed_b = one_vs_rest_f1(y_test, preds_b, target_class)
    observed_diff = observed_a - observed_b

    diffs = np.empty(n_iterations)
    for i in range(n_iterations):
        idx = rng.randint(0, n, size=n)
        f1_a = one_vs_rest_f1(y_test[idx], preds_a[idx], target_class)
        f1_b = one_vs_rest_f1(y_test[idx], preds_b[idx], target_class)
        diffs[i] = f1_a - f1_b

    alpha = 0.05
    ci_low, ci_high = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    alpha_corrected = alpha / n_comparisons
    ci_low_c, ci_high_c = np.percentile(
        diffs, [100 * alpha_corrected / 2, 100 * (1 - alpha_corrected / 2)])

    return {
        "observed_f1_a": observed_a, "observed_f1_b": observed_b,
        "observed_diff": observed_diff,
        "bootstrap_mean_diff": float(diffs.mean()),
        "ci_95_low": float(ci_low), "ci_95_high": float(ci_high),
        "ci_95_bonferroni_low": float(ci_low_c), "ci_95_bonferroni_high": float(ci_high_c),
        "significant_uncorrected": bool(not (ci_low <= 0 <= ci_high)),
        "significant_bonferroni": bool(not (ci_low_c <= 0 <= ci_high_c)),
        "n_iterations": n_iterations, "n_comparisons_correction": n_comparisons,
    }


def main():
    ensure_project_dir()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 2: staging Carnatic (audio + real tonic)...")
    carnatic_pairs = stage_carnatic_with_tonic()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 3: staging Hindustani (audio + real tonic)...")
    hindustani_pairs = stage_hindustani_with_tonic()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 4: sampling Dominant-pool tracks from "
          f"Program 08's cached FMA crops...")
    fma_local = os.path.join(LOCAL_ROOT, "crops_fma.pt")
    dominant_waveforms = []
    if copy_from_drive("08_crops_fma.pt", fma_local, skip_if_exists=True):
        data = torch.load(fma_local, map_location="cpu", weights_only=False)
        crops = data["crops"]  # [N, n_crops, SEGMENT_LENGTH] int16
        n = min(DOMINANT_SAMPLE_SIZE, crops.shape[0])
        rng = np.random.RandomState(RANDOM_SEED)
        idx = rng.choice(crops.shape[0], size=n, replace=False)
        for i in idx:
            wf = (crops[i, 0].float() / 32767.0).numpy()
            dominant_waveforms.append(wf)
        del data, crops  # the ~4.2GB FMA tensor is done being used once the
                          # 300-track sample is extracted -- must be freed
                          # explicitly, or it stays resident through the rest
                          # of the script (os.remove only deletes the on-disk
                          # copy, not this in-memory reference)
        os.remove(fma_local)
    print(f"[DOMINANT] {len(dominant_waveforms)} tracks sampled.")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 5: extracting crops and computing "
          f"chroma-phi(t), CFE, CFCC for all tracks (or reusing a cached feature matrix)...")
    features_cache_local = os.path.join(LOCAL_ROOT, "10_features_cache.pt")
    features_cache_remote = "10_features_cache.pt"

    if copy_from_drive(features_cache_remote, features_cache_local, skip_if_exists=True):
        cached = torch.load(features_cache_local, map_location="cpu", weights_only=False)
        X_chroma, X_cfe, X_cfcc, y = cached["X_chroma"], cached["X_cfe"], cached["X_cfcc"], cached["y"]
        print(f"[CACHE] Reused cached features from Google Drive: {X_chroma.shape[0]} tracks.")
    else:
        X_chroma, X_cfe, X_cfcc, y = [], [], [], []

        for path, tonic in tqdm(carnatic_pairs, desc="[FEATURES] Carnatic"):
            wf = load_crop_from_path(path)
            if wf is None:
                continue
            X_chroma.append(compute_chroma_phi(wf, SAMPLE_RATE))
            X_cfe.append(compute_cfe_pooled(wf, SAMPLE_RATE, tonic))
            X_cfcc.append(compute_cfcc_pooled(wf, SAMPLE_RATE, tonic))
            y.append("Carnatic")

        for path, tonic in tqdm(hindustani_pairs, desc="[FEATURES] Hindustani"):
            wf = load_crop_from_path(path)
            if wf is None:
                continue
            X_chroma.append(compute_chroma_phi(wf, SAMPLE_RATE))
            X_cfe.append(compute_cfe_pooled(wf, SAMPLE_RATE, tonic))
            X_cfcc.append(compute_cfcc_pooled(wf, SAMPLE_RATE, tonic))
            y.append("Hindustani")

        for wf in tqdm(dominant_waveforms, desc="[FEATURES] Dominant (proxy tonic via pyin)"):
            tonic = estimate_proxy_tonic(wf)
            X_chroma.append(compute_chroma_phi(wf, SAMPLE_RATE))
            X_cfe.append(compute_cfe_pooled(wf, SAMPLE_RATE, tonic))
            X_cfcc.append(compute_cfcc_pooled(wf, SAMPLE_RATE, tonic))
            y.append("Dominant")

        X_chroma = np.stack(X_chroma)
        X_cfe = np.stack(X_cfe)
        X_cfcc = np.stack(X_cfcc)
        y = np.array(y)

        torch.save({"X_chroma": X_chroma, "X_cfe": X_cfe, "X_cfcc": X_cfcc, "y": y}, features_cache_local)
        copy_to_drive(features_cache_local, features_cache_remote)

    print(f"[FEATURES] Shapes -- chroma-phi(t): {X_chroma.shape}, CFE: {X_cfe.shape}, CFCC: {X_cfcc.shape}")
    print(f"[FEATURES] Class distribution: {pd.Series(y).value_counts().to_dict()}")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 6: fitting linear probes "
          f"({N_FOLDS}-fold stratified cross-validation)...")
    results = {}
    predictions = {}
    for variant_name, X in [("chroma_phi", X_chroma), ("cfe", X_cfe), ("cfcc", X_cfcc)]:
        metrics, y_oof, preds_oof = evaluate_variant(X, y, variant_name)
        results[variant_name] = metrics
        predictions[variant_name] = (y_oof, preds_oof)

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 6b: paired bootstrap significance test "
          f"for Carnatic and Hindustani (chroma_phi vs cfe, chroma_phi vs cfcc), using "
          f"out-of-fold predictions across all {len(y)} tracks...")
    y_oof_chroma, preds_chroma = predictions["chroma_phi"]
    y_oof_cfe, preds_cfe = predictions["cfe"]
    y_oof_cfcc, preds_cfcc = predictions["cfcc"]
    # Verify the pairing assumption empirically -- confirm the three
    # variants' out-of-fold arrays actually line up with the same
    # underlying track order before treating the comparison as paired,
    # rather than just relying on "same y/seed/shuffle should match"
    # reasoning.
    assert np.array_equal(y_oof_chroma, y_oof_cfe) and np.array_equal(y_oof_chroma, y_oof_cfcc), \
        "Out-of-fold label arrays differ across variants -- paired bootstrap requires an identical fold split."

    bootstrap_results = {
        "carnatic_chroma_vs_cfe": paired_bootstrap_f1_diff(
            y_oof_chroma, preds_chroma, preds_cfe, "Carnatic", n_comparisons=4),
        "carnatic_chroma_vs_cfcc": paired_bootstrap_f1_diff(
            y_oof_chroma, preds_chroma, preds_cfcc, "Carnatic", n_comparisons=4),
        "hindustani_chroma_vs_cfe": paired_bootstrap_f1_diff(
            y_oof_chroma, preds_chroma, preds_cfe, "Hindustani", n_comparisons=4),
        "hindustani_chroma_vs_cfcc": paired_bootstrap_f1_diff(
            y_oof_chroma, preds_chroma, preds_cfcc, "Hindustani", n_comparisons=4),
    }
    for name, r in bootstrap_results.items():
        sig = "SIGNIFICANT" if r["significant_bonferroni"] else "not significant"
        print(f"[BOOTSTRAP] {name}: observed_diff={r['observed_diff']:.4f}  "
              f"95% CI=({r['ci_95_low']:.4f}, {r['ci_95_high']:.4f})  "
              f"Bonferroni-corrected (n=4): {sig}")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 6c: narrow comparison -- plain chroma "
          f"alone (no MFCC/contrast/Tonnetz) vs CFE vs CFCC. Isolates whether the Step 6b result "
          f"was about feature-FAMILY breadth (chroma_phi combines 4 families, CFE/CFCC are single-"
          f"family) rather than cent-scale/real-tonic resolution itself. Sliced directly from the "
          f"cached chroma_phi features -- chroma is already a sub-block of that 76-dim vector, so "
          f"no audio reprocessing is needed.")
    mfcc_dim_pooled = 2 * N_MFCC
    chroma_dim_pooled = 2 * 12  # chroma_stft is always 12 pitch classes
    X_chroma_only = X_chroma[:, mfcc_dim_pooled:mfcc_dim_pooled + chroma_dim_pooled]
    print(f"[NARROW] Plain-chroma-only feature shape: {X_chroma_only.shape}")

    metrics_chroma_only, y_oof_narrow, preds_chroma_only = evaluate_variant(
        X_chroma_only, y, "chroma_only")
    assert np.array_equal(y_oof_narrow, y_oof_chroma), \
        "Narrow-chroma out-of-fold labels differ from the full chroma_phi run -- fold split mismatch."

    narrow_bootstrap_results = {
        "carnatic_chromaonly_vs_cfe": paired_bootstrap_f1_diff(
            y_oof_narrow, preds_chroma_only, preds_cfe, "Carnatic", n_comparisons=4),
        "carnatic_chromaonly_vs_cfcc": paired_bootstrap_f1_diff(
            y_oof_narrow, preds_chroma_only, preds_cfcc, "Carnatic", n_comparisons=4),
        "hindustani_chromaonly_vs_cfe": paired_bootstrap_f1_diff(
            y_oof_narrow, preds_chroma_only, preds_cfe, "Hindustani", n_comparisons=4),
        "hindustani_chromaonly_vs_cfcc": paired_bootstrap_f1_diff(
            y_oof_narrow, preds_chroma_only, preds_cfcc, "Hindustani", n_comparisons=4),
    }
    for name, r in narrow_bootstrap_results.items():
        sig = "SIGNIFICANT" if r["significant_bonferroni"] else "not significant"
        print(f"[BOOTSTRAP-NARROW] {name}: observed_diff={r['observed_diff']:.4f}  "
              f"95% CI=({r['ci_95_low']:.4f}, {r['ci_95_high']:.4f})  "
              f"Bonferroni-corrected (n=4): {sig}")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 7: saving results and figure...")
    local_results_path = os.path.join(LOCAL_ROOT, "10_cfe_cfcc_probe_results.json")
    elapsed_seconds = time.time() - RUN_START_TIME
    output = {
        "program": "10_e2b_cfe_cfcc_probe.py",
        "n_carnatic": len([1 for c in y if c == "Carnatic"]),
        "n_hindustani": len([1 for c in y if c == "Hindustani"]),
        "n_dominant": len([1 for c in y if c == "Dominant"]),
        "dominant_tonic_note": "proxy Hz tonic via librosa.pyin pitch-histogram peak, NOT a genuine tonic",
        "results": results,
        "carnatic_bootstrap_significance": bootstrap_results,
        "chroma_only_results": metrics_chroma_only,
        "narrow_comparison_note": "plain 12-bin chroma alone (24-dim pooled), sliced from the "
                                   "cached chroma_phi vector -- isolates feature-family breadth "
                                   "from cent-scale/real-tonic resolution as the explanation for "
                                   "the Step 6b result",
        "narrow_bootstrap_significance": narrow_bootstrap_results,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "elapsed_hms": time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds)),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(local_results_path, "w") as f:
        json.dump(output, f, indent=2)

    classes = sorted(results["chroma_phi"]["per_class"].keys())
    variants = ["chroma_phi", "cfe", "cfcc"]
    variant_labels = ["Chroma phi(t)", "CFE", "CFCC"]
    x = np.arange(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (variant, label) in enumerate(zip(variants, variant_labels)):
        f1_vals = [results[variant]["per_class"][c]["f1"] for c in classes]
        ax.bar(x + (i - 1) * width, f1_vals, width, label=label)
    ax.set_ylabel("F1 score")
    ax.set_title("E2b: Per-class F1 -- chroma phi(t) vs CFE vs CFCC (real tonic)")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.0)
    ax.legend()
    fig.tight_layout()
    local_fig_path = os.path.join(LOCAL_ROOT, "fig_10_per_class_f1_comparison.png")
    fig.savefig(local_fig_path, dpi=300)
    plt.close(fig)

    copy_to_drive(local_results_path, "10_cfe_cfcc_probe_results.json")
    copy_to_drive(local_fig_path, "fig_10_per_class_f1_comparison.png")

    return output


if __name__ == "__main__":
    summary = main()
    elapsed = time.time() - RUN_START_TIME
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Program 10_e2b_cfe_cfcc_probe.py finished.")
    print(f"[TOTAL RUNNING TIME] {time.strftime('%H:%M:%S', time.gmtime(elapsed))} ({elapsed:.1f}s)")