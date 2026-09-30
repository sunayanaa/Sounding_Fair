# =============================================================================
# Program      : 11_e3_subband_leakage_probe.py
# Version      : 1.1
# GPU Required : NO (uses GPU automatically if available, for speed only --
#                 this is inference-only, no backprop, so CPU is workable)
# Description  : Experiment 3 -- Sub-band Leakage Probing. Locates which
#                 Mel sub-bands in the TRAINED E1 backbone's internal
#                 representation still carry production-confound
#                 information (loudness), after augmentation-invariance
#                 training that was specifically meant to remove it.
#
#                 The backbone's FINAL embedding is a single pooled
#                 256-dim vector with no frequency structure left --
#                 "which sub-band leaks information" can't be asked of
#                 it. Instead, this probes the encoder's PRE-POOLING
#                 intermediate activation (model.conv[:-1], i.e.
#                 everything except the final AdaptiveAvgPool2d): at that
#                 point the representation still has 8 frequency
#                 positions (64 Mel bins / 2^3, from the three
#                 MaxPool2d(2) layers in the conv stack), which are split
#                 into low/mid/high thirds, each separately pooled over
#                 time (and over the remaining frequency positions within
#                 that third) into a 256-dim per-sub-band vector.
#
#                 Confound label: loudness (RMS dB of the raw crop
#                 waveform), MEDIAN-SPLIT into Loud/Quiet. This split is
#                 deliberately artificial -- used only as a balanced
#                 diagnostic classification target for the probe, not
#                 claimed to reflect a real bimodal loudness distribution
#                 (the exact mistake flagged earlier in this project's own
#                 cataloging of TDFS's rebalance_pitch.py, which forced a
#                 50/50 split and then treated it as if it meant something
#                 about the real data). E1's own augmentation pipeline
#                 randomly perturbs gain +/-6dB specifically to teach
#                 loudness invariance -- this program checks whether it
#                 worked, and where residual leakage (if any) is
#                 concentrated in frequency.
#
#                 Same evaluation methodology as Program 10 (5-fold
#                 stratified CV, paired bootstrap significance, 10,000
#                 iterations, Bonferroni-corrected) for consistency and
#                 because the earlier single-split version of that
#                 methodology was shown to give misleading conclusions at
#                 low n -- not repeating that mistake here.
#
#                 Reuses E1's 7 per-source crop caches AND the trained
#                 08_backbone_final.pt directly from Google Drive -- no Drive
#                 re-staging. Learning from Program 08's RAM-
#                 doubling saga: per-source tensors are kept as a LIST,
#                 never concatenated into one tensor, even though this
#                 program does no training and has more memory headroom.
#
# INPUT:
#                  Google Drive (from Program 08, no restaging):
#                    08_crops_fma.pt, 08_crops_carnatic.pt,
#                    08_crops_hindustani.pt, 08_crops_jingju_base.pt,
#                    08_crops_jacrc.pt, 08_crops_arab_andalusian.pt,
#                    08_crops_jamendo.pt, 08_backbone_final.pt
#
# STEPS:
#                  Step 1  Ensure PROJECT_DIR exists
#                  Step 2  Copy all 7 per-source crop caches + the
#                          trained backbone weights
#                  Step 3  Build the AudioEncoder (same architecture as
#                          Program 08), load trained weights, eval mode
#                  Step 4  For each track (crop index 0 only): compute
#                          mel-spectrogram, forward through model.conv[:-1],
#                          split into low/mid/high frequency thirds, pool
#                          each into a 256-dim sub-band vector; compute
#                          the loudness confound label from the raw
#                          waveform
#                  Step 5  5-fold stratified CV per sub-band (low/mid/high),
#                          TRAINED model, logistic-regression probe for
#                          Loud/Quiet
#                  Step 5b Same, for an UNTRAINED baseline encoder (same
#                          architecture, random weights) -- isolates
#                          whether the low>mid>high pattern from the
#                          first run is a training effect or just an
#                          architectural/spectral artifact (RMS loudness
#                          is dominated by low-frequency energy almost by
#                          definition)
#                  Step 5c Paired bootstrap significance: trained
#                          pairwise (low/mid/high), untrained pairwise,
#                          and trained-vs-untrained per band -- 9
#                          comparisons total, Bonferroni-corrected together
#                  Step 6  Save results JSON + a per-sub-band accuracy
#                          comparison figure, copy both to Google Drive
#
# OUTPUT FILES : under PROJECT_DIR
#                  11_subband_leakage_results.json
#                  fig_11_subband_leakage_comparison.png
#
# Hyperparameters : SAMPLE_RATE=22050, SEGMENT_SECONDS=4, N_MELS=64
#                    (must match Program 08 exactly, or the loaded
#                    weights won't be architecturally compatible),
#                    EMBEDDING_DIM=256, PROJECTION_DIM=128, N_FOLDS=5,
#                    RANDOM_SEED=42, LOGISTIC_MAX_ITER=2000,
#                    N_BOOTSTRAP_ITERATIONS=10000.
# Dependencies : torch, torchaudio, scikit-learn, pandas, tqdm,
#                 matplotlib
#
# Change Log   :
#   v1.0  2026-09-23  Initial version
#   v1.1  2026-09-23  The first run found a clean, statistically
#                      significant low>mid>high loudness-leakage
#                      ordering, but this is ambiguous on its own: RMS
#                      loudness is dominated by low-frequency energy
#                      almost by construction, so an untrained network
#                      could show the same pattern for reasons having
#                      nothing to do with training. Added an untrained
#                      baseline encoder (same architecture, random
#                      weights) as a control, run through the identical
#                      pipeline, plus trained-vs-untrained bootstrap
#                      comparisons per band (9 comparisons total now,
#                      Bonferroni-corrected together). Also added
#                      embedding-level caching (11_embeddings_cache.pt)
#                      given the ~34-minute extraction cost.
# Running time: 34:28
# !pip install torch torchaudio scikit-learn pandas tqdm matplotlib
# =============================================================================

!pip install torch torchaudio scikit-learn pandas tqdm matplotlib

import os
import time
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

from google.colab import drive
drive.mount('/content/drive')

RUN_START_TIME = time.time()
print(f"[{datetime.now().strftime('%H:%M:%S')}] Program 11_e3_subband_leakage_probe.py started.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] Using {DEVICE} (GPU not required for this program; used automatically if present).")

# --- Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/sounding-fair/"

SAMPLE_RATE = 22050
SEGMENT_SECONDS = 4
SEGMENT_LENGTH = SAMPLE_RATE * SEGMENT_SECONDS
N_MELS = 64
EMBEDDING_DIM = 256
PROJECTION_DIM = 128
N_FOLDS = 5
RANDOM_SEED = 42
LOGISTIC_MAX_ITER = 2000
N_BOOTSTRAP_ITERATIONS = 10000

SOURCES = ["fma", "carnatic", "hindustani", "jingju_base", "jacrc", "arab_andalusian", "jamendo"]

LOCAL_ROOT = "/content/local_data_e3"
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


# --- Same AudioEncoder architecture as Program 08 -- must match exactly
# for the trained state_dict to load correctly. ---
class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embedding_head = nn.Linear(256, EMBEDDING_DIM)
        self.projection_head = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM), nn.ReLU(),
            nn.Linear(EMBEDDING_DIM, PROJECTION_DIM),
        )

    def forward(self, x):
        h = self.conv(x).flatten(1)
        embedding = self.embedding_head(h)
        projection = self.projection_head(embedding)
        return embedding, projection

    def forward_prepool(self, x):
        """Everything except the final AdaptiveAvgPool2d(1) -- i.e.
        self.conv[:-1]. Returns [B, 256, F', T'] with F'=8 given three
        MaxPool2d(2) layers acting on N_MELS=64 (64 -> 32 -> 16 -> 8)."""
        return self.conv[:-1](x)


def load_trained_encoder(checkpoint_path):
    model = AudioEncoder().to(DEVICE)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model


_MEL_TRANSFORM = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=N_MELS).to(DEVICE)
_AMPLITUDE_TO_DB = torchaudio.transforms.AmplitudeToDB().to(DEVICE)


def crop_to_mel(waveform_int16):
    """int16 crop -> normalized float waveform -> log-mel spectrogram,
    identical transform to what the encoder was trained on. Transforms
    are constructed once at module level, not per call -- this function
    runs ~17,000 times across the full pool."""
    waveform = (waveform_int16.float() / 32767.0).unsqueeze(0).unsqueeze(0).to(DEVICE)
    mel = _AMPLITUDE_TO_DB(_MEL_TRANSFORM(waveform))
    return mel  # [1, 1, N_MELS, T]


def compute_loudness_db(waveform_int16):
    waveform = waveform_int16.float() / 32767.0
    rms = torch.sqrt(torch.mean(waveform ** 2) + 1e-10)
    return float(20 * torch.log10(rms + 1e-10))


def split_and_pool(activ):
    """activ: [1, 256, F'=8, T'] pre-pooling activation from
    model.forward_prepool(). Splits frequency into low/mid/high thirds
    and pools each into a 256-dim vector. Model-agnostic -- used for
    both the trained and untrained encoders."""
    activ = activ.squeeze(0)  # [256, 8, T']
    f_total = activ.shape[1]
    third = f_total // 3
    low = activ[:, 0:third, :]
    mid = activ[:, third:2 * third, :]
    high = activ[:, 2 * third:, :]

    def pool(x):
        return x.mean(dim=(1, 2)).cpu().numpy()  # mean over freq-in-band and time -> [256]

    return pool(low), pool(mid), pool(high)


def create_untrained_encoder(seed=RANDOM_SEED):
    """Freshly initialized AudioEncoder, same architecture, random
    weights (fixed seed for reproducibility) -- a control condition
    isolating whether the low>mid>high loudness-leakage pattern seen in
    the first run is a property of TRAINING (real invariance failure)
    or just an architectural/spectral artifact: RMS loudness is, almost
    by definition, dominated by low-frequency energy, so an untrained
    random network processing the same mel-spectrograms could plausibly
    show the same ordering regardless of what (if anything) was learned."""
    torch.manual_seed(seed)
    model = AudioEncoder().to(DEVICE)
    model.eval()
    return model


def evaluate_subband(X, y, band_name, n_folds=N_FOLDS):
    """Same 5-fold stratified CV methodology as Program 10."""
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
    report = classification_report(y, oof_preds, output_dict=True, zero_division=0)
    macro_f1 = report["macro avg"]["f1-score"]

    print(f"[PROBE-CV:{band_name}] accuracy={accuracy:.4f} "
          f"(per-fold range {min(fold_accuracies):.3f}-{max(fold_accuracies):.3f})  "
          f"macro_f1={macro_f1:.4f}")

    metrics = {
        "accuracy": accuracy, "macro_f1": macro_f1,
        "fold_accuracies": fold_accuracies, "n_folds": n_folds,
    }
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


def paired_bootstrap_acc_diff(y_test, preds_a, preds_b, n_iterations=N_BOOTSTRAP_ITERATIONS,
                               seed=RANDOM_SEED, n_comparisons=3):
    """Paired bootstrap on overall accuracy (binary Loud/Quiet task, so
    accuracy is a clean, directly-interpretable statistic here -- no
    per-class breakdown needed the way the multi-class E2b comparisons
    required)."""
    rng = np.random.RandomState(seed)
    n = len(y_test)
    observed_a = accuracy_score(y_test, preds_a)
    observed_b = accuracy_score(y_test, preds_b)
    observed_diff = observed_a - observed_b

    diffs = np.empty(n_iterations)
    for i in range(n_iterations):
        idx = rng.randint(0, n, size=n)
        acc_a = accuracy_score(y_test[idx], preds_a[idx])
        acc_b = accuracy_score(y_test[idx], preds_b[idx])
        diffs[i] = acc_a - acc_b

    alpha = 0.05
    ci_low, ci_high = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    alpha_c = alpha / n_comparisons
    ci_low_c, ci_high_c = np.percentile(diffs, [100 * alpha_c / 2, 100 * (1 - alpha_c / 2)])

    return {
        "observed_acc_a": observed_a, "observed_acc_b": observed_b,
        "observed_diff": observed_diff, "bootstrap_mean_diff": float(diffs.mean()),
        "ci_95_low": float(ci_low), "ci_95_high": float(ci_high),
        "ci_95_bonferroni_low": float(ci_low_c), "ci_95_bonferroni_high": float(ci_high_c),
        "significant_uncorrected": bool(not (ci_low <= 0 <= ci_high)),
        "significant_bonferroni": bool(not (ci_low_c <= 0 <= ci_high_c)),
        "n_iterations": n_iterations, "n_comparisons_correction": n_comparisons,
    }


def main():
    ensure_project_dir()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 2: copying trained backbone...")
    checkpoint_local = os.path.join(LOCAL_ROOT, "08_backbone_final.pt")
    if not copy_from_drive("08_backbone_final.pt", checkpoint_local, skip_if_exists=True):
        raise RuntimeError("Could not fetch 08_backbone_final.pt -- run Program 08 first.")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 3: loading trained encoder + "
          f"creating untrained baseline encoder...")
    model = load_trained_encoder(checkpoint_local)
    untrained_model = create_untrained_encoder()
    print(f"[MODEL] Trained AudioEncoder loaded, untrained baseline created "
          f"(same architecture, seed={RANDOM_SEED}), both eval mode, device={DEVICE}.")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 4: extracting sub-band embeddings "
          f"(trained AND untrained, same mel-spectrogram shared across both) + loudness "
          f"labels (crop index 0 per track, all 7 sources), or reusing a cached embedding matrix...")
    embeddings_cache_local = os.path.join(LOCAL_ROOT, "11_embeddings_cache.pt")
    embeddings_cache_remote = "11_embeddings_cache.pt"

    if copy_from_drive(embeddings_cache_remote, embeddings_cache_local, skip_if_exists=True):
        cached = torch.load(embeddings_cache_local, map_location="cpu", weights_only=False)
        X_low, X_mid, X_high = cached["X_low"], cached["X_mid"], cached["X_high"]
        X_low_u, X_mid_u, X_high_u = cached["X_low_u"], cached["X_mid_u"], cached["X_high_u"]
        loudness_values = cached["loudness_values"]
        print(f"[CACHE] Reused cached embeddings from Google Drive: {len(loudness_values)} tracks.")
    else:
        X_low, X_mid, X_high = [], [], []
        X_low_u, X_mid_u, X_high_u = [], [], []
        loudness_values = []
        for name in SOURCES:
            crops_local = os.path.join(LOCAL_ROOT, f"crops_{name}.pt")
            if not copy_from_drive(f"08_crops_{name}.pt", crops_local, skip_if_exists=True):
                print(f"[WARNING] Could not fetch 08_crops_{name}.pt -- skipping this source entirely.")
                continue
            data = torch.load(crops_local, map_location="cpu", weights_only=False)
            crops = data["crops"]  # [N, n_crops, SEGMENT_LENGTH] int16
            for i in tqdm(range(crops.shape[0]), desc=f"[SUBBAND] {name}"):
                waveform = crops[i, 0]
                mel = crop_to_mel(waveform)  # computed once, shared across both models
                with torch.no_grad():
                    activ = model.forward_prepool(mel)
                    activ_u = untrained_model.forward_prepool(mel)
                low, mid, high = split_and_pool(activ)
                low_u, mid_u, high_u = split_and_pool(activ_u)
                X_low.append(low); X_mid.append(mid); X_high.append(high)
                X_low_u.append(low_u); X_mid_u.append(mid_u); X_high_u.append(high_u)
                loudness_values.append(compute_loudness_db(waveform))
            del data, crops
            os.remove(crops_local)
            print(f"[SOURCE] {name}: {len(loudness_values)} tracks processed so far (cumulative).")

        X_low, X_mid, X_high = np.stack(X_low), np.stack(X_mid), np.stack(X_high)
        X_low_u, X_mid_u, X_high_u = np.stack(X_low_u), np.stack(X_mid_u), np.stack(X_high_u)
        loudness_values = np.array(loudness_values)

        torch.save({"X_low": X_low, "X_mid": X_mid, "X_high": X_high,
                    "X_low_u": X_low_u, "X_mid_u": X_mid_u, "X_high_u": X_high_u,
                    "loudness_values": loudness_values}, embeddings_cache_local)
        copy_to_drive(embeddings_cache_local, embeddings_cache_remote)

    median_loudness = np.median(loudness_values)
    y = np.where(loudness_values >= median_loudness, "Loud", "Quiet")
    print(f"[LABELS] Median loudness: {median_loudness:.2f} dB. "
          f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
    print(f"[FEATURES] Shapes -- low: {X_low.shape}, mid: {X_mid.shape}, high: {X_high.shape}")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 5: fitting per-sub-band probes "
          f"(TRAINED model, 5-fold stratified CV)...")
    results = {}
    predictions = {}
    for band_name, X in [("low", X_low), ("mid", X_mid), ("high", X_high)]:
        metrics, y_oof, preds_oof = evaluate_subband(X, y, band_name)
        results[band_name] = metrics
        predictions[band_name] = (y_oof, preds_oof)

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 5b: fitting per-sub-band probes "
          f"(UNTRAINED baseline, same architecture, random weights)...")
    results_untrained = {}
    predictions_untrained = {}
    for band_name, X in [("low", X_low_u), ("mid", X_mid_u), ("high", X_high_u)]:
        metrics, y_oof, preds_oof = evaluate_subband(X, y, f"{band_name}_untrained")
        results_untrained[band_name] = metrics
        predictions_untrained[band_name] = (y_oof, preds_oof)

    y_oof_low, preds_low = predictions["low"]
    y_oof_mid, preds_mid = predictions["mid"]
    y_oof_high, preds_high = predictions["high"]
    y_oof_low_u, preds_low_u = predictions_untrained["low"]
    y_oof_mid_u, preds_mid_u = predictions_untrained["mid"]
    y_oof_high_u, preds_high_u = predictions_untrained["high"]
    assert all(np.array_equal(y_oof_low, arr) for arr in
               [y_oof_mid, y_oof_high, y_oof_low_u, y_oof_mid_u, y_oof_high_u]), \
        "Out-of-fold label arrays differ across bands/models -- fold split mismatch."

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 5c: paired bootstrap significance "
          f"tests -- trained pairwise, untrained pairwise, and trained-vs-untrained per band "
          f"(9 comparisons total, Bonferroni-corrected together)...")
    N_TOTAL_COMPARISONS = 9
    bootstrap_results = {
        "trained_low_vs_mid": paired_bootstrap_acc_diff(
            y_oof_low, preds_low, preds_mid, n_comparisons=N_TOTAL_COMPARISONS),
        "trained_mid_vs_high": paired_bootstrap_acc_diff(
            y_oof_low, preds_mid, preds_high, n_comparisons=N_TOTAL_COMPARISONS),
        "trained_low_vs_high": paired_bootstrap_acc_diff(
            y_oof_low, preds_low, preds_high, n_comparisons=N_TOTAL_COMPARISONS),
        "untrained_low_vs_mid": paired_bootstrap_acc_diff(
            y_oof_low, preds_low_u, preds_mid_u, n_comparisons=N_TOTAL_COMPARISONS),
        "untrained_mid_vs_high": paired_bootstrap_acc_diff(
            y_oof_low, preds_mid_u, preds_high_u, n_comparisons=N_TOTAL_COMPARISONS),
        "untrained_low_vs_high": paired_bootstrap_acc_diff(
            y_oof_low, preds_low_u, preds_high_u, n_comparisons=N_TOTAL_COMPARISONS),
        "trained_vs_untrained_low": paired_bootstrap_acc_diff(
            y_oof_low, preds_low, preds_low_u, n_comparisons=N_TOTAL_COMPARISONS),
        "trained_vs_untrained_mid": paired_bootstrap_acc_diff(
            y_oof_low, preds_mid, preds_mid_u, n_comparisons=N_TOTAL_COMPARISONS),
        "trained_vs_untrained_high": paired_bootstrap_acc_diff(
            y_oof_low, preds_high, preds_high_u, n_comparisons=N_TOTAL_COMPARISONS),
    }
    for name, r in bootstrap_results.items():
        sig = "SIGNIFICANT" if r["significant_bonferroni"] else "not significant"
        print(f"[BOOTSTRAP] {name}: observed_diff={r['observed_diff']:.4f}  "
              f"95% CI=({r['ci_95_low']:.4f}, {r['ci_95_high']:.4f})  "
              f"Bonferroni-corrected (n={N_TOTAL_COMPARISONS}): {sig}")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 6: saving results and figure...")
    local_results_path = os.path.join(LOCAL_ROOT, "11_subband_leakage_results.json")
    elapsed_seconds = time.time() - RUN_START_TIME
    output = {
        "program": "11_e3_subband_leakage_probe.py",
        "n_tracks": int(len(y)),
        "median_loudness_db": float(median_loudness),
        "loudness_label_note": "median-split for probe balance -- NOT a claim of real bimodal "
                                "loudness distribution in the data",
        "class_distribution": pd.Series(y).value_counts().to_dict(),
        "results_trained": results,
        "results_untrained": results_untrained,
        "subband_bootstrap_significance": bootstrap_results,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "elapsed_hms": time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds)),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(local_results_path, "w") as f:
        json.dump(output, f, indent=2)

    bands = ["low", "mid", "high"]
    trained_accs = [results[b]["accuracy"] for b in bands]
    untrained_accs = [results_untrained[b]["accuracy"] for b in bands]
    x = np.arange(len(bands))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, trained_accs, width, label="Trained E1 backbone", color="#1f77b4")
    ax.bar(x + width / 2, untrained_accs, width, label="Untrained baseline", color="#aec7e8")
    ax.axhline(0.5, color="gray", linestyle="--", label="Chance (balanced binary)")
    ax.set_ylabel("Loudness probe accuracy")
    ax.set_title("E3: Loudness leakage by Mel sub-band -- trained vs untrained")
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_ylim(0, 1.0)
    ax.legend()
    fig.tight_layout()
    local_fig_path = os.path.join(LOCAL_ROOT, "fig_11_subband_leakage_comparison.png")
    fig.savefig(local_fig_path, dpi=300)
    plt.close(fig)

    copy_to_drive(local_results_path, "11_subband_leakage_results.json")
    copy_to_drive(local_fig_path, "fig_11_subband_leakage_comparison.png")

    return output


if __name__ == "__main__":
    summary = main()
    elapsed = time.time() - RUN_START_TIME
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Program 11_e3_subband_leakage_probe.py finished.")
    print(f"[TOTAL RUNNING TIME] {time.strftime('%H:%M:%S', time.gmtime(elapsed))} ({elapsed:.1f}s)")