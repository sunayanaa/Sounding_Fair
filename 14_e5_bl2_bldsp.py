# =============================================================================
# Program      : 14_e5_bl2_bldsp.py
# Version      : 1.0
# GPU Required : NO (uses GPU automatically if available, for speed only)
# Description  : Experiment 5, Part 1 -- the two "cheap" baselines in the
#                 Baseline Comparison Suite: BL2 (FA*IR post-hoc re-ranking)
#                 and BL-DSP (loudness/EQ normalization, non-learned sanity
#                 check). BL3 (HSIC-adversarial retraining) and BL5 (HSIC-
#                 without-dVAE) are separate programs given their very
#                 different cost (both need actual training; these two do
#                 not). BL6 is deferred to E6 itself -- it's defined as
#                 "the full method minus the penalty term," which requires
#                 the full method to exist first.
#
#                 BL2 -- FA*IR (Zehlike et al., 2017), adapted from its
#                 original binary-protected-group re-ranking to this
#                 project's 5-group retrieval setting: for each query,
#                 take a wider candidate pool (top-M by cosine similarity,
#                 M=50) and greedily re-rank down to top-K=10, at each
#                 rank position picking the group furthest below its
#                 target proportional representation so far, subject to
#                 only choosing among the top-M candidates (bounding how
#                 far a pick can reach beyond pure similarity, in FA*IR's
#                 spirit of only re-ranking among "qualified" candidates).
#                 No retraining needed -- reuses Program 13's cached pool
#                 embeddings directly.
#
#                 BL-DSP -- loudness/EQ normalization directly on the
#                 audio, no learning at all. Reference spectral envelope
#                 is computed from a sample of the DOMINANT pool
#                 specifically (not a pool-wide average) -- the point of
#                 this baseline is testing whether moving every
#                 tradition's recordings toward the dominant pool's
#                 typical production characteristics, via simple signal
#                 processing alone, reduces the E4 disparities. STFT-
#                 domain gain correction (mel-filterbank-weighted,
#                 upsampled to linear frequency bins) plus RMS loudness
#                 normalization; corrected audio is then run through the
#                 ALREADY-TRAINED E1 encoder (no retraining) to get
#                 embeddings for the same E4-style pool-level metrics.
#
# INPUT:
#                  Google Drive:
#                    08_crops_fma.pt, 08_crops_carnatic.pt,
#                    08_crops_hindustani.pt, 08_crops_jingju_base.pt,
#                    08_crops_jacrc.pt, 08_crops_arab_andalusian.pt,
#                    08_crops_jamendo.pt, 08_backbone_final.pt
#                    13_pool_embeddings.pt (Program 13's cache, for BL2)
#
# STEPS:
#                  Step 1  Ensure PROJECT_DIR exists
#                  Step 2  BL2: load cached pool embeddings, fair-rerank
#                          top-K per query, recompute the 3 E4 pool-level
#                          metrics with bootstrap CIs; save+copy
#                          immediately (independent of BL-DSP below)
#                  Step 3  BL-DSP: copy trained backbone + all 7 crop
#                          caches; compute a reference mel envelope from
#                          a Dominant-pool sample; for every track, apply
#                          STFT-domain EQ correction + RMS normalization,
#                          then embed via the trained encoder
#                  Step 4  Recompute the 3 E4 pool-level metrics on the
#                          BL-DSP embeddings, same bootstrap methodology
#                  Step 5  Save results JSON + comparison figures, copy
#                          to Google Drive
#
# OUTPUT FILES : under PROJECT_DIR
#                  14_bl2_fair_results.json
#                  14_bldsp_results.json
#                  fig_14_bl2_vs_vanilla_heatmap.png
#                  fig_14_bldsp_vs_vanilla_heatmap.png
#
# Hyperparameters : SAMPLE_RATE=22050, SEGMENT_SECONDS=4, N_MELS=64,
#                    EMBEDDING_DIM=256 (must match Program 08),
#                    RETRIEVAL_K=10, FAIR_CANDIDATE_POOL_M=50,
#                    REFERENCE_ENVELOPE_SAMPLE_SIZE=500,
#                    N_FFT=1024, HOP_LENGTH=256, TARGET_RMS=0.1,
#                    N_BOOTSTRAP_ITERATIONS=10000, RANDOM_SEED=42.
# Dependencies : torch, torchaudio, librosa, numpy, pandas, tqdm,
#                 matplotlib
#
# Change Log   :
#   v1.0  2026-09-23  Initial version
# Running time: 6 hrs
# !pip install torch torchaudio librosa pandas tqdm matplotlib
# =============================================================================

!pip install torch torchaudio librosa pandas tqdm matplotlib

import os
import time
import json
from datetime import datetime

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

RUN_START_TIME = time.time()
print(f"[{datetime.now().strftime('%H:%M:%S')}] Program 14_e5_bl2_bldsp.py started.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] Using {DEVICE} (GPU not required; used automatically if present).")

# --- Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/sounding-fair/"

SAMPLE_RATE = 22050
SEGMENT_SECONDS = 4
SEGMENT_LENGTH = SAMPLE_RATE * SEGMENT_SECONDS
N_MELS = 64
EMBEDDING_DIM = 256
PROJECTION_DIM = 128
RETRIEVAL_K = 10
FAIR_CANDIDATE_POOL_M = 50
REFERENCE_ENVELOPE_SAMPLE_SIZE = 500
N_FFT = 1024
HOP_LENGTH = 256
TARGET_RMS = 0.1
N_BOOTSTRAP_ITERATIONS = 10000
RANDOM_SEED = 42

POOL_SOURCES = ["fma", "carnatic", "hindustani", "jingju_base", "jacrc", "arab_andalusian", "jamendo"]
SOURCE_TO_GROUP = {
    "fma": "Dominant", "jamendo": "Dominant",
    "carnatic": "Carnatic", "hindustani": "Hindustani",
    "jingju_base": "Jingju", "jacrc": "Jingju",
    "arab_andalusian": "ArabAndalusian",
}
GROUPS = ["Dominant", "Carnatic", "Hindustani", "Jingju", "ArabAndalusian"]
DOMINANT_SOURCES = ["fma", "jamendo"]

LOCAL_ROOT = "/content/local_data_e5"
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


# --- Same AudioEncoder architecture as Program 08/11/13 ---
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


def load_trained_encoder(checkpoint_path):
    model = AudioEncoder().to(DEVICE)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model


_MEL_TRANSFORM = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS).to(DEVICE)
_AMPLITUDE_TO_DB = torchaudio.transforms.AmplitudeToDB().to(DEVICE)


def embed_float_waveform(model, waveform_float_np):
    """waveform_float_np: already-normalized [-1,1] float array (e.g. from
    BL-DSP's EQ correction) -- not int16, so no /32767 division here."""
    wf = torch.tensor(waveform_float_np, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
    mel = _AMPLITUDE_TO_DB(_MEL_TRANSFORM(wf))
    with torch.no_grad():
        embedding, _ = model(mel)
    return embedding.squeeze(0).cpu().numpy()


def bootstrap_mean_ci(values, n_iterations=N_BOOTSTRAP_ITERATIONS, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    n = len(values)
    observed = float(np.mean(values))
    means = np.empty(n_iterations)
    for i in range(n_iterations):
        idx = rng.randint(0, n, size=n)
        means[i] = values[idx].mean()
    ci_low, ci_high = np.percentile(means, [2.5, 97.5])
    return {"observed": observed, "ci_95_low": float(ci_low), "ci_95_high": float(ci_high)}


def compute_pool_level_metrics(fractions, groups):
    """Identical to Program 13's version -- same 3 metrics from the same
    underlying [N_queries x 5-group] fraction matrix."""
    group_idx = np.array([GROUPS.index(g) for g in groups])
    pool_counts = pd.Series(groups).value_counts()
    pool_proportions = {g: pool_counts.get(g, 0) / len(groups) for g in GROUPS}

    cross_tradition_matrix = {}
    self_consistency = {}
    for g_source in GROUPS:
        mask = group_idx == GROUPS.index(g_source)
        if mask.sum() == 0:
            continue
        row = {}
        for g_target in GROUPS:
            col = GROUPS.index(g_target)
            values = fractions[mask, col]
            row[g_target] = bootstrap_mean_ci(values)
        cross_tradition_matrix[g_source] = row
        self_consistency[g_source] = row[g_source]

    exposure_disparity = {}
    for g_target in GROUPS:
        col = GROUPS.index(g_target)
        values = fractions[:, col]
        result = bootstrap_mean_ci(values)
        p_g = pool_proportions[g_target]
        result["pool_proportion"] = p_g
        result["disparity_ratio"] = result["observed"] / p_g if p_g > 0 else None
        result["disparity_ratio_ci_low"] = result["ci_95_low"] / p_g if p_g > 0 else None
        result["disparity_ratio_ci_high"] = result["ci_95_high"] / p_g if p_g > 0 else None
        exposure_disparity[g_target] = result

    return {
        "pool_proportions": pool_proportions,
        "self_consistency": self_consistency,
        "cross_tradition_retrievability_matrix": cross_tradition_matrix,
        "exposure_disparity": exposure_disparity,
    }


def save_heatmap(pool_metrics, title, local_path):
    matrix_vals = np.array([[pool_metrics["cross_tradition_retrievability_matrix"][gs][gt]["observed"]
                              for gt in GROUPS] for gs in GROUPS])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix_vals, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(GROUPS))); ax.set_xticklabels(GROUPS, rotation=30, ha="right")
    ax.set_yticks(range(len(GROUPS))); ax.set_yticklabels(GROUPS)
    ax.set_xlabel("Retrieved group"); ax.set_ylabel("Query group")
    ax.set_title(title)
    for i in range(len(GROUPS)):
        for j in range(len(GROUPS)):
            ax.text(j, i, f"{matrix_vals[i, j]:.2f}", ha="center", va="center",
                     color="white" if matrix_vals[i, j] < 0.5 else "black")
    fig.colorbar(im, ax=ax, label="Fraction of top-K")
    fig.tight_layout()
    fig.savefig(local_path, dpi=300)
    plt.close(fig)


# =============================================================================
# BL2 -- FA*IR re-ranking
# =============================================================================

def fair_rerank_topk(sims_row, group_idx_all, target_proportions, k=RETRIEVAL_K, m=FAIR_CANDIDATE_POOL_M):
    """sims_row: [N] similarity of one query against the full pool (self
    already excluded, i.e. set to -inf). Returns the fair-reranked top-K
    INDICES. Greedy: at each rank position, among the top-M candidates by
    raw similarity not yet picked, choose the group furthest below its
    target proportional share so far, then pick that group's highest-
    similarity remaining candidate."""
    top_m_idx = np.argpartition(-sims_row, m)[:m]
    top_m_idx = top_m_idx[np.argsort(-sims_row[top_m_idx])]  # sorted by similarity desc

    picked = []
    group_counts = {g: 0 for g in GROUPS}

    remaining = list(top_m_idx)
    for rank in range(1, k + 1):
        if not remaining:
            break
        deficits = {g: target_proportions[g] * rank - group_counts[g] for g in GROUPS}
        # Groups actually present among remaining candidates
        present_groups = set(GROUPS[group_idx_all[idx]] for idx in remaining)
        best_group = max((g for g in present_groups), key=lambda g: deficits[g])
        # Highest-similarity remaining candidate belonging to best_group
        for idx in remaining:
            if GROUPS[group_idx_all[idx]] == best_group:
                picked.append(idx)
                group_counts[best_group] += 1
                remaining.remove(idx)
                break
    return np.array(picked[:k])


def run_bl2(pool_embeddings, pool_groups):
    emb_t = torch.tensor(pool_embeddings, device=DEVICE)
    emb_norm = torch.nn.functional.normalize(emb_t, dim=1)
    n = emb_norm.shape[0]
    group_idx = np.array([GROUPS.index(g) for g in pool_groups])
    pool_counts = pd.Series(pool_groups).value_counts()
    target_proportions = {g: pool_counts.get(g, 0) / n for g in GROUPS}

    fractions = np.zeros((n, len(GROUPS)), dtype=np.float32)
    batch_size = 500
    for start in tqdm(range(0, n, batch_size), desc="[BL2] fair rerank"):
        end = min(start + batch_size, n)
        batch = emb_norm[start:end]
        sims = (batch @ emb_norm.T).cpu().numpy()
        row_ids = np.arange(start, end)
        sims[np.arange(end - start), row_ids] = -np.inf
        for local_i, global_i in enumerate(row_ids):
            reranked_idx = fair_rerank_topk(sims[local_i], group_idx, target_proportions)
            reranked_groups = group_idx[reranked_idx]
            for g_i in range(len(GROUPS)):
                fractions[global_i, g_i] = np.mean(reranked_groups == g_i) if len(reranked_groups) > 0 else 0.0

    return compute_pool_level_metrics(fractions, pool_groups)


# =============================================================================
# BL-DSP -- loudness/EQ normalization
# =============================================================================

_MEL_FB = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS)  # [N_MELS, n_freq_bins]
_MEL_FB_NORM = _MEL_FB / (_MEL_FB.sum(axis=0, keepdims=True) + 1e-10)  # per-freq-bin weighted upsampling


def compute_mel_envelope(waveform_np):
    mel = librosa.feature.melspectrogram(y=waveform_np, sr=SAMPLE_RATE, n_fft=N_FFT,
                                          hop_length=HOP_LENGTH, n_mels=N_MELS)
    return librosa.power_to_db(mel).mean(axis=1)  # [N_MELS]


def apply_loudness_eq_normalization(waveform_np, reference_mel_envelope):
    stft = librosa.stft(waveform_np, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude, phase = np.abs(stft), np.angle(stft)

    track_envelope = compute_mel_envelope(waveform_np)
    gain_db_per_mel = reference_mel_envelope - track_envelope
    gain_linear_per_mel = 10 ** (gain_db_per_mel / 20)
    gain_linear_per_bin = _MEL_FB_NORM.T @ gain_linear_per_mel
    gain_linear_per_bin = np.clip(gain_linear_per_bin, 0.1, 10.0)  # bound extreme corrections

    corrected_magnitude = magnitude * gain_linear_per_bin[:, None]
    corrected_stft = corrected_magnitude * np.exp(1j * phase)
    corrected_waveform = librosa.istft(corrected_stft, hop_length=HOP_LENGTH, length=len(waveform_np))

    current_rms = np.sqrt(np.mean(corrected_waveform ** 2) + 1e-10)
    corrected_waveform = corrected_waveform * (TARGET_RMS / current_rms)
    return np.clip(corrected_waveform, -1.0, 1.0).astype(np.float32)


def run_bldsp(model):
    print(f"[BL-DSP] Computing reference envelope from {REFERENCE_ENVELOPE_SAMPLE_SIZE} "
          f"Dominant-pool tracks...")
    reference_envelopes = []
    rng = np.random.RandomState(RANDOM_SEED)
    for name in DOMINANT_SOURCES:
        crops_local = os.path.join(LOCAL_ROOT, f"crops_{name}.pt")
        if not copy_from_drive(f"08_crops_{name}.pt", crops_local, skip_if_exists=True):
            continue
        data = torch.load(crops_local, map_location="cpu", weights_only=False)
        crops = data["crops"]
        n_sample = min(REFERENCE_ENVELOPE_SAMPLE_SIZE // len(DOMINANT_SOURCES), crops.shape[0])
        idx = rng.choice(crops.shape[0], size=n_sample, replace=False)
        for i in tqdm(idx, desc=f"[BL-DSP] reference envelope from {name}"):
            wf = (crops[i, 0].float() / 32767.0).numpy()
            reference_envelopes.append(compute_mel_envelope(wf))
        del data, crops
        os.remove(crops_local)
    reference_mel_envelope = np.mean(reference_envelopes, axis=0)
    print(f"[BL-DSP] Reference envelope computed from {len(reference_envelopes)} tracks.")

    print(f"[BL-DSP] Applying EQ+loudness correction and embedding the full pool...")
    embeddings, groups = [], []
    for name in POOL_SOURCES:
        crops_local = os.path.join(LOCAL_ROOT, f"crops_{name}.pt")
        if not copy_from_drive(f"08_crops_{name}.pt", crops_local, skip_if_exists=True):
            print(f"[WARNING] Could not fetch 08_crops_{name}.pt -- skipping.")
            continue
        data = torch.load(crops_local, map_location="cpu", weights_only=False)
        crops = data["crops"]
        group = SOURCE_TO_GROUP[name]
        for i in tqdm(range(crops.shape[0]), desc=f"[BL-DSP] {name}"):
            crop_embeddings = []
            for c in range(crops.shape[1]):
                wf = (crops[i, c].float() / 32767.0).numpy()
                corrected = apply_loudness_eq_normalization(wf, reference_mel_envelope)
                crop_embeddings.append(embed_float_waveform(model, corrected))
            embeddings.append(np.mean(crop_embeddings, axis=0))
            groups.append(group)
        del data, crops
        os.remove(crops_local)
        print(f"[BL-DSP] {name}: {len(groups)} tracks embedded so far (cumulative).")

    embeddings = np.stack(embeddings).astype(np.float32)
    groups = np.array(groups)

    cache_local = os.path.join(LOCAL_ROOT, "14_bldsp_embeddings.pt")
    torch.save({"embeddings": embeddings, "groups": groups}, cache_local)
    copy_to_drive(cache_local, "14_bldsp_embeddings.pt")

    fractions = compute_bldsp_fractions(embeddings, groups)
    return compute_pool_level_metrics(fractions, groups)


def compute_bldsp_fractions(embeddings, groups, k=RETRIEVAL_K, batch_size=500):
    emb_t = torch.tensor(embeddings, device=DEVICE)
    emb_norm = torch.nn.functional.normalize(emb_t, dim=1)
    n = emb_norm.shape[0]
    group_idx = np.array([GROUPS.index(g) for g in groups])
    group_idx_t = torch.tensor(group_idx, device=DEVICE)

    fractions = np.zeros((n, len(GROUPS)), dtype=np.float32)
    for start in tqdm(range(0, n, batch_size), desc="[BL-DSP] retrieval top-K"):
        end = min(start + batch_size, n)
        batch = emb_norm[start:end]
        sims = batch @ emb_norm.T
        row_ids = torch.arange(start, end, device=DEVICE)
        sims[torch.arange(end - start), row_ids] = -float("inf")
        topk_idx = torch.topk(sims, k=k, dim=1).indices
        topk_groups = group_idx_t[topk_idx]
        for g_i in range(len(GROUPS)):
            fractions[start:end, g_i] = (topk_groups == g_i).float().mean(dim=1).cpu().numpy()
    return fractions


def main():
    ensure_project_dir()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 2: BL2 (FA*IR re-ranking)...")
    pool_cache_local = os.path.join(LOCAL_ROOT, "13_pool_embeddings.pt")
    if not copy_from_drive("13_pool_embeddings.pt", pool_cache_local, skip_if_exists=True):
        raise RuntimeError("Could not fetch 13_pool_embeddings.pt -- run Program 13 first.")
    cached = torch.load(pool_cache_local, map_location="cpu", weights_only=False)
    pool_embeddings, pool_groups = cached["embeddings"], cached["groups"]
    print(f"[BL2] Reusing {pool_embeddings.shape[0]} cached pool embeddings from Program 13.")

    bl2_metrics = run_bl2(pool_embeddings, pool_groups)
    print("[BL2] Self-consistency:")
    for g, r in bl2_metrics["self_consistency"].items():
        print(f"    {g:16s} {r['observed']:.4f}")

    bl2_output = {
        "program": "14_e5_bl2_bldsp.py", "baseline": "BL2_FAIR",
        "retrieval_k": RETRIEVAL_K, "candidate_pool_m": FAIR_CANDIDATE_POOL_M,
        "n_pool_tracks": int(pool_embeddings.shape[0]),
        "pool_level_metrics": bl2_metrics,
        "elapsed_seconds": round(time.time() - RUN_START_TIME, 1),
    }
    local_bl2_path = os.path.join(LOCAL_ROOT, "14_bl2_fair_results.json")
    with open(local_bl2_path, "w") as f:
        json.dump(bl2_output, f, indent=2)
    copy_to_drive(local_bl2_path, "14_bl2_fair_results.json")

    local_fig1 = os.path.join(LOCAL_ROOT, "fig_14_bl2_vs_vanilla_heatmap.png")
    save_heatmap(bl2_metrics, f"E5 BL2 (FA*IR): Cross-tradition retrievability (K={RETRIEVAL_K})", local_fig1)
    copy_to_drive(local_fig1, "fig_14_bl2_vs_vanilla_heatmap.png")
    print(f"[BL2] Results saved and copied independently of BL-DSP below.")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 3: BL-DSP (loudness/EQ normalization)...")
    checkpoint_local = os.path.join(LOCAL_ROOT, "08_backbone_final.pt")
    if not copy_from_drive("08_backbone_final.pt", checkpoint_local, skip_if_exists=True):
        raise RuntimeError("Could not fetch 08_backbone_final.pt -- run Program 08 first.")
    model = load_trained_encoder(checkpoint_local)

    bldsp_metrics = run_bldsp(model)
    print("[BL-DSP] Self-consistency:")
    for g, r in bldsp_metrics["self_consistency"].items():
        print(f"    {g:16s} {r['observed']:.4f}")

    bldsp_output = {
        "program": "14_e5_bl2_bldsp.py", "baseline": "BL_DSP",
        "retrieval_k": RETRIEVAL_K,
        "pool_level_metrics": bldsp_metrics,
        "elapsed_seconds": round(time.time() - RUN_START_TIME, 1),
    }
    local_bldsp_path = os.path.join(LOCAL_ROOT, "14_bldsp_results.json")
    with open(local_bldsp_path, "w") as f:
        json.dump(bldsp_output, f, indent=2)
    copy_to_drive(local_bldsp_path, "14_bldsp_results.json")

    local_fig2 = os.path.join(LOCAL_ROOT, "fig_14_bldsp_vs_vanilla_heatmap.png")
    save_heatmap(bldsp_metrics, f"E5 BL-DSP: Cross-tradition retrievability (K={RETRIEVAL_K})", local_fig2)
    copy_to_drive(local_fig2, "fig_14_bldsp_vs_vanilla_heatmap.png")

    return bl2_output, bldsp_output


if __name__ == "__main__":
    summary = main()
    elapsed = time.time() - RUN_START_TIME
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Program 14_e5_bl2_bldsp.py finished.")
    print(f"[TOTAL RUNNING TIME] {time.strftime('%H:%M:%S', time.gmtime(elapsed))} ({elapsed:.1f}s)")