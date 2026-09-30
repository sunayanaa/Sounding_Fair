# =============================================================================
# Program      : 16_e5_bl3_hsic_adversarial.py
# Version      : 1.1
# GPU Required : YES (full CNN backbone retraining, comparable cost to
#                 Program 08)
# Description  : Experiment 5, Part 3 -- BL3, the standard adversarial
#                 fairness paradigm: retrain the SAME AudioEncoder
#                 architecture and NT-Xent contrastive objective as E1
#                 (Program 08), with an added HSIC dependence penalty
#                 between the embedding (256-dim, pre-projection) and the
#                 tradition-group one-hot label, computed per batch.
#                 Total loss = NT-Xent + lambda * HSIC(embedding, group).
#                 No style-transfer augmentation, no content/style
#                 disentanglement -- this is the most direct competitor
#                 to E6's own Stage-2 contribution, since both approaches
#                 aim to reduce demographic/tradition information in the
#                 embedding space, but BL3 does so without E6's
#                 disentanglement step.
#
#                 Two deliberate departures from E1's exact setup, both
#                 documented rather than silently applied:
#                   1. STRATIFIED batches (20 tracks/group, 100 total per
#                      batch) instead of E1's natural random sampling
#                      from the 94:6-imbalanced pool. A random batch from
#                      that pool would be almost entirely Dominant
#                      examples, giving HSIC little signal about the 4
#                      minority traditions -- same reasoning as BL5's
#                      stratified sampling, and the standard practice for
#                      this kind of adversarial-debiasing training in the
#                      literature. This does mean BL3's contrastive
#                      negative-sampling distribution differs from
#                      vanilla E1's, a genuine confound worth flagging
#                      rather than treating the comparison as perfectly
#                      clean.
#                   2. lambda_hsic=1.0, matching BL5 for comparability
#                      between the two HSIC-penalized baselines -- not
#                      independently tuned for BL3.
#
#                 LESSON APPLIED FROM THE USER'S FEEDBACK ON PROGRAM 15:
#                 crop caches are loaded into a SHARED local path
#                 (/content/shared_crops_cache/) and are NOT deleted
#                 after use. Every prior program (08, 11, 13, 14, 15)
#                 downloaded the same 7 crop caches into its own private,
#                 self-deleting local folder, so nothing ever persisted
#                 for the next program to reuse -- a real, repeated waste
#                 across this whole pipeline that was only caught after
#                 the fact. This program is the first to fix it going
#                 forward: whatever runs after this one, in the same
#                 session, can check the shared path first and skip the
#                 Drive copy entirely.
#
#                 All inputs and outputs for this program live in
#                 PROJECT_DIR on Google Drive. There is no separate
#                 cache/mirror directory: every checkpoint, results JSON,
#                 figure PNG, and .pt cache produced anywhere in this
#                 pipeline resides under PROJECT_DIR, and this program
#                 reads its inputs (crop caches, resume checkpoints)
#                 from that same location.
#
#                 Same hard-won lessons from Program 08's training saga
#                 are applied directly: per-source tensors kept as a
#                 LIST (never concatenated) to avoid the RAM-doubling
#                 bugs that took multiple attempts to fix there;
#                 bounded, resumable epoch-level checkpointing (a
#                 Colab disconnect costs at most one epoch, not the
#                 whole run); the same two-view augmentation and NT-Xent
#                 formulation as E1's own training.
#
# INPUT:
#                  Google Drive (all under PROJECT_DIR):
#                    08_crops_fma.pt, 08_crops_carnatic.pt,
#                    08_crops_hindustani.pt, 08_crops_jingju_base.pt,
#                    08_crops_jacrc.pt, 08_crops_arab_andalusian.pt,
#                    08_crops_jamendo.pt, and any resume checkpoint
#
# STEPS:
#                  Step 1  Mount Drive, ensure PROJECT_DIR exists
#                  Step 2  Load all 7 crop caches into the SHARED,
#                          non-deleting local cache directory (shared
#                          local cache first, then PROJECT_DIR);
#                          build per-source tensor list + matching
#                          group labels
#                  Step 3  Check for an existing BL3 checkpoint in
#                          PROJECT_DIR; resume from it if present,
#                          otherwise start a fresh AudioEncoder
#                  Step 4  Train: stratified batches, two augmented views
#                          per track (gain+noise, independently sampled
#                          crop positions, matching E1), NT-Xent +
#                          lambda*HSIC(embedding, group) loss; checkpoint
#                          to PROJECT_DIR after every epoch
#                  Step 5  Encode the full pool through the trained
#                          model; compute the 3 E4-style pool-level
#                          metrics with bootstrap CIs
#                  Step 6  Save results JSON + heatmap figure to
#                          PROJECT_DIR
#
# OUTPUT FILES : under PROJECT_DIR
#                  16_bl3_checkpoint.pt       (model+optimizer state,
#                                              epoch, training history --
#                                              resumable)
#                  16_bl3_final_model.pt      (final trained weights only)
#                  16_bl3_results.json
#                  fig_16_bl3_vs_vanilla_heatmap.png
#
# Hyperparameters : SAMPLE_RATE=22050, SEGMENT_SECONDS=4, N_MELS=64,
#                    EMBEDDING_DIM=256, PROJECTION_DIM=128,
#                    BATCH_SIZE=100 (20/group, stratified),
#                    STEPS_PER_EPOCH=170, TOTAL_EPOCHS=50, LR=3e-4,
#                    TEMPERATURE=0.5 (NT-Xent), LAMBDA_HSIC=1.0 (matches
#                    BL5, not independently tuned), RETRIEVAL_K=10,
#                    N_BOOTSTRAP_ITERATIONS=10000, RANDOM_SEED=42.
# Dependencies : torch, torchaudio, numpy, pandas, tqdm, matplotlib
#
# Change Log   :
#   v1.0  2026-09-24  Initial version
#   v1.1  2026-09-24  Fixed the same skip-check bug caught in Program 17:
#                      ftp.size() needs TYPE I (binary) mode on this
#                      server, and the old fallback treated "couldn't
#                      verify remote size" as "skip" instead of "re-
#                      download to be safe" -- exactly backwards. Also
#                      added a Drive-first fetch chain (local shared
#                      cache -> Drive cache -> FTP as a last resort),
#                      since Program 17 now bulk-downloads everything to
#                      Drive once; FTP should rarely if ever be hit by
#                      this program going forward.
# Running time:
# !pip install torch torchaudio pandas tqdm matplotlib
# =============================================================================

!pip install -q  torch torchaudio pandas tqdm matplotlib

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

from google.colab import drive
drive.mount('/content/drive')

RUN_START_TIME = time.time()
print(f"[{datetime.now().strftime('%H:%M:%S')}] Program 16_e5_bl3_hsic_adversarial.py started.")

if not torch.cuda.is_available():
    print("[WARNING] No GPU detected. This program retrains a full CNN backbone "
          "(comparable cost to Program 08) and will be extremely slow on CPU.")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] Using {DEVICE}.")

# --- Configuration ---
# PROJECT_DIR is the single source of truth for the pipeline: every
# checkpoint, results JSON, figure PNG, and .pt/.npy cache produced by
# any program in this project lives here, and every program reads its
# inputs from here too. No separate cache/mirror directory exists.
PROJECT_DIR = "/content/drive/MyDrive/paper/sounding-fair/"

SAMPLE_RATE = 22050
SEGMENT_SECONDS = 4
SEGMENT_LENGTH = SAMPLE_RATE * SEGMENT_SECONDS
N_MELS = 64
EMBEDDING_DIM = 256
PROJECTION_DIM = 128
BATCH_SIZE = 100  # 20 per group, stratified
STEPS_PER_EPOCH = 170
TOTAL_EPOCHS = 50
LEARNING_RATE = 3e-4
TEMPERATURE = 0.5
LAMBDA_HSIC = 1.0  # matches BL5 (Program 15), not independently tuned
RETRIEVAL_K = 10
N_BOOTSTRAP_ITERATIONS = 10000
RANDOM_SEED = 42
GAIN_DB_RANGE = 6.0
NOISE_STD = 0.005

POOL_SOURCES = ["fma", "carnatic", "hindustani", "jingju_base", "jacrc", "arab_andalusian", "jamendo"]
SOURCE_TO_GROUP = {
    "fma": "Dominant", "jamendo": "Dominant",
    "carnatic": "Carnatic", "hindustani": "Hindustani",
    "jingju_base": "Jingju", "jacrc": "Jingju",
    "arab_andalusian": "ArabAndalusian",
}
GROUPS = ["Dominant", "Carnatic", "Hindustani", "Jingju", "ArabAndalusian"]

# Shared, non-deleting local cache -- the fix going forward for the
# per-program re-download waste identified during Program 15.
SHARED_CACHE_DIR = "/content/shared_crops_cache"
os.makedirs(SHARED_CACHE_DIR, exist_ok=True)
LOCAL_ROOT = "/content/local_data_e5_bl3"
os.makedirs(LOCAL_ROOT, exist_ok=True)

# --- Begin Google Drive Helper Functions ---
def ensure_project_dir():
    """Create PROJECT_DIR on Google Drive if it doesn't already exist."""
    os.makedirs(PROJECT_DIR, exist_ok=True)
    print(f"[Drive] Project directory ready: {PROJECT_DIR}")


def copy_to_drive(local_filepath, remote_filename):
    """Copy a local file to PROJECT_DIR on Google Drive."""
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
    """Copy a file from PROJECT_DIR on Google Drive to local disk.
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


_MEL_TRANSFORM = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=N_MELS).to(DEVICE)
_AMPLITUDE_TO_DB = torchaudio.transforms.AmplitudeToDB().to(DEVICE)


def crop_to_mel_batch(waveform_batch):
    """waveform_batch: [B, SEGMENT_LENGTH] float32 in [-1,1]. Returns
    [B, 1, N_MELS, T] mel-spectrograms."""
    wf = waveform_batch.unsqueeze(1)  # [B, 1, SEGMENT_LENGTH]
    return _AMPLITUDE_TO_DB(_MEL_TRANSFORM(wf))


def augment_waveform(waveform_float):
    """Gain perturbation (+/-6dB) + additive Gaussian noise, matching
    E1's augmentation pipeline."""
    gain_db = (torch.rand(1, device=waveform_float.device) * 2 - 1) * GAIN_DB_RANGE
    gain_linear = 10 ** (gain_db / 20)
    augmented = waveform_float * gain_linear
    noise = torch.randn_like(augmented) * NOISE_STD
    return torch.clamp(augmented + noise, -1.0, 1.0)


def copy_from_drive_with_progress(src, dst):
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
        print(f"[COPY] {dst} already present locally, skipping copy.")
        return
    total_size = os.path.getsize(src)
    chunk_size = 1024 * 1024 * 8
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst, tqdm(
        total=total_size, unit="B", unit_scale=True, unit_divisor=1024,
        desc=f"[DRIVE] {os.path.basename(src)}",
    ) as pbar:
        while True:
            chunk = fsrc.read(chunk_size)
            if not chunk:
                break
            fdst.write(chunk)
            pbar.update(len(chunk))


def fetch_pool_source(remote_filename, shared_path):
    """Tiered fetch for the bulk input crop caches:
      1. Local shared cache (free, best case after any prior program).
      2. PROJECT_DIR on Google Drive (the single source of truth for
         everything this pipeline produces).
    A file is only accepted from the shared cache if its size matches
    the Drive copy's -- a stale partial file from an earlier interrupted
    run would otherwise be silently treated as complete."""
    drive_path = os.path.join(PROJECT_DIR, remote_filename)
    drive_size = os.path.getsize(drive_path) if os.path.exists(drive_path) else None

    if os.path.exists(shared_path) and os.path.getsize(shared_path) > 0:
        local_size = os.path.getsize(shared_path)
        if drive_size is not None and local_size != drive_size:
            print(f"[LOCAL] {remote_filename} present locally but size differs from Drive "
                  f"({local_size} vs {drive_size} bytes) -- treating as a stale/partial file "
                  f"and re-copying from Drive.")
            os.remove(shared_path)
        else:
            print(f"[LOCAL] {remote_filename} already present at {shared_path}.")
            return True

    if drive_size is not None and drive_size > 0:
        print(f"[DRIVE] {remote_filename} found in {PROJECT_DIR} -- copying locally.")
        copy_from_drive_with_progress(drive_path, shared_path)
        return True

    print(f"[ERROR] {remote_filename} not found locally or in {PROJECT_DIR}.")
    return False


def download_pool_crops_shared():
    """Loads all 7 crop caches via the shared/Drive fallback chain above
    into the SHARED, non-deleting cache dir. Returns a list of
    (source_name, group, crops_tensor) -- crops kept as a LIST across
    sources, never concatenated, per Program 08's hard-won RAM lesson."""
    per_source = []
    for name in POOL_SOURCES:
        shared_path = os.path.join(SHARED_CACHE_DIR, f"08_crops_{name}.pt")
        if not fetch_pool_source(f"08_crops_{name}.pt", shared_path):
            print(f"[WARNING] Could not fetch 08_crops_{name}.pt -- skipping this source.")
            continue
        data = torch.load(shared_path, map_location="cpu", weights_only=False)
        crops = data["crops"]  # [N, 3, SEGMENT_LENGTH] int16
        per_source.append((name, SOURCE_TO_GROUP[name], crops))
        print(f"[SHARED-CACHE] {name}: {crops.shape[0]} tracks (kept in place at {shared_path}, "
              f"not deleted -- available to any later program in this session).")
    return per_source


def build_group_indices(per_source):
    """Maps each GROUP to a list of (source_idx_in_per_source, row_idx)
    pairs, so stratified sampling can draw from the right underlying
    tensor(s) -- Jingju spans two sources (jingju_base + jacrc)."""
    group_to_locations = {g: [] for g in GROUPS}
    for src_idx, (name, group, crops) in enumerate(per_source):
        for row in range(crops.shape[0]):
            group_to_locations[group].append((src_idx, row))
    for g, locs in group_to_locations.items():
        print(f"[GROUPS] {g}: {len(locs)} tracks available for stratified sampling.")
    return group_to_locations


def sample_stratified_batch(per_source, group_to_locations, batch_size, n_groups, rng):
    per_group = batch_size // n_groups
    view1_list, view2_list, group_idx_list = [], [], []
    for g_i, g in enumerate(GROUPS):
        locations = group_to_locations[g]
        n_available = len(locations)
        chosen = rng.choice(n_available, size=per_group, replace=(n_available < per_group))
        for loc_idx in chosen:
            src_idx, row = locations[loc_idx]
            crops = per_source[src_idx][2]  # [n_crops, SEGMENT_LENGTH] for this track
            n_crops = crops.shape[1]
            c1, c2 = rng.choice(n_crops, size=2, replace=(n_crops < 2))
            wf1 = (crops[row, c1].float() / 32767.0)
            wf2 = (crops[row, c2].float() / 32767.0)
            view1_list.append(wf1)
            view2_list.append(wf2)
            group_idx_list.append(g_i)
    view1 = torch.stack(view1_list).to(DEVICE)
    view2 = torch.stack(view2_list).to(DEVICE)
    view1 = torch.stack([augment_waveform(w) for w in view1])
    view2 = torch.stack([augment_waveform(w) for w in view2])
    group_idx = torch.tensor(group_idx_list, device=DEVICE)
    return view1, view2, group_idx


def nt_xent_loss(proj1, proj2, temperature=TEMPERATURE):
    """Standard SimCLR NT-Xent loss, matching E1's formulation."""
    batch_size = proj1.shape[0]
    z = torch.cat([proj1, proj2], dim=0)
    z = F.normalize(z, dim=1)
    sim = z @ z.T / temperature
    sim.fill_diagonal_(-float("inf"))
    positive_idx = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z.device),
        torch.arange(0, batch_size, device=z.device),
    ])
    return F.cross_entropy(sim, positive_idx)


def rbf_kernel_median_heuristic(X):
    pairwise_sq_dists = torch.cdist(X, X) ** 2
    nonzero = pairwise_sq_dists[pairwise_sq_dists > 0]
    sigma_sq = torch.median(nonzero) + 1e-8 if nonzero.numel() > 0 else torch.tensor(1.0, device=X.device)
    return torch.exp(-pairwise_sq_dists / (2 * sigma_sq))


def hsic_loss(Z, G_onehot):
    B = Z.shape[0]
    K_Z = rbf_kernel_median_heuristic(Z)
    K_G = G_onehot @ G_onehot.T
    H = torch.eye(B, device=Z.device) - torch.ones(B, B, device=Z.device) / B
    return torch.trace(K_Z @ H @ K_G @ H) / ((B - 1) ** 2)


def try_resume_checkpoint():
    checkpoint_local = os.path.join(LOCAL_ROOT, "16_bl3_checkpoint.pt")
    if copy_from_drive("16_bl3_checkpoint.pt", checkpoint_local, skip_if_exists=True):
        checkpoint = torch.load(checkpoint_local, map_location=DEVICE, weights_only=False)
        print(f"[RESUME] Found checkpoint at epoch {checkpoint['epoch']}. Resuming.")
        return checkpoint
    print("[RESUME] No checkpoint found. Starting fresh.")
    return None


def save_checkpoint(model, optimizer, epoch, history):
    checkpoint_local = os.path.join(LOCAL_ROOT, "16_bl3_checkpoint.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch, "history": history,
    }, checkpoint_local)
    copy_to_drive(checkpoint_local, "16_bl3_checkpoint.pt")


def train_bl3(per_source, group_to_locations):
    model = AudioEncoder().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    rng = np.random.RandomState(RANDOM_SEED)

    start_epoch = 0
    history = []
    checkpoint = try_resume_checkpoint()
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        history = checkpoint["history"]

    for epoch in range(start_epoch, TOTAL_EPOCHS):
        epoch_start = time.time()
        model.train()
        epoch_contrastive, epoch_hsic = 0.0, 0.0
        for step in tqdm(range(STEPS_PER_EPOCH), desc=f"[BL3] epoch {epoch}/{TOTAL_EPOCHS}"):
            view1, view2, group_idx = sample_stratified_batch(
                per_source, group_to_locations, BATCH_SIZE, len(GROUPS), rng)
            mel1 = crop_to_mel_batch(view1)
            mel2 = crop_to_mel_batch(view2)

            emb1, proj1 = model(mel1)
            emb2, proj2 = model(mel2)

            contrastive = nt_xent_loss(proj1, proj2)
            group_onehot = F.one_hot(group_idx, num_classes=len(GROUPS)).float()
            group_onehot_doubled = torch.cat([group_onehot, group_onehot], dim=0)
            emb_doubled = torch.cat([emb1, emb2], dim=0)
            hsic = hsic_loss(emb_doubled, group_onehot_doubled)

            loss = contrastive + LAMBDA_HSIC * hsic

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_contrastive += contrastive.item()
            epoch_hsic += hsic.item()

        avg_contrastive = epoch_contrastive / STEPS_PER_EPOCH
        avg_hsic = epoch_hsic / STEPS_PER_EPOCH
        epoch_seconds = time.time() - epoch_start
        history.append({"epoch": epoch, "contrastive_loss": avg_contrastive,
                         "hsic_loss": avg_hsic, "epoch_seconds": epoch_seconds})
        print(f"[BL3] epoch {epoch}: contrastive={avg_contrastive:.4f}  hsic={avg_hsic:.6f}  "
              f"({epoch_seconds:.1f}s)")
        save_checkpoint(model, optimizer, epoch, history)

    model.eval()
    return model, history


def embed_full_pool(model, per_source):
    embeddings, groups = [], []
    for name, group, crops in per_source:
        for i in tqdm(range(crops.shape[0]), desc=f"[EMBED] {name}"):
            crop_embeddings = []
            for c in range(crops.shape[1]):
                wf = (crops[i, c].float() / 32767.0).to(DEVICE).unsqueeze(0)
                mel = crop_to_mel_batch(wf)
                with torch.no_grad():
                    emb, _ = model(mel)
                crop_embeddings.append(emb.squeeze(0).cpu().numpy())
            embeddings.append(np.mean(crop_embeddings, axis=0))
            groups.append(group)
    return np.stack(embeddings).astype(np.float32), np.array(groups)


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


def batched_topk_group_fractions(embeddings, groups, k=RETRIEVAL_K, batch_size=500):
    emb_t = torch.tensor(embeddings, device=DEVICE)
    emb_norm = torch.nn.functional.normalize(emb_t, dim=1)
    n = emb_norm.shape[0]
    group_idx = np.array([GROUPS.index(g) for g in groups])
    group_idx_t = torch.tensor(group_idx, device=DEVICE)

    fractions = np.zeros((n, len(GROUPS)), dtype=np.float32)
    for start in tqdm(range(0, n, batch_size), desc="[BL3] retrieval top-K"):
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


def main():
    ensure_project_dir()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 2: loading crop caches into the "
          f"shared, non-deleting cache (shared local cache -> PROJECT_DIR)...")
    per_source = download_pool_crops_shared()
    group_to_locations = build_group_indices(per_source)

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 3-4: training (or resuming) BL3...")
    model, history = train_bl3(per_source, group_to_locations)

    local_final_model = os.path.join(LOCAL_ROOT, "16_bl3_final_model.pt")
    torch.save(model.state_dict(), local_final_model)
    copy_to_drive(local_final_model, "16_bl3_final_model.pt")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 5: embedding the full pool...")
    embeddings, groups = embed_full_pool(model, per_source)
    fractions = batched_topk_group_fractions(embeddings, groups)
    bl3_metrics = compute_pool_level_metrics(fractions, groups)
    print("[BL3] Self-consistency:")
    for g, r in bl3_metrics["self_consistency"].items():
        print(f"    {g:16s} {r['observed']:.4f}  95% CI=({r['ci_95_low']:.4f}, {r['ci_95_high']:.4f})")
    print("[BL3] Exposure disparity (ratio to pool proportion; 1.0 = proportional):")
    for g, r in bl3_metrics["exposure_disparity"].items():
        print(f"    {g:16s} ratio={r['disparity_ratio']:.3f}")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 6: saving results and figure...")
    local_results_path = os.path.join(LOCAL_ROOT, "16_bl3_results.json")
    elapsed_seconds = time.time() - RUN_START_TIME
    output = {
        "program": "16_e5_bl3_hsic_adversarial.py", "baseline": "BL3_HSIC_adversarial",
        "retrieval_k": RETRIEVAL_K, "lambda_hsic": LAMBDA_HSIC,
        "batch_size": BATCH_SIZE, "total_epochs": TOTAL_EPOCHS,
        "n_pool_tracks": int(embeddings.shape[0]),
        "training_history_tail": history[-5:],
        "pool_level_metrics": bl3_metrics,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "elapsed_hms": time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds)),
        "completed_at": datetime.now().isoformat(t