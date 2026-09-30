# =============================================================================
# Program      : 15_e5_bl5_hsic_no_dvae.py
# Version      : 1.0
# GPU Required : NO (small MLP; GPU used automatically if present, mainly
#                 helps the retrieval-metric step, not training)
# Description  : Experiment 5, Part 2 -- BL5, the "closest competing
#                 hypothesis" baseline: an HSIC dependence penalty applied
#                 directly to the raw, UNLEARNED handcrafted feature
#                 representation phi(t) (MFCC + chroma + spectral
#                 contrast + Tonnetz, the same standard variant computed
#                 in Program 09/E2), with NO contrastive pretraining, NO
#                 dVAE, and NO content/style disentanglement. This tests
#                 whether the trained encoder's contrastive learning step
#                 contributes anything beyond just penalizing dependence
#                 on off-the-shelf features.
#
#                 Architecture: a small fair-autoencoder -- encoder maps
#                 standardized phi(t) (76-dim) to a 256-dim embedding
#                 (matching E1's output dim for a fair downstream
#                 retrieval comparison), decoder reconstructs phi(t) from
#                 that embedding. Loss = reconstruction MSE + lambda *
#                 HSIC(embedding, group_one_hot). HSIC uses an RBF kernel
#                 on the embedding (median-heuristic bandwidth, computed
#                 fresh per batch) and a linear kernel on the one-hot
#                 group labels (the natural choice for categorical
#                 labels -- equals 1 for same-group pairs, 0 otherwise).
#
#                 Batches are STRATIFIED (equal count per group, sampled
#                 with replacement for the smaller traditions) rather
#                 than drawn randomly from the natural 94:6-imbalanced
#                 pool -- a random batch would be almost entirely
#                 Dominant-pool examples, giving the HSIC penalty little
#                 signal about the 4 minority traditions. This same
#                 stratified-sampling choice will be used for BL3's
#                 retraining too, for a fair comparison between the two
#                 HSIC-penalized baselines.
#
#                 phi(t) is computed fresh here (Program 09/E2 computed
#                 the same standard variant but never cached the raw
#                 feature matrix, only the aggregate probe results), and
#                 IS cached this time so any follow-up doesn't repeat the
#                 ~27-minute extraction.
#
# INPUT:
#                  Google Drive:
#                    08_crops_fma.pt, 08_crops_carnatic.pt,
#                    08_crops_hindustani.pt, 08_crops_jingju_base.pt,
#                    08_crops_jacrc.pt, 08_crops_arab_andalusian.pt,
#                    08_crops_jamendo.pt
#
# STEPS:
#                  Step 1  Ensure PROJECT_DIR exists
#                  Step 2  Compute standard phi(t) (MFCC+chroma+contrast+
#                          Tonnetz, crop index 0) for the full pool,
#                          tagged by group; cache to Google Drive
#                  Step 3  Standardize phi(t); train the fair autoencoder
#                          (stratified batches, reconstruction + HSIC
#                          loss) for a fixed number of steps
#                  Step 4  Encode the full pool through the trained
#                          encoder to get BL5 embeddings
#                  Step 5  Recompute the 3 E4-style pool-level metrics
#                          (self-consistency, cross-tradition
#                          retrievability, exposure disparity) with
#                          bootstrap CIs, same methodology as Programs
#                          13/14
#                  Step 6  Save results JSON + heatmap figure, copy to
#                          Google Drive
#
# OUTPUT FILES : under PROJECT_DIR
#                  15_bl5_phi_cache.pt        {phi, groups}
#                  15_bl5_results.json
#                  fig_15_bl5_vs_vanilla_heatmap.png
#
# Hyperparameters : PHI_DIM=76 (13 MFCC + 12 chroma + 7 contrast + 6
#                    Tonnetz, mean+std pooled = 2*(13+12+7+6)), N_MFCC=13,
#                    EMBEDDING_DIM=256, HIDDEN_DIM=128, BATCH_SIZE=500
#                    (100/group, stratified), N_EPOCHS=100,
#                    STEPS_PER_EPOCH=50, LEARNING_RATE=1e-3,
#                    LAMBDA_HSIC=1.0 (a reasonable default for
#                    standardized inputs, not extensively tuned --
#                    baselines don't need the same tuning rigor as the
#                    main method), RETRIEVAL_K=10,
#                    N_BOOTSTRAP_ITERATIONS=10000, RANDOM_SEED=42.
# Dependencies : torch, torchaudio, librosa, numpy, pandas, tqdm,
#                 matplotlib
#
# Change Log   :
#   v1.0  2026-09-23  Initial version
# Running time: 28 mins
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
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

RUN_START_TIME = time.time()
print(f"[{datetime.now().strftime('%H:%M:%S')}] Program 15_e5_bl5_hsic_no_dvae.py started.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] Using {DEVICE} (GPU not required; used automatically if present).")

# --- Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/sounding-fair/"

SAMPLE_RATE = 22050
N_MFCC = 13
EMBEDDING_DIM = 256
HIDDEN_DIM = 128
BATCH_SIZE = 500  # 100 per group, stratified
N_EPOCHS = 100
STEPS_PER_EPOCH = 50
LEARNING_RATE = 1e-3
LAMBDA_HSIC = 1.0
RETRIEVAL_K = 10
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

LOCAL_ROOT = "/content/local_data_e5_bl5"
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


def compute_chroma_phi(waveform_np, sr):
    """Same construction as Program 09's standard phi(t): MFCC + chroma +
    spectral contrast + Tonnetz, mean+std pooled. 76-dim."""
    mfcc = librosa.feature.mfcc(y=waveform_np, sr=sr, n_mfcc=N_MFCC)
    chroma = librosa.feature.chroma_stft(y=waveform_np, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=waveform_np, sr=sr)
    tonnetz = librosa.feature.tonnetz(chroma=chroma, sr=sr)

    def pool(feat):
        return np.concatenate([feat.mean(axis=1), feat.std(axis=1)])

    return np.concatenate([pool(mfcc), pool(chroma), pool(contrast), pool(tonnetz)])


def compute_pool_phi(cache_local, cache_remote):
    if copy_from_drive(cache_remote, cache_local, skip_if_exists=True):
        cached = torch.load(cache_local, map_location="cpu", weights_only=False)
        print(f"[CACHE] Reused cached phi(t): {cached['phi'].shape[0]} tracks.")
        return cached["phi"], cached["groups"]

    phi_list, groups = [], []
    for name in POOL_SOURCES:
        crops_local = os.path.join(LOCAL_ROOT, f"crops_{name}.pt")
        if not copy_from_drive(f"08_crops_{name}.pt", crops_local, skip_if_exists=True):
            print(f"[WARNING] Could not fetch 08_crops_{name}.pt -- skipping.")
            continue
        data = torch.load(crops_local, map_location="cpu", weights_only=False)
        crops = data["crops"]  # [N, 3, SEGMENT_LENGTH] int16
        group = SOURCE_TO_GROUP[name]
        for i in tqdm(range(crops.shape[0]), desc=f"[PHI] {name}"):
            wf = (crops[i, 0].float() / 32767.0).numpy()
            phi_list.append(compute_chroma_phi(wf, SAMPLE_RATE))
            groups.append(group)
        del data, crops
        os.remove(crops_local)
        print(f"[SOURCE] {name}: {len(groups)} tracks processed so far (cumulative).")

    phi = np.stack(phi_list).astype(np.float32)
    groups = np.array(groups)
    torch.save({"phi": phi, "groups": groups}, cache_local)
    copy_to_drive(cache_local, cache_remote)
    return phi, groups


class FairAutoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon


def rbf_kernel_median_heuristic(X):
    """X: [B, D]. Pairwise RBF kernel with bandwidth set by the median
    heuristic (median of the batch's pairwise squared distances),
    recomputed fresh each call rather than fixed, since a fixed bandwidth
    chosen once could become stale as the embedding evolves during
    training."""
    pairwise_sq_dists = torch.cdist(X, X) ** 2
    nonzero = pairwise_sq_dists[pairwise_sq_dists > 0]
    sigma_sq = torch.median(nonzero) + 1e-8 if nonzero.numel() > 0 else torch.tensor(1.0, device=X.device)
    return torch.exp(-pairwise_sq_dists / (2 * sigma_sq))


def hsic_loss(Z, G_onehot):
    """Biased HSIC estimator: RBF kernel on the embedding Z (median
    heuristic), linear kernel on the one-hot group labels G_onehot (the
    natural choice for categorical labels -- 1 for same-group pairs, 0
    otherwise)."""
    B = Z.shape[0]
    K_Z = rbf_kernel_median_heuristic(Z)
    K_G = G_onehot @ G_onehot.T
    H = torch.eye(B, device=Z.device) - torch.ones(B, B, device=Z.device) / B
    return torch.trace(K_Z @ H @ K_G @ H) / ((B - 1) ** 2)


def stratified_batch_indices(group_to_indices, batch_size, n_groups, rng):
    per_group = batch_size // n_groups
    batch_idx = []
    for g in GROUPS:
        idx_pool = group_to_indices[g]
        replace = len(idx_pool) < per_group
        batch_idx.extend(rng.choice(idx_pool, size=per_group, replace=replace))
    return np.array(batch_idx)


def train_fair_autoencoder(phi, groups):
    n, input_dim = phi.shape
    phi_mean = phi.mean(axis=0, keepdims=True)
    phi_std = phi.std(axis=0, keepdims=True) + 1e-8
    phi_standardized = (phi - phi_mean) / phi_std

    group_idx = np.array([GROUPS.index(g) for g in groups])
    group_to_indices = {g: np.where(group_idx == GROUPS.index(g))[0] for g in GROUPS}
    for g, idx in group_to_indices.items():
        print(f"[BL5] Group '{g}': {len(idx)} tracks available for stratified sampling.")

    phi_t = torch.tensor(phi_standardized, dtype=torch.float32, device=DEVICE)
    group_onehot_full = F.one_hot(torch.tensor(group_idx), num_classes=len(GROUPS)).float().to(DEVICE)

    model = FairAutoencoder(input_dim=input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    rng = np.random.RandomState(RANDOM_SEED)

    history = []
    for epoch in range(N_EPOCHS):
        epoch_recon, epoch_hsic = 0.0, 0.0
        for step in range(STEPS_PER_EPOCH):
            batch_idx = stratified_batch_indices(group_to_indices, BATCH_SIZE, len(GROUPS), rng)
            x_batch = phi_t[batch_idx]
            g_batch = group_onehot_full[batch_idx]

            z, x_recon = model(x_batch)
            recon_loss = F.mse_loss(x_recon, x_batch)
            hsic = hsic_loss(z, g_batch)
            loss = recon_loss + LAMBDA_HSIC * hsic

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_recon += recon_loss.item()
            epoch_hsic += hsic.item()

        avg_recon = epoch_recon / STEPS_PER_EPOCH
        avg_hsic = epoch_hsic / STEPS_PER_EPOCH
        history.append({"epoch": epoch, "recon_loss": avg_recon, "hsic_loss": avg_hsic})
        if epoch % 10 == 0 or epoch == N_EPOCHS - 1:
            print(f"[BL5] epoch {epoch}/{N_EPOCHS}  recon_loss={avg_recon:.4f}  hsic_loss={avg_hsic:.6f}")

    model.eval()
    with torch.no_grad():
        full_embeddings = model.encoder(phi_t).cpu().numpy()
    return full_embeddings, history


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
    for start in tqdm(range(0, n, batch_size), desc="[BL5] retrieval top-K"):
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

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 2: computing/reusing pool phi(t)...")
    phi_cache_local = os.path.join(LOCAL_ROOT, "15_bl5_phi_cache.pt")
    phi, groups = compute_pool_phi(phi_cache_local, "15_bl5_phi_cache.pt")
    print(f"[PHI] Shape: {phi.shape}. Class distribution: {pd.Series(groups).value_counts().to_dict()}")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 3: training the fair autoencoder "
          f"(stratified batches, reconstruction + HSIC loss)...")
    bl5_embeddings, training_history = train_fair_autoencoder(phi, groups)

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 5: pool-level retrieval metrics "
          f"(K={RETRIEVAL_K})...")
    fractions = batched_topk_group_fractions(bl5_embeddings, groups)
    bl5_metrics = compute_pool_level_metrics(fractions, groups)
    print("[BL5] Self-consistency:")
    for g, r in bl5_metrics["self_consistency"].items():
        print(f"    {g:16s} {r['observed']:.4f}  95% CI=({r['ci_95_low']:.4f}, {r['ci_95_high']:.4f})")
    print("[BL5] Exposure disparity (ratio to pool proportion; 1.0 = proportional):")
    for g, r in bl5_metrics["exposure_disparity"].items():
        print(f"    {g:16s} ratio={r['disparity_ratio']:.3f}")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 6: saving results and figure...")
    local_results_path = os.path.join(LOCAL_ROOT, "15_bl5_results.json")
    elapsed_seconds = time.time() - RUN_START_TIME
    output = {
        "program": "15_e5_bl5_hsic_no_dvae.py", "baseline": "BL5_HSIC_no_dVAE",
        "retrieval_k": RETRIEVAL_K, "lambda_hsic": LAMBDA_HSIC,
        "n_pool_tracks": int(phi.shape[0]), "phi_dim": int(phi.shape[1]),
        "training_history_tail": training_history[-5:],
        "pool_level_metrics": bl5_metrics,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "elapsed_hms": time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds)),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(local_results_path, "w") as f:
        json.dump(output, f, indent=2)
    copy_to_drive(local_results_path, "15_bl5_results.json")

    local_fig = os.path.join(LOCAL_ROOT, "fig_15_bl5_vs_vanilla_heatmap.png")
    save_heatmap(bl5_metrics, f"E5 BL5 (HSIC, no dVAE): Cross-tradition retrievability (K={RETRIEVAL_K})", local_fig)
    copy_to_drive(local_fig, "fig_15_bl5_vs_vanilla_heatmap.png")

    return output


if __name__ == "__main__":
    summary = main()
    elapsed = time.time() - RUN_START_TIME
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Program 15_e5_bl5_hsic_no_dvae.py finished.")
    print(f"[TOTAL RUNNING TIME] {time.strftime('%H:%M:%S', time.gmtime(elapsed))} ({elapsed:.1f}s)")