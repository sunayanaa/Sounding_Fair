# =============================================================================
# Program      : 19_e6_refine_and_validate.py
# Version      : 1.2
# GPU Required : NO (small MLP-based refinement network; GPU used
#                 automatically if present)
# Description  : Experiment 6, Part 2 -- trains a lightweight residual
#                 refinement network on E1's embedding space using
#                 Program 18's augmented corpus, then runs the full
#                 validation suite against E4's vanilla-backbone results.
#
#                 BL3's approach (retrain the CNN from raw audio with an
#                 HSIC penalty) does NOT directly apply here: Program
#                 18's counterfactual embeddings are synthetic decoder
#                 output, with no corresponding audio to feed a CNN.
#                 Instead, a small residual MLP (output = input +
#                 MLP(input), 256->256) is trained directly on the
#                 embedding space, using two signals:
#                   - ALIGNMENT: each counterfactual is paired with its
#                     source track (Program 18's cf_source_indices) --
#                     since they share content and differ only in style
#                     by construction, pulling their refined embeddings
#                     together directly teaches "style should not
#                     determine retrieval neighbors."
#                   - HSIC: same dependence penalty as BL3/BL5 (Programs
#                     15/16), now applied to the refined embedding vs.
#                     tradition-group, using stratified sampling across
#                     the pool.
#                 The residual architecture (not a full replacement
#                 mapping) keeps the refined embedding close to E1's
#                 original unless the training signal actively pulls it
#                 away -- implicit "don't discard general semantic
#                 quality" regularization, without a separate penalty
#                 term.
#
#                 Validation reuses everything already built, rather
#                 than inventing new metrics:
#                   - The 3 E4 pool-level metrics (self-consistency,
#                     cross-tradition retrievability, exposure disparity)
#                     on refined embeddings, same bootstrap methodology
#                     as Programs 13/14/15/16 -- compared directly
#                     against E4's vanilla-backbone numbers.
#                   - E4's Intra-Group Fidelity (precision@K within each
#                     tradition's real fine-grained label), computed on
#                     REFINED fidelity-corpus embeddings (Program 13's
#                     cached fidelity embeddings, run through the trained
#                     refinement network) -- this is the direct "did we
#                     erase culture" check: it should stay roughly
#                     unchanged, not degrade, unlike BL5's catastrophic
#                     collapse.
#                   - A quick loudness linear probe (median-split,
#                     matching E3's exact convention) comparing refined
#                     vs. vanilla embeddings -- confirms the production
#                     confound actually shrank, not just that some other
#                     unrelated change happened. Loudness labels reused
#                     directly from Program 18's cached auxiliary targets
#                     (same pool, same order) -- no audio touched again.
#
#                 All inputs and outputs for this program live in
#                 PROJECT_DIR on Google Drive. There is no separate
#                 cache/mirror directory: every checkpoint, results
#                 JSON, .pt cache, and figure PNG produced anywhere in
#                 this pipeline resides under PROJECT_DIR, and this
#                 program reads its inputs (Program 18's augmented
#                 corpus + auxiliary targets, Program 13's fidelity
#                 embeddings, its own resume checkpoint) from that same
#                 location.
#
# INPUT:
#                  Google Drive (all under PROJECT_DIR):
#                    18_augmented_corpus.pt (Program 18 -- original +
#                      counterfactual embeddings, groups, cf_source_indices)
#                    18_aux_targets_cache.pt (Program 18 -- loudness
#                      labels for the original pool, same order)
#                    13_fidelity_embeddings.pt (Program 13 -- E1
#                      embeddings for the 4 labeled fidelity corpora)
#
# STEPS:
#                  Step 1  Mount Drive, ensure PROJECT_DIR exists
#                  Step 2  Load Program 18's augmented corpus + loudness
#                          labels, Program 13's fidelity embeddings
#                  Step 3  Train the refinement network: alignment loss
#                          (source/counterfactual pairs) + HSIC penalty
#                          (stratified batches); checkpoint every epoch
#                  Step 4  Apply the trained network to the full original
#                          pool + the 4 fidelity corpora
#                  Step 5  Recompute the 3 E4 pool-level metrics, E4's
#                          Intra-Group Fidelity, and the loudness probe,
#                          all compared directly against vanilla (E4)
#                  Step 6  Save results JSON + comparison figures to
#                          PROJECT_DIR
#
# OUTPUT FILES : under PROJECT_DIR
#                  19_refinement_checkpoint.pt  (resumable training state)
#                  19_refinement_final_model.pt (final network weights)
#                  19_validation_results.json
#                  fig_19_group_retrieval_heatmap.png
#                  fig_19_fidelity_vs_vanilla.png
#
# Hyperparameters : EMBEDDING_DIM=256, HIDDEN_DIM=256, LAMBDA_ALIGN=1.0,
#                    LAMBDA_HSIC=1.0, HSIC_BATCH_SIZE=500 (~100/group,
#                    stratified), ALIGN_BATCH_SIZE=256, N_EPOCHS=50,
#                    STEPS_PER_EPOCH=100, LEARNING_RATE=1e-3,
#                    RETRIEVAL_K=10, N_BOOTSTRAP_ITERATIONS=10000,
#                    RANDOM_SEED=42.
# Dependencies : torch, numpy, pandas, tqdm, matplotlib, scikit-learn
#
# Change Log   :
#   v1.0  2026-09-24  Initial version
#   v1.1  2026-09-24  First run showed align converge to ~0.000026 while
#                      hsic converged to ~0.0007 -- at LAMBDA_HSIC=1.0,
#                      hsic's raw magnitude contributed almost nothing to
#                      the gradient at any point in training, and the
#                      network satisfied alignment almost perfectly while
#                      self-consistency got WORSE than vanilla for every
#                      group (Hindustani -0.137, ArabAndalusian -0.100,
#                      Carnatic's exposure disparity ratio more than
#                      doubling to 2.17) -- fairness was never seriously
#                      contested during training. Fixed via dynamic loss
#                      normalization: each raw loss divided by a
#                      CONTINUOUSLY-UPDATING EMA (decay=0.99) of its own
#                      magnitude before weighting, so both terms
#                      contribute comparably to the gradient throughout
#                      training, not just at initialization (a fixed-
#                      point normalization, e.g. dividing by the first
#                      batch's value, would not have fixed this for the
#                      whole run). Both raw and normalized values, plus
#                      the EMAs themselves, now logged per epoch for
#                      direct monitoring. EMA state is saved/restored in
#                      the checkpoint so a resume doesn't reset the
#                      balance. Fidelity (the cultural-erasure check) was
#                      NOT the problem in v1.0 -- it held essentially
#                      unchanged -- so no change was made there.
#   v1.2  2026-09-24  v1.1's EMA-based normalization caused a runaway
#                      gradient: align_ema decayed exponentially toward
#                      zero as alignment converged (visible directly in
#                      the logs: 6.4e-19 -> 1.1e-20 across the tail
#                      epochs), and dividing by a near-zero, still-
#                      shrinking denominator amplified tiny residual
#                      gradient noise into an effectively unbounded
#                      gradient -- even though the forward-pass loss
#                      value looked like a converged ~0. Result: every
#                      query group collapsed to retrieving ~95-96%
#                      Dominant-pool tracks (including Dominant querying
#                      ITSELF at only 0.956, down from 0.987), and
#                      fidelity degraded across all 4 traditions for the
#                      first time (Jingju 0.642->0.487) -- signs of
#                      representational collapse, not a targeted fairness
#                      trade-off. Fixed by replacing the continuously-
#                      updating EMA with a FIXED reference scale: each
#                      loss trains with a neutral 1.0 divisor for the
#                      first NORM_WARMUP_STEPS steps while its typical
#                      early-training magnitude is recorded, then that
#                      average is FROZEN as a constant divisor for the
#                      rest of training -- a fixed denominator can never
#                      shrink toward zero no matter how well either loss
#                      converges, so it cannot reproduce this failure
#                      mode. NOTE: checkpoint fields changed again
#                      (align_ema/hsic_ema -> align_scale/hsic_scale plus
#                      warmup state) -- the v1.1 checkpoint must be
#                      deleted before rerunning.
# Running time:
# !pip install torch pandas tqdm matplotlib scikit-learn
# =============================================================================

!pip install torch pandas tqdm matplotlib scikit-learn

import os
import time
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from google.colab import drive
drive.mount('/content/drive')

RUN_START_TIME = time.time()
print(f"[{datetime.now().strftime('%H:%M:%S')}] Program 19_e6_refine_and_validate.py started.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] Using {DEVICE} (GPU not required; used automatically if present).")

# --- Configuration ---
# PROJECT_DIR is the single source of truth for the pipeline: every
# checkpoint, results JSON, figure PNG, .pt/.npy cache, and .json
# produced by any program in this project lives here, and every program
# reads its inputs from here too. No separate cache/mirror directory
# exists.
PROJECT_DIR = "/content/drive/MyDrive/paper/sounding-fair/"

EMBEDDING_DIM = 256
HIDDEN_DIM = 256
LAMBDA_ALIGN = 1.0  # applied to the NORMALIZED loss (dimensionless, ~O(1) after scale
                     # normalization) -- these now express RELATIVE priority between
                     # already-comparable-scale terms, not absolute scale-fixing weights
LAMBDA_HSIC = 1.0
NORM_WARMUP_STEPS = 100  # compute FIXED reference scales from the first 100 steps (one
                         # epoch's worth), then FREEZE them for the rest of training -- a
                         # continuously-updating EMA denominator (v1.1's approach) shrinks
                         # toward zero as a loss converges, and dividing by a
                         # near-zero denominator amplifies tiny residual gradient noise
                         # into a runaway gradient even when the forward-pass loss looks
                         # like ~0. A fixed denominator can never do that.
HSIC_BATCH_SIZE = 500  # ~100 per group, stratified
ALIGN_BATCH_SIZE = 256
N_EPOCHS = 50
STEPS_PER_EPOCH = 100
LEARNING_RATE = 1e-3
RETRIEVAL_K = 10
N_BOOTSTRAP_ITERATIONS = 10000
RANDOM_SEED = 42
N_FOLDS = 5  # for the loudness probe, matching E3/Program 11's methodology

GROUPS = ["Dominant", "Carnatic", "Hindustani", "Jingju", "ArabAndalusian"]

LOCAL_ROOT = "/content/local_data_e6_refine"
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


class RefinementNetwork(nn.Module):
    """Residual MLP: output = input + MLP(input). The residual form keeps
    the refined embedding close to E1's original unless training signal
    actively pulls it away -- implicit regularization against discarding
    general semantic quality, no separate penalty term needed."""
    def __init__(self, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x):
        return x + self.mlp(x)


def rbf_kernel_median_heuristic(X):
    pairwise_sq_dists = torch.cdist(X, X) ** 2
    nonzero = pairwise_sq_dists[pairwise_sq_dists > 0]
    sigma_sq = torch.median(nonzero) + 1e-8 if nonzero.numel() > 0 else torch.tensor(1.0, device=X.device)
    return torch.exp(-pairwise_sq_dists / (2 * sigma_sq))


def hsic_categorical(Z, G_onehot):
    """HSIC between a continuous embedding Z and a categorical one-hot
    group label G -- same formulation as BL3/BL5 (Programs 15/16):
    RBF kernel on Z, linear kernel on G_onehot."""
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


def try_resume_checkpoint():
    checkpoint_local = os.path.join(LOCAL_ROOT, "19_refinement_checkpoint.pt")
    if copy_from_drive("19_refinement_checkpoint.pt", checkpoint_local, skip_if_exists=True):
        checkpoint = torch.load(checkpoint_local, map_location=DEVICE, weights_only=False)
        print(f"[RESUME] Found checkpoint at epoch {checkpoint['epoch']}. Resuming.")
        return checkpoint
    print("[RESUME] No checkpoint found. Starting fresh.")
    return None


def save_checkpoint(model, optimizer, epoch, history, align_scale, hsic_scale,
                     warmup_align_vals, warmup_hsic_vals, global_step):
    checkpoint_local = os.path.join(LOCAL_ROOT, "19_refinement_checkpoint.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch, "history": history,
        "align_scale": align_scale, "hsic_scale": hsic_scale,
        "warmup_align_vals": warmup_align_vals, "warmup_hsic_vals": warmup_hsic_vals,
        "global_step": global_step,
    }, checkpoint_local)
    copy_to_drive(checkpoint_local, "19_refinement_checkpoint.pt")


def train_refinement(all_embeddings, all_groups, cf_source_indices, n_original):
    emb_t = torch.tensor(all_embeddings, dtype=torch.float32, device=DEVICE)
    group_idx_all = np.array([GROUPS.index(g) for g in all_groups])
    group_to_indices_all = {g: np.where(group_idx_all == GROUPS.index(g))[0] for g in GROUPS}
    for g, idx in group_to_indices_all.items():
        print(f"[REFINE] Group '{g}': {len(idx)} tracks (original+counterfactual) available.")

    n_cf = len(cf_source_indices)
    cf_global_indices = np.arange(n_original, n_original + n_cf)
    print(f"[REFINE] {n_cf} (source, counterfactual) alignment pairs available.")

    model = RefinementNetwork().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    rng = np.random.RandomState(RANDOM_SEED)

    start_epoch = 0
    history = []
    align_scale, hsic_scale = None, None  # FIXED reference scales -- set once during
                                           # warmup, then frozen for the rest of training
    warmup_align_vals, warmup_hsic_vals = [], []
    global_step = 0
    checkpoint = try_resume_checkpoint()
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        history = checkpoint["history"]
        align_scale = checkpoint.get("align_scale")
        hsic_scale = checkpoint.get("hsic_scale")
        warmup_align_vals = checkpoint.get("warmup_align_vals", [])
        warmup_hsic_vals = checkpoint.get("warmup_hsic_vals", [])
        global_step = checkpoint.get("global_step", 0)

    for epoch in range(start_epoch, N_EPOCHS):
        epoch_losses = {"align": 0.0, "hsic": 0.0, "align_normalized": 0.0, "hsic_normalized": 0.0}
        for step in range(STEPS_PER_EPOCH):
            hsic_idx = stratified_batch_indices(group_to_indices_all, HSIC_BATCH_SIZE, len(GROUPS), rng)
            hsic_batch = emb_t[hsic_idx]
            hsic_groups = torch.tensor(group_idx_all[hsic_idx], device=DEVICE)
            refined_hsic = model(hsic_batch)
            group_onehot = F.one_hot(hsic_groups, num_classes=len(GROUPS)).float()
            l_hsic = hsic_categorical(refined_hsic, group_onehot)

            align_sample = rng.choice(n_cf, size=min(ALIGN_BATCH_SIZE, n_cf), replace=False)
            orig_idx = cf_source_indices[align_sample]
            cf_idx = cf_global_indices[align_sample]
            refined_orig = model(emb_t[orig_idx])
            refined_cf = model(emb_t[cf_idx])
            l_align = 1 - F.cosine_similarity(refined_orig, refined_cf, dim=1).mean()

            # Fixed-scale loss normalization: during the first
            # NORM_WARMUP_STEPS steps, train with a neutral scale of 1.0
            # while collecting each loss's typical early-training
            # magnitude; once the warmup window closes, FREEZE the
            # average observed magnitude as a constant divisor for the
            # rest of training. Unlike a continuously-updating EMA, this
            # denominator can never shrink toward zero as a loss
            # converges, so it can't amplify residual gradient noise
            # into a runaway gradient the way v1.1's EMA-based version did.
            with torch.no_grad():
                align_val, hsic_val = l_align.item(), l_hsic.item()
                if align_scale is None:
                    warmup_align_vals.append(align_val)
                    warmup_hsic_vals.append(hsic_val)
                    global_step += 1
                    align_scale_current, hsic_scale_current = 1.0, 1.0
                    if global_step >= NORM_WARMUP_STEPS:
                        align_scale = float(np.mean(warmup_align_vals)) + 1e-8
                        hsic_scale = float(np.mean(warmup_hsic_vals)) + 1e-8
                        print(f"[REFINE] Warmup complete at step {global_step}: FIXED "
                              f"align_scale={align_scale:.6f}, hsic_scale={hsic_scale:.6f}")
                else:
                    align_scale_current, hsic_scale_current = align_scale, hsic_scale
                    global_step += 1

            l_align_norm = l_align / align_scale_current
            l_hsic_norm = l_hsic / hsic_scale_current

            loss = LAMBDA_ALIGN * l_align_norm + LAMBDA_HSIC * l_hsic_norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses["align"] += align_val
            epoch_losses["hsic"] += hsic_val
            epoch_losses["align_normalized"] += l_align_norm.item()
            epoch_losses["hsic_normalized"] += l_hsic_norm.item()

        avg_losses = {k: v / STEPS_PER_EPOCH for k, v in epoch_losses.items()}
        avg_losses["epoch"] = epoch
        avg_losses["align_scale"] = align_scale
        avg_losses["hsic_scale"] = hsic_scale
        history.append(avg_losses)
        if epoch % 10 == 0 or epoch == N_EPOCHS - 1:
            print(f"[REFINE] epoch {epoch}/{N_EPOCHS}  "
                  f"align={avg_losses['align']:.6f} (norm={avg_losses['align_normalized']:.3f})  "
                  f"hsic={avg_losses['hsic']:.6f} (norm={avg_losses['hsic_normalized']:.3f})")
        save_checkpoint(model, optimizer, epoch, history, align_scale, hsic_scale,
                         warmup_align_vals, warmup_hsic_vals, global_step)

    model.eval()
    return model, history


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
    for start in tqdm(range(0, n, batch_size), desc="[VALIDATE] pool top-K"):
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


def compute_intragroup_fidelity(fidelity_data, k=RETRIEVAL_K):
    results = {}
    for tradition, d in fidelity_data.items():
        embeddings = d["embeddings"]
        labels = d["labels"]
        n = embeddings.shape[0]
        eff_k = min(k, n - 1)
        emb_t = torch.tensor(embeddings, device=DEVICE)
        emb_norm = torch.nn.functional.normalize(emb_t, dim=1)
        sims = emb_norm @ emb_norm.T
        sims[torch.arange(n), torch.arange(n)] = -float("inf")
        topk_idx = torch.topk(sims, k=eff_k, dim=1).indices.cpu().numpy()

        per_query_precision = np.zeros(n, dtype=np.float32)
        for i in range(n):
            neighbor_labels = labels[topk_idx[i]]
            per_query_precision[i] = np.mean(neighbor_labels == labels[i])

        n_classes = len(set(labels))
        chance_level = 1.0 / n_classes if n_classes > 0 else None
        result = bootstrap_mean_ci(per_query_precision)
        result["n_tracks"] = int(n)
        result["n_classes"] = int(n_classes)
        result["chance_level"] = chance_level
        results[tradition] = result
        print(f"[FIDELITY] {tradition}: precision@{eff_k}={result['observed']:.4f} "
              f"(chance={chance_level:.4f})")
    return results


def loudness_probe_accuracy(embeddings, loudness_values):
    """Same median-split + 5-fold stratified CV methodology as
    Program 11/E3 -- a lower accuracy here (closer to chance) than the
    vanilla-embedding probe means the production confound genuinely
    shrank, not just that some other unrelated change happened."""
    median_loudness = np.median(loudness_values)
    y = np.where(loudness_values >= median_loudness, "Loud", "Quiet")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    oof_preds = np.empty(len(y), dtype=object)
    for train_idx, test_idx in skf.split(embeddings, y):
        scaler = StandardScaler().fit(embeddings[train_idx])
        X_train_s = scaler.transform(embeddings[train_idx])
        X_test_s = scaler.transform(embeddings[test_idx])
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_train_s, y[train_idx])
        oof_preds[test_idx] = clf.predict(X_test_s)
    return accuracy_score(y, oof_preds)


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

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 2: loading Program 18's augmented "
          f"corpus and Program 13's fidelity embeddings...")
    corpus_local = os.path.join(LOCAL_ROOT, "18_augmented_corpus.pt")
    if not copy_from_drive("18_augmented_corpus.pt", corpus_local, skip_if_exists=True):
        raise RuntimeError("Could not fetch 18_augmented_corpus.pt -- run Program 18 first.")
    corpus = torch.load(corpus_local, map_location="cpu", weights_only=False)
    all_embeddings, all_groups = corpus["embeddings"], corpus["groups"]
    is_counterfactual, cf_source_indices = corpus["is_counterfactual"], corpus["cf_source_indices"]
    n_original = int((~is_counterfactual).sum())
    print(f"[CORPUS] {n_original} original + {len(cf_source_indices)} counterfactual tracks.")

    aux_local = os.path.join(LOCAL_ROOT, "18_aux_targets_cache.pt")
    if not copy_from_drive("18_aux_targets_cache.pt", aux_local, skip_if_exists=True):
        raise RuntimeError("Could not fetch 18_aux_targets_cache.pt -- run Program 18 first.")
    aux_data = torch.load(aux_local, map_location="cpu", weights_only=False)
    loudness_values = aux_data["style_target"].reshape(-1)  # same order as the original pool

    fidelity_local = os.path.join(LOCAL_ROOT, "13_fidelity_embeddings.pt")
    if not copy_from_drive("13_fidelity_embeddings.pt", fidelity_local, skip_if_exists=True):
        raise RuntimeError("Could not fetch 13_fidelity_embeddings.pt -- run Program 13 first.")
    fidelity_data_vanilla = torch.load(fidelity_local, map_location="cpu", weights_only=False)

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 3: training the refinement network...")
    model, history = train_refinement(all_embeddings, all_groups, cf_source_indices, n_original)

    local_final_model = os.path.join(LOCAL_ROOT, "19_refinement_final_model.pt")
    torch.save(model.state_dict(), local_final_model)
    copy_to_drive(local_final_model, "19_refinement_final_model.pt")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 4: applying the refinement network "
          f"to the original pool and the 4 fidelity corpora...")
    original_embeddings = all_embeddings[:n_original]
    original_groups = all_groups[:n_original]
    with torch.no_grad():
        refined_pool = model(torch.tensor(original_embeddings, dtype=torch.float32, device=DEVICE)) \
            .cpu().numpy()

    refined_fidelity_data = {}
    for tradition, d in fidelity_data_vanilla.items():
        with torch.no_grad():
            refined_emb = model(torch.tensor(d["embeddings"], dtype=torch.float32, device=DEVICE)) \
                .cpu().numpy()
        refined_fidelity_data[tradition] = {"embeddings": refined_emb, "labels": d["labels"]}

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 5: validation -- pool-level metrics, "
          f"fidelity, loudness probe...")
    fractions_refined = batched_topk_group_fractions(refined_pool, original_groups)
    pool_metrics_refined = compute_pool_level_metrics(fractions_refined, original_groups)
    print("[VALIDATE] Self-consistency (refined):")
    for g, r in pool_metrics_refined["self_consistency"].items():
        print(f"    {g:16s} {r['observed']:.4f}")
    print("[VALIDATE] Exposure disparity ratio (refined):")
    for g, r in pool_metrics_refined["exposure_disparity"].items():
        print(f"    {g:16s} ratio={r['disparity_ratio']:.3f}")

    fidelity_refined = compute_intragroup_fidelity(refined_fidelity_data)
    fidelity_vanilla = compute_intragroup_fidelity(fidelity_data_vanilla)

    print(f"\n[LOUDNESS PROBE] Fitting on vanilla vs. refined pool embeddings...")
    loudness_acc_vanilla = loudness_probe_accuracy(original_embeddings, loudness_values)
    loudness_acc_refined = loudness_probe_accuracy(refined_pool, loudness_values)
    print(f"    Vanilla accuracy:  {loudness_acc_vanilla:.4f}")
    print(f"    Refined accuracy:  {loudness_acc_refined:.4f}  "
          f"(lower = confound reduced; chance = 0.500)")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 6: saving results and figures...")
    local_results_path = os.path.join(LOCAL_ROOT, "19_validation_results.json")
    elapsed_seconds = time.time() - RUN_START_TIME
    output = {
        "program": "19_e6_refine_and_validate.py",
        "n_original_tracks": n_original, "n_counterfactuals": len(cf_source_indices),
        "training_history_tail": history[-5:],
        "pool_level_metrics_refined": pool_metrics_refined,
        "fidelity_refined": fidelity_refined,
        "fidelity_vanilla_recomputed": fidelity_vanilla,
        "loudness_probe": {
            "vanilla_accuracy": loudness_acc_vanilla, "refined_accuracy": loudness_acc_refined,
            "chance": 0.5,
        },
        "elapsed_seconds": round(elapsed_seconds, 1),
        "elapsed_hms": time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds)),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(local_results_path, "w") as f:
        json.dump(output, f, indent=2)
    copy_to_drive(local_results_path, "19_validation_results.json")

    local_fig1 = os.path.join(LOCAL_ROOT, "fig_19_group_retrieval_heatmap.png")
    save_heatmap(pool_metrics_refined, f"E6 (refined): Cross-tradition retrievability (K={RETRIEVAL_K})",
                 local_fig1)
    copy_to_drive(local_fig1, "fig_19_group_retrieval_heatmap.png")

    traditions = list(fidelity_refined.keys())
    x = np.arange(len(traditions))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, [fidelity_vanilla[t]["observed"] for t in traditions], width, label="Vanilla (E4)")
    ax.bar(x + width / 2, [fidelity_refined[t]["observed"] for t in traditions], width, label="Refined (E6)")
    ax.set_xticks(x); ax.set_xticklabels(traditions)
    ax.set_ylabel("Precision@10"); ax.set_title("E6: Intra-group fidelity, vanilla vs. refined")
    ax.legend()
    fig.tight_layout()
    local_fig2 = os.path.join(LOCAL_ROOT, "fig_19_fidelity_vs_vanilla.png")
    fig.savefig(local_fig2, dpi=300)
    plt.close(fig)
    copy_to_drive(local_fig2, "fig_19_fidelity_vs_vanilla.png")

    return output


if __name__ == "__main__":
    summary = main()
    elapsed = time.time() - RUN_START_TIME
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Program 19_e6_refine_and_validate.py finished.")
    print(f"[TOTAL RUNNING TIME] {time.strftime('%H:%M:%S', time.gmtime(elapsed))} ({elapsed:.1f}s)")