# =============================================================================
# Program      : 20_e8_e12_lambda_sweep.py
# Version      : 1.0
# GPU Required : NO (small MLP-based refinement network; GPU used
#                 automatically if present)
# Description  : Experiments E8 (Augmentation vs. Penalty Attribution)
#                 and E12 (Penalty Weight Sweep), combined into one sweep
#                 over lambda_HSIC. Reuses Program 19's now-correct
#                 training code (v1.2's fixed-scale normalization, which
#                 resolved the gradient-explosion bug) unchanged --
#                 nothing architectural is new here, only repeated
#                 execution at different lambda_HSIC values.
#
#                 Directly closes the gap flagged during the discussion
#                 that led to stopping E6: two configurations were tried
#                 (lambda=1.0 both times, once with EMA normalization
#                 that collapsed, once with fixed-scale normalization
#                 that held fidelity but still lost the fairness
#                 objective) but lambda_HSIC itself was never
#                 systematically swept between "too weak" (v1.0's
#                 unnormalized sum, HSIC contributing ~0.0007 out of a
#                 total loss around 1.0) and "too strong" (v1.2's tail
#                 epochs, hsic_normalized ~200x larger than
#                 align_normalized). This either finds a working middle
#                 ground, or turns the "across three configurations we
#                 consistently observed X" finding into "across a full
#                 sweep from 0 to 10x, we consistently observed X" --
#                 a substantially harder claim to challenge either way.
#
#                 lambda_HSIC=0 is one point on the sweep, not a special
#                 case: this gives E8's "alignment only, no fairness
#                 penalty at all" ablation for free -- does counterfactual-
#                 based alignment alone move cross-tradition retrieval at
#                 all, independent of HSIC entirely?
#
#                 All inputs and outputs for this program live in
#                 PROJECT_DIR on Google Drive. There is no separate
#                 cache/mirror directory: every checkpoint, results
#                 JSON, and figure PNG produced anywhere in this
#                 pipeline resides under PROJECT_DIR, and this program
#                 reads its inputs (Program 18's augmented corpus +
#                 auxiliary targets, Program 13's fidelity embeddings)
#                 from that same location.
#
# INPUT:
#                  Google Drive (all under PROJECT_DIR):
#                    18_augmented_corpus.pt, 18_aux_targets_cache.pt,
#                    13_fidelity_embeddings.pt (identical to Program 19)
#
# STEPS:
#                  Step 1  Mount Drive, ensure PROJECT_DIR exists
#                  Step 2  Load the augmented corpus, loudness labels,
#                          and fidelity embeddings ONCE (not re-loaded
#                          per sweep point)
#                  Step 3  For each lambda_HSIC in the sweep: train a
#                          FRESH refinement network (own checkpoint,
#                          independent of other sweep points) using
#                          Program 19's exact training procedure, then
#                          run the full validation suite
#                  Step 4  Save per-point results + a sweep summary
#                          figure (key metrics vs. lambda_HSIC) to
#                          PROJECT_DIR
#
# OUTPUT FILES : under PROJECT_DIR
#                  20_sweep_checkpoint_lambda<value>.pt  (one per sweep
#                                                          point, resumable)
#                  20_sweep_results.json
#                  fig_20_sweep_self_consistency.png
#                  fig_20_sweep_fidelity_and_loudness.png
#
# Hyperparameters : LAMBDA_HSIC_SWEEP=[0.0, 0.1, 0.3, 1.0, 3.0, 10.0],
#                    all else identical to Program 19 v1.2 (EMBEDDING_DIM=256,
#                    HIDDEN_DIM=256, LAMBDA_ALIGN=1.0, NORM_WARMUP_STEPS=100,
#                    HSIC_BATCH_SIZE=500, ALIGN_BATCH_SIZE=256, N_EPOCHS=50,
#                    STEPS_PER_EPOCH=100, LEARNING_RATE=1e-3, RETRIEVAL_K=10,
#                    N_BOOTSTRAP_ITERATIONS=10000, RANDOM_SEED=42).
# Dependencies : torch, numpy, pandas, tqdm, matplotlib, scikit-learn
#
# Change Log   :
#   v1.0  2026-09-25  Initial version
# Running time:
# !pip install torch pandas tqdm matplotlib scikit-learn
# =============================================================================

!pip install -q torch pandas tqdm matplotlib scikit-learn

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
print(f"[{datetime.now().strftime('%H:%M:%S')}] Program 20_e8_e12_lambda_sweep.py started.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] Using {DEVICE} (GPU not required; used automatically if present).")

# --- Configuration ---
# PROJECT_DIR is the single source of truth for the pipeline: every
# checkpoint, results JSON, figure PNG, and .pt cache produced by any
# program in this project lives here, and every program reads its inputs
# from here too. No separate cache/mirror directory exists.
PROJECT_DIR = "/content/drive/MyDrive/paper/sounding-fair/"

EMBEDDING_DIM = 256
HIDDEN_DIM = 256
LAMBDA_ALIGN = 1.0
LAMBDA_HSIC_SWEEP = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]
NORM_WARMUP_STEPS = 100
HSIC_BATCH_SIZE = 500
ALIGN_BATCH_SIZE = 256
N_EPOCHS = 50
STEPS_PER_EPOCH = 100
LEARNING_RATE = 1e-3
RETRIEVAL_K = 10
N_BOOTSTRAP_ITERATIONS = 10000
RANDOM_SEED = 42
N_FOLDS = 5

GROUPS = ["Dominant", "Carnatic", "Hindustani", "Jingju", "ArabAndalusian"]
TRADITION_GROUPS = ["Carnatic", "Hindustani", "Jingju", "ArabAndalusian"]

LOCAL_ROOT = "/content/local_data_e8_e12_sweep"
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


def try_resume_checkpoint(lambda_hsic):
    checkpoint_remote = f"20_sweep_checkpoint_lambda{lambda_hsic}.pt"
    checkpoint_local = os.path.join(LOCAL_ROOT, checkpoint_remote)
    if copy_from_drive(checkpoint_remote, checkpoint_local, skip_if_exists=True):
        checkpoint = torch.load(checkpoint_local, map_location=DEVICE, weights_only=False)
        print(f"[RESUME] lambda={lambda_hsic}: found checkpoint at epoch {checkpoint['epoch']}. Resuming.")
        return checkpoint
    print(f"[RESUME] lambda={lambda_hsic}: no checkpoint found. Starting fresh.")
    return None


def save_checkpoint(lambda_hsic, model, optimizer, epoch, history, align_scale, hsic_scale,
                     warmup_align_vals, warmup_hsic_vals, global_step):
    checkpoint_remote = f"20_sweep_checkpoint_lambda{lambda_hsic}.pt"
    checkpoint_local = os.path.join(LOCAL_ROOT, checkpoint_remote)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch, "history": history,
        "align_scale": align_scale, "hsic_scale": hsic_scale,
        "warmup_align_vals": warmup_align_vals, "warmup_hsic_vals": warmup_hsic_vals,
        "global_step": global_step,
    }, checkpoint_local)
    copy_to_drive(checkpoint_local, checkpoint_remote)


def train_refinement(lambda_hsic, all_embeddings, all_groups, cf_source_indices, n_original):
    """Identical training procedure to Program 19 v1.2 (fixed-scale
    normalization) -- only lambda_hsic varies across sweep points."""
    emb_t = torch.tensor(all_embeddings, dtype=torch.float32, device=DEVICE)
    group_idx_all = np.array([GROUPS.index(g) for g in all_groups])
    group_to_indices_all = {g: np.where(group_idx_all == GROUPS.index(g))[0] for g in GROUPS}

    n_cf = len(cf_source_indices)
    cf_global_indices = np.arange(n_original, n_original + n_cf)

    model = RefinementNetwork().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    rng = np.random.RandomState(RANDOM_SEED)

    start_epoch = 0
    history = []
    align_scale, hsic_scale = None, None
    warmup_align_vals, warmup_hsic_vals = [], []
    global_step = 0
    checkpoint = try_resume_checkpoint(lambda_hsic)
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
                        print(f"[SWEEP lambda={lambda_hsic}] Warmup complete: "
                              f"align_scale={align_scale:.6f}, hsic_scale={hsic_scale:.6f}")
                else:
                    align_scale_current, hsic_scale_current = align_scale, hsic_scale
                    global_step += 1

            l_align_norm = l_align / align_scale_current
            l_hsic_norm = l_hsic / hsic_scale_current

            loss = LAMBDA_ALIGN * l_align_norm + lambda_hsic * l_hsic_norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses["align"] += align_val
            epoch_losses["hsic"] += hsic_val
            epoch_losses["align_normalized"] += l_align_norm.item()
            epoch_losses["hsic_normalized"] += l_hsic_norm.item()

        avg_losses = {k: v / STEPS_PER_EPOCH for k, v in epoch_losses.items()}
        avg_losses["epoch"] = epoch
        history.append(avg_losses)
        if epoch % 20 == 0 or epoch == N_EPOCHS - 1:
            print(f"[SWEEP lambda={lambda_hsic}] epoch {epoch}/{N_EPOCHS}  "
                  f"align={avg_losses['align']:.6f}  hsic={avg_losses['hsic']:.6f}")
        save_checkpoint(lambda_hsic, model, optimizer, epoch, history, align_scale, hsic_scale,
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

    self_consistency = {}
    for g_source in GROUPS:
        mask = group_idx == GROUPS.index(g_source)
        if mask.sum() == 0:
            continue
        col = GROUPS.index(g_source)
        self_consistency[g_source] = bootstrap_mean_ci(fractions[mask, col])

    exposure_disparity = {}
    for g_target in GROUPS:
        col = GROUPS.index(g_target)
        result = bootstrap_mean_ci(fractions[:, col])
        p_g = pool_proportions[g_target]
        result["pool_proportion"] = p_g
        result["disparity_ratio"] = result["observed"] / p_g if p_g > 0 else None
        exposure_disparity[g_target] = result

    return {"self_consistency": self_consistency, "exposure_disparity": exposure_disparity}


def batched_topk_group_fractions(embeddings, groups, k=RETRIEVAL_K, batch_size=500):
    emb_t = torch.tensor(embeddings, device=DEVICE)
    emb_norm = torch.nn.functional.normalize(emb_t, dim=1)
    n = emb_norm.shape[0]
    group_idx = np.array([GROUPS.index(g) for g in groups])
    group_idx_t = torch.tensor(group_idx, device=DEVICE)

    fractions = np.zeros((n, len(GROUPS)), dtype=np.float32)
    for start in range(0, n, batch_size):
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

        result = bootstrap_mean_ci(per_query_precision)
        results[tradition] = result
    return results


def loudness_probe_accuracy(embeddings, loudness_values):
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


def main():
    ensure_project_dir()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 2: loading augmented corpus, "
          f"loudness labels, and fidelity embeddings (once, shared across the sweep)...")
    corpus_local = os.path.join(LOCAL_ROOT, "18_augmented_corpus.pt")
    if not copy_from_drive("18_augmented_corpus.pt", corpus_local, skip_if_exists=True):
        raise RuntimeError("Could not fetch 18_augmented_corpus.pt -- run Program 18 first.")
    corpus = torch.load(corpus_local, map_location="cpu", weights_only=False)
    all_embeddings, all_groups = corpus["embeddings"], corpus["groups"]
    is_counterfactual, cf_source_indices = corpus["is_counterfactual"], corpus["cf_source_indices"]
    n_original = int((~is_counterfactual).sum())

    aux_local = os.path.join(LOCAL_ROOT, "18_aux_targets_cache.pt")
    if not copy_from_drive("18_aux_targets_cache.pt", aux_local, skip_if_exists=True):
        raise RuntimeError("Could not fetch 18_aux_targets_cache.pt -- run Program 18 first.")
    aux_data = torch.load(aux_local, map_location="cpu", weights_only=False)
    loudness_values = aux_data["style_target"].reshape(-1)

    fidelity_local = os.path.join(LOCAL_ROOT, "13_fidelity_embeddings.pt")
    if not copy_from_drive("13_fidelity_embeddings.pt", fidelity_local, skip_if_exists=True):
        raise RuntimeError("Could not fetch 13_fidelity_embeddings.pt -- run Program 13 first.")
    fidelity_data_vanilla = torch.load(fidelity_local, map_location="cpu", weights_only=False)

    original_embeddings = all_embeddings[:n_original]
    original_groups = all_groups[:n_original]

    print(f"[FIDELITY-VANILLA] Computing once, reused across all sweep points...")
    fidelity_vanilla = compute_intragroup_fidelity(fidelity_data_vanilla)
    loudness_acc_vanilla = loudness_probe_accuracy(original_embeddings, loudness_values)
    print(f"[LOUDNESS-VANILLA] accuracy={loudness_acc_vanilla:.4f}")

    sweep_results = {}
    for lambda_hsic in LAMBDA_HSIC_SWEEP:
        print(f"\n{'=' * 70}\n[{datetime.now().strftime('%H:%M:%S')}] SWEEP POINT: "
              f"lambda_HSIC={lambda_hsic}\n{'=' * 70}")
        model, history = train_refinement(lambda_hsic, all_embeddings, all_groups,
                                           cf_source_indices, n_original)

        with torch.no_grad():
            refined_pool = model(torch.tensor(original_embeddings, dtype=torch.float32, device=DEVICE)) \
                .cpu().numpy()
        refined_fidelity_data = {}
        for tradition, d in fidelity_data_vanilla.items():
            with torch.no_grad():
                refined_emb = model(torch.tensor(d["embeddings"], dtype=torch.float32, device=DEVICE)) \
                    .cpu().numpy()
            refined_fidelity_data[tradition] = {"embeddings": refined_emb, "labels": d["labels"]}

        fractions = batched_topk_group_fractions(refined_pool, original_groups)
        pool_metrics = compute_pool_level_metrics(fractions, original_groups)
        fidelity_refined = compute_intragroup_fidelity(refined_fidelity_data)
        loudness_acc_refined = loudness_probe_accuracy(refined_pool, loudness_values)

        print(f"[SWEEP lambda={lambda_hsic}] Self-consistency: " +
              ", ".join(f"{g}={r['observed']:.3f}" for g, r in pool_metrics["self_consistency"].items()))
        print(f"[SWEEP lambda={lambda_hsic}] Loudness accuracy: {loudness_acc_refined:.4f} "
              f"(vanilla={loudness_acc_vanilla:.4f})")

        sweep_results[str(lambda_hsic)] = {
            "lambda_hsic": lambda_hsic,
            "training_history_tail": history[-3:],
            "pool_level_metrics": pool_metrics,
            "fidelity_refined": fidelity_refined,
            "loudness_accuracy": loudness_acc_refined,
        }

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Saving sweep results and figures...")
    local_results_path = os.path.join(LOCAL_ROOT, "20_sweep_results.json")
    elapsed_seconds = time.time() - RUN_START_TIME
    output = {
        "program": "20_e8_e12_lambda_sweep.py",
        "lambda_hsic_sweep": LAMBDA_HSIC_SWEEP,
        "n_original_tracks": n_original, "n_counterfactuals": len(cf_source_indices),
        "fidelity_vanilla": fidelity_vanilla,
        "loudness_accuracy_vanilla": loudness_acc_vanilla,
        "sweep_results": sweep_results,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "elapsed_hms": time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds)),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(local_results_path, "w") as f:
        json.dump(output, f, indent=2)
    copy_to_drive(local_results_path, "20_sweep_results.json")

    # --- Figure 1: self-consistency vs lambda, one line per tradition ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for g in TRADITION_GROUPS:
        vals = [sweep_results[str(l)]["pool_level_metrics"]["self_consistency"][g]["observed"]
                for l in LAMBDA_HSIC_SWEEP]
        ax.plot(LAMBDA_HSIC_SWEEP, vals, marker="o", label=g)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("lambda_HSIC"); ax.set_ylabel("Self-consistency")
    ax.set_title("E12: Self-consistency vs. lambda_HSIC (E8 = lambda=0 point)")
    ax.legend()
    fig.tight_layout()
    local_fig1 = os.path.join(LOCAL_ROOT, "fig_20_sweep_self_consistency.png")
    fig.savefig(local_fig1, dpi=300)
    plt.close(fig)
    copy_to_drive(local_fig1, "fig_20_sweep_self_consistency.png")

    # --- Figure 2: mean fidelity ratio and loudness accuracy vs lambda ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    mean_fidelity_vanilla = np.mean([fidelity_vanilla[t]["observed"] for t in fidelity_vanilla])
    mean_fidelity_ratios = [
        np.mean([sweep_results[str(l)]["fidelity_refined"][t]["observed"] for t in fidelity_vanilla])
        / mean_fidelity_vanilla
        for l in LAMBDA_HSIC_SWEEP
    ]
    ax1.plot(LAMBDA_HSIC_SWEEP, mean_fidelity_ratios, marker="o", color="green")
    ax1.axhline(1.0, color="gray", linestyle="--", label="Vanilla level")
    ax1.set_xscale("symlog", linthresh=0.1)
    ax1.set_xlabel("lambda_HSIC"); ax1.set_ylabel("Mean fidelity ratio (refined/vanilla)")
    ax1.set_title("Fidelity preservation vs. lambda_HSIC")
    ax1.legend()

    loudness_vals = [sweep_results[str(l)]["loudness_accuracy"] for l in LAMBDA_HSIC_SWEEP]
    ax2.plot(LAMBDA_HSIC_SWEEP, loudness_vals, marker="o", color="purple")
    ax2.axhline(loudness_acc_vanilla, color="gray", linestyle="--", label="Vanilla")
    ax2.axhline(0.5, color="black", linestyle=":", label="Chance")
    ax2.set_xscale("symlog", linthresh=0.1)
    ax2.set_xlabel("lambda_HSIC"); ax2.set_ylabel("Loudness probe accuracy")
    ax2.set_title("Confound reduction vs. lambda_HSIC")
    ax2.legend()
    fig.tight_layout()
    local_fig2 = os.path.join(LOCAL_ROOT, "fig_20_sweep_fidelity_and_loudness.png")
    fig.savefig(local_fig2, dpi=300)
    plt.close(fig)
    copy_to_drive(local_fig2, "fig_20_sweep_fidelity_and_loudness.png")

    return output


if __name__ == "__main__":
    summary = main()
    elapsed = time.time() - RUN_START_TIME
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Program 20_e8_e12_lambda_sweep.py finished.")
    print(f"[TOTAL RUNNING TIME] {time.strftime('%H:%M:%S', time.gmtime(elapsed))} ({elapsed:.1f}s)")