# =============================================================================
# Program      : 22_e14_diversity_homogenization.py
# Version      : 1.0
# GPU Required : NO
# Description  : Experiment E14 -- Diversity/Homogenization Check.
#                 Directly tests the "HSIC homogenizes the retrieval
#                 space" hypothesis that has been the mechanistic
#                 explanation since BL5 (Program 15), but was never
#                 measured directly -- self-consistency and exposure
#                 disparity (E4's metrics, reused throughout E5/E6) tell
#                 us about a query's OWN group representation in its
#                 results, not about how DIVERSE those results are
#                 overall. A query could have low self-consistency
#                 (results mostly outside its own group) while still
#                 being either genuinely diverse (spread across several
#                 other groups) or homogenized (funneled into just one
#                 other group, typically Dominant) -- these are different
#                 failure modes that self-consistency alone can't tell
#                 apart.
#
#                 Two entropy metrics, both computed from the SAME
#                 per-query top-K group-fraction matrix already used
#                 throughout E4/E5/E6 (Shannon entropy in bits, max =
#                 log2(5) = 2.322 for a perfectly uniform 5-group split):
#                   - PER-QUERY entropy, averaged within each query GROUP
#                     separately (does an average Carnatic query's top-10
#                     span multiple groups, or collapse into one?)
#                   - AGGREGATE pool-level entropy (the same "observed"
#                     exposure-share values already computed for E4's
#                     Exposure Disparity metric, but entropy of that
#                     distribution instead of a per-group ratio -- how
#                     evenly is retrieval exposure spread across the
#                     whole pool, overall, regardless of any single
#                     query's pattern)
#
#                 Computed on VANILLA (Program 13's cache) plus every
#                 point from Program 21's lambda_HSIC sweep -- reuses
#                 those already-trained checkpoints directly (no new
#                 training at all), encoding the pool once per checkpoint
#                 and computing entropy on top. Genuinely cheap: this is
#                 a new metric on existing model outputs, not a new
#                 experiment requiring new training.
#
#                 All inputs and outputs for this program live in
#                 PROJECT_DIR on Google Drive. There is no separate
#                 cache/mirror directory: every checkpoint, results
#                 JSON, figure PNG, and .pt cache produced by any
#                 program in this project lives here, and every program
#                 reads its inputs from here too.
#
# INPUT:
#                  Google Drive (all under PROJECT_DIR):
#                    13_pool_embeddings.pt (vanilla E1 embeddings)
#                    21_sweep_checkpoint_lambda<value>.pt (one per sweep
#                      point, Program 21's finer lambda_HSIC sweep)
#
# STEPS:
#                  Step 1  Mount Drive, ensure PROJECT_DIR exists
#                  Step 2  Load vanilla pool embeddings
#                  Step 3  For vanilla and each lambda sweep point:
#                          (re-load the trained RefinementNetwork from
#                          its checkpoint for sweep points; vanilla needs
#                          no model at all) encode the pool, compute
#                          top-K group fractions, then both entropy
#                          metrics
#                  Step 4  Save results JSON + a comparison figure
#                          (entropy vs. lambda_HSIC, vanilla as a
#                          reference line) to PROJECT_DIR
#
# OUTPUT FILES : under PROJECT_DIR
#                  22_diversity_results.json
#                  fig_22_entropy_vs_lambda.png
#
# Hyperparameters : EMBEDDING_DIM=256, HIDDEN_DIM=256, RETRIEVAL_K=10,
#                    LAMBDA_HSIC_SWEEP=[0.0, 0.03, 0.1, 0.3, 0.5, 0.7,
#                    1.0, 1.5, 2.0, 3.0, 5.0, 10.0] (matches Program 21
#                    exactly, so checkpoint filenames line up).
# Dependencies : torch, numpy, pandas, matplotlib
#
# Change Log   :
#   v1.0  2026-09-25  Initial version
# Running time:
# !pip install torch pandas matplotlib
# =============================================================================

!pip install torch pandas matplotlib

import os
import time
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

RUN_START_TIME = time.time()
print(f"[{datetime.now().strftime('%H:%M:%S')}] Program 22_e14_diversity_homogenization.py started.")

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
RETRIEVAL_K = 10
LAMBDA_HSIC_SWEEP = [0.0, 0.03, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

GROUPS = ["Dominant", "Carnatic", "Hindustani", "Jingju", "ArabAndalusian"]
TRADITION_GROUPS = ["Carnatic", "Hindustani", "Jingju", "ArabAndalusian"]
MAX_ENTROPY_BITS = np.log2(len(GROUPS))  # 2.322 -- perfectly uniform 5-group split

LOCAL_ROOT = "/content/local_data_e14"
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
    with open(local_filepath, "rb") as fsrc, open(dest, "wb") as fdst:
        while True:
            chunk = fsrc.read(chunk_size)
            if not chunk:
                break
            fdst.write(chunk)


def copy_from_drive(remote_filename, local_filepath, skip_if_exists=True):
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
    with open(src, "rb") as fsrc, open(local_filepath, "wb") as fdst:
        while True:
            chunk = fsrc.read(chunk_size)
            if not chunk:
                break
            fdst.write(chunk)
    return True
# --- End Google Drive Helper Functions ---


class RefinementNetwork(nn.Module):
    """Identical architecture to Programs 19/20/21 -- required to load
    their saved checkpoints."""
    def __init__(self, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x):
        return x + self.mlp(x)


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


def shannon_entropy_bits(distribution, eps=1e-12):
    """distribution: array of non-negative values summing to ~1 (or a
    2D array of such rows). Returns entropy in bits; 0 for a fully
    concentrated (one-hot) distribution, log2(n) for a uniform one."""
    p = np.clip(distribution, eps, 1.0)
    return -np.sum(p * np.log2(p), axis=-1)


def compute_diversity_metrics(fractions, groups):
    """Returns per-query-group mean entropy (does an average query FROM
    this group get diverse top-K results?) and the aggregate pool-level
    entropy (how evenly is overall retrieval exposure spread across the
    whole pool, regardless of any single query)."""
    group_idx = np.array([GROUPS.index(g) for g in groups])
    per_query_entropy = shannon_entropy_bits(fractions)  # [N]

    per_group_mean_entropy = {}
    for g in GROUPS:
        mask = group_idx == GROUPS.index(g)
        if mask.sum() > 0:
            per_group_mean_entropy[g] = float(per_query_entropy[mask].mean())

    aggregate_distribution = fractions.mean(axis=0)  # [5] -- overall exposure share per group
    aggregate_entropy = float(shannon_entropy_bits(aggregate_distribution))

    return {
        "per_group_mean_entropy": per_group_mean_entropy,
        "aggregate_entropy": aggregate_entropy,
        "aggregate_distribution": {g: float(aggregate_distribution[i]) for i, g in enumerate(GROUPS)},
    }


def main():
    ensure_project_dir()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 2: loading vanilla pool embeddings...")
    pool_cache_local = os.path.join(LOCAL_ROOT, "13_pool_embeddings.pt")
    if not copy_from_drive("13_pool_embeddings.pt", pool_cache_local):
        raise RuntimeError("Could not fetch 13_pool_embeddings.pt -- run Program 13 first.")
    cached = torch.load(pool_cache_local, map_location="cpu", weights_only=False)
    vanilla_embeddings, groups = cached["embeddings"], cached["groups"]
    print(f"[POOL] {vanilla_embeddings.shape[0]} tracks, max possible entropy = "
          f"{MAX_ENTROPY_BITS:.3f} bits (uniform across {len(GROUPS)} groups).")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 3: vanilla diversity metrics...")
    vanilla_fractions = batched_topk_group_fractions(vanilla_embeddings, groups)
    vanilla_diversity = compute_diversity_metrics(vanilla_fractions, groups)
    print(f"[VANILLA] Aggregate entropy: {vanilla_diversity['aggregate_entropy']:.4f} bits")
    for g, e in vanilla_diversity["per_group_mean_entropy"].items():
        print(f"    {g:16s} per-query mean entropy = {e:.4f} bits")

    sweep_diversity = {}
    for lambda_hsic in LAMBDA_HSIC_SWEEP:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] lambda_HSIC={lambda_hsic}: "
              f"loading checkpoint and encoding pool...")
        checkpoint_remote = f"21_sweep_checkpoint_lambda{lambda_hsic}.pt"
        checkpoint_local = os.path.join(LOCAL_ROOT, checkpoint_remote)
        if not copy_from_drive(checkpoint_remote, checkpoint_local):
            print(f"[WARNING] Could not fetch {checkpoint_remote} -- skipping lambda={lambda_hsic}.")
            continue
        checkpoint = torch.load(checkpoint_local, map_location=DEVICE, weights_only=False)
        model = RefinementNetwork().to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        with torch.no_grad():
            refined_embeddings = model(
                torch.tensor(vanilla_embeddings, dtype=torch.float32, device=DEVICE)
            ).cpu().numpy()

        fractions = batched_topk_group_fractions(refined_embeddings, groups)
        diversity = compute_diversity_metrics(fractions, groups)
        print(f"[lambda={lambda_hsic}] Aggregate entropy: {diversity['aggregate_entropy']:.4f} bits "
              f"(vanilla: {vanilla_diversity['aggregate_entropy']:.4f})")
        sweep_diversity[str(lambda_hsic)] = diversity
        os.remove(checkpoint_local)

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Saving results and figure...")
    local_results_path = os.path.join(LOCAL_ROOT, "22_diversity_results.json")
    elapsed_seconds = time.time() - RUN_START_TIME
    output = {
        "program": "22_e14_diversity_homogenization.py",
        "max_entropy_bits": float(MAX_ENTROPY_BITS),
        "vanilla_diversity": vanilla_diversity,
        "lambda_hsic_sweep": LAMBDA_HSIC_SWEEP,
        "sweep_diversity": sweep_diversity,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "elapsed_hms": time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds)),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(local_results_path, "w") as f:
        json.dump(output, f, indent=2)
    copy_to_drive(local_results_path, "22_diversity_results.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.axhline(vanilla_diversity["aggregate_entropy"], color="gray", linestyle="--", label="Vanilla")
    ax1.axhline(MAX_ENTROPY_BITS, color="black", linestyle=":", label="Max (uniform)")
    lambdas_present = [l for l in LAMBDA_HSIC_SWEEP if str(l) in sweep_diversity]
    agg_vals = [sweep_diversity[str(l)]["aggregate_entropy"] for l in lambdas_present]
    ax1.plot(lambdas_present, agg_vals, marker="o", color="teal")
    ax1.set_xscale("symlog", linthresh=0.1)
    ax1.set_xlabel("lambda_HSIC"); ax1.set_ylabel("Aggregate pool entropy (bits)")
    ax1.set_title("Overall retrieval exposure diversity vs. lambda_HSIC")
    ax1.legend()

    for g in TRADITION_GROUPS:
        vals = [sweep_diversity[str(l)]["per_group_mean_entropy"].get(g) for l in lambdas_present]
        ax2.plot(lambdas_present, vals, marker="o", label=g)
        ax2.axhline(vanilla_diversity["per_group_mean_entropy"][g], linestyle="--", alpha=0.4)
    ax2.set_xscale("symlog", linthresh=0.1)
    ax2.set_xlabel("lambda_HSIC"); ax2.set_ylabel("Per-query mean entropy (bits)")
    ax2.set_title("Per-tradition query-result diversity vs. lambda_HSIC\n(dashed = vanilla)")
    ax2.legend()

    fig.tight_layout()
    local_fig = os.path.join(LOCAL_ROOT, "fig_22_entropy_vs_lambda.png")
    fig.savefig(local_fig, dpi=300)
    plt.close(fig)
    copy_to_drive(local_fig, "fig_22_entropy_vs_lambda.png")

    return output


if __name__ == "__main__":
    summary = main()
    elapsed = time.time() - RUN_START_TIME
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Program 22_e14_diversity_homogenization.py finished.")
    print(f"[TOTAL RUNNING TIME] {time.strftime('%H:%M:%S', time.gmtime(elapsed))} ({elapsed:.1f}s)")