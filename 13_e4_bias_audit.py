# =============================================================================
# Program      : 13_e4_bias_audit.py
# Version      : 1.0
# GPU Required : NO (uses GPU automatically if available, for speed only)
# Description  : Experiment 4 -- Stage 1 Bias Audit. Quantifies baseline
#                 retrieval bias against all 4 traditions using the
#                 trained E1 backbone, BEFORE any mitigation.
#
#                 Two embedding sets, two different jobs:
#                   - E1's 7 anonymous crop caches (~16,900 tracks,
#                     tagged only by tradition -- Dominant/Carnatic/
#                     Hindustani/Jingju/ArabAndalusian) -> the 3
#                     POOL-LEVEL metrics, which only need to know which
#                     tradition a track belongs to, not its fine label.
#                   - Program 12's 4 labeled fidelity caches (CMD raga,
#                     HMD raga, Jingju role-type, Arab-Andalusian nawba)
#                     -> Intra-Group Retrieval Fidelity, which needs the
#                     fine-grained label E1's caches don't carry.
#
#                 All three pool-level metrics (Exposure Disparity,
#                 Self-Consistency, Cross-Tradition Retrievability) come
#                 from ONE underlying computation: for every query track,
#                 what fraction of its top-K nearest neighbors (cosine
#                 similarity, self excluded) belongs to each of the 5
#                 groups. This gives a per-query row that sums to 1
#                 across groups; the three metrics are just different
#                 views of the same [N_queries x 5] matrix:
#                   - Self-Consistency(g)   = mean, over queries FROM g,
#                                             of their own-group fraction
#                                             (the diagonal of the full
#                                             group x group matrix)
#                   - Cross-Tradition       = the FULL group x group
#                     Retrievability          matrix (mean, over queries
#                                             from g_source, of their
#                                             g_target fraction)
#                   - Exposure Disparity(g) = mean, over ALL queries
#                                             regardless of source, of
#                                             their g fraction (g's
#                                             overall share of retrieval
#                                             slots) divided by g's pool
#                                             proportion (>1 = over-
#                                             exposed, <1 = under-exposed)
#
#                 Bootstrap CIs (10,000 iterations) resample QUERIES,
#                 reusing each query's precomputed group-fraction row --
#                 the expensive similarity computation happens once, not
#                 once per bootstrap iteration.
#
#                 Intra-Group Fidelity is computed separately per
#                 tradition, WITHIN that tradition's own fidelity corpus
#                 only (a raga/nawba/role-type label has no meaning
#                 relative to a track from a different tradition):
#                 precision@K = mean fraction of a query's top-K (within
#                 its own tradition, self excluded) sharing its exact
#                 fine-grained label.
#
# INPUT:
#                  Google Drive:
#                    08_crops_fma.pt, 08_crops_carnatic.pt,
#                    08_crops_hindustani.pt, 08_crops_jingju_base.pt,
#                    08_crops_jacrc.pt, 08_crops_arab_andalusian.pt,
#                    08_crops_jamendo.pt, 08_backbone_final.pt
#                    12_fidelity_cmd_carnatic.pt, 12_fidelity_hmd_hindustani.pt,
#                    12_fidelity_jingju_roletype.pt,
#                    12_fidelity_arab_andalusian_nawba.pt
#                    (NOTE: despite Program 12's own header comment saying
#                    these names, its actual code uploads them WITHOUT a
#                    "12_fidelity_" prefix -- just cmd_carnatic.pt,
#                    hmd_hindustani.pt, jingju_roletype.pt,
#                    arab_andalusian_nawba.pt. This program copies the
#                    real names, not the documented-but-wrong ones.)
#
# STEPS:
#                  Step 1  Ensure PROJECT_DIR exists
#                  Step 2  Copy trained backbone + all 11 crop caches
#                  Step 3  Compute pool embeddings (mean over the 3
#                          stored crops/track) for the 7 E1 caches,
#                          tagged by tradition group; compute fidelity
#                          embeddings (1 crop/track) for the 4 labeled
#                          caches, tagged by fine label. Cache both
#                          (embeddings are tiny compared to raw audio)
#                  Step 4  Pool-level: batched cosine-similarity top-K
#                          (K=10, self excluded) against the full pool;
#                          build the [N_queries x 5-group] fraction
#                          matrix; derive Self-Consistency, Cross-
#                          Tradition Retrievability, Exposure Disparity,
#                          each with a bootstrap 95% CI
#                  Step 5  Intra-Group Fidelity: per-tradition top-K
#                          (within that tradition's fidelity corpus only)
#                          precision@K against the fine-grained label,
#                          with a bootstrap 95% CI per tradition
#                  Step 6  Save results JSON + a group x group heatmap
#                          and a fidelity precision@K bar chart, copy
#                          both to Google Drive
#
# OUTPUT FILES : under PROJECT_DIR
#                  13_bias_audit_results.json
#                  fig_13_group_retrieval_heatmap.png
#                  fig_13_intragroup_fidelity.png
#
# Hyperparameters : SAMPLE_RATE=22050, SEGMENT_SECONDS=4, N_MELS=64,
#                    EMBEDDING_DIM=256 (must match Program 08 exactly),
#                    RETRIEVAL_K=10, N_BOOTSTRAP_ITERATIONS=10000,
#                    RANDOM_SEED=42, SIMILARITY_BATCH_SIZE=500.
# Dependencies : torch, torchaudio, numpy, pandas, tqdm, matplotlib
#
# Change Log   :
#   v1.0  2026-09-23  Initial version
# Running time: 31:00
# !pip install torch torchaudio pandas tqdm matplotlib
# =============================================================================

!pip install torch torchaudio pandas tqdm matplotlib

import os
import time
import json
from datetime import datetime

import numpy as np
import pandas as pd
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
print(f"[{datetime.now().strftime('%H:%M:%S')}] Program 13_e4_bias_audit.py started.")

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
N_BOOTSTRAP_ITERATIONS = 10000
RANDOM_SEED = 42
SIMILARITY_BATCH_SIZE = 500

POOL_SOURCES = ["fma", "carnatic", "hindustani", "jingju_base", "jacrc", "arab_andalusian", "jamendo"]
SOURCE_TO_GROUP = {
    "fma": "Dominant", "jamendo": "Dominant",
    "carnatic": "Carnatic", "hindustani": "Hindustani",
    "jingju_base": "Jingju", "jacrc": "Jingju",
    "arab_andalusian": "ArabAndalusian",
}
GROUPS = ["Dominant", "Carnatic", "Hindustani", "Jingju", "ArabAndalusian"]

FIDELITY_CORPORA = {
    "Carnatic": "cmd_carnatic", "Hindustani": "hmd_hindustani",
    "Jingju": "jingju_roletype", "ArabAndalusian": "arab_andalusian_nawba",
}

LOCAL_ROOT = "/content/local_data_e4audit"
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


# --- Same AudioEncoder architecture as Program 08/11 -- must match
# exactly for the trained state_dict to load correctly. ---
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
    sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=N_MELS).to(DEVICE)
_AMPLITUDE_TO_DB = torchaudio.transforms.AmplitudeToDB().to(DEVICE)


def crop_to_mel(waveform_int16):
    waveform = (waveform_int16.float() / 32767.0).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return _AMPLITUDE_TO_DB(_MEL_TRANSFORM(waveform))


def embed_crop(model, waveform_int16):
    mel = crop_to_mel(waveform_int16)
    with torch.no_grad():
        embedding, _ = model(mel)
    return embedding.squeeze(0).cpu().numpy()  # [256]


def compute_pool_embeddings(model):
    """Mean embedding over the 3 stored crops/track, tagged by tradition
    GROUP (not fine label -- that's what the fidelity corpora are for)."""
    cache_local = os.path.join(LOCAL_ROOT, "13_pool_embeddings.pt")
    if copy_from_drive("13_pool_embeddings.pt", cache_local, skip_if_exists=True):
        cached = torch.load(cache_local, map_location="cpu", weights_only=False)
        print(f"[CACHE] Reused cached pool embeddings: {cached['embeddings'].shape[0]} tracks.")
        return cached["embeddings"], cached["groups"]

    embeddings, groups = [], []
    for name in POOL_SOURCES:
        crops_local = os.path.join(LOCAL_ROOT, f"crops_{name}.pt")
        if not copy_from_drive(f"08_crops_{name}.pt", crops_local, skip_if_exists=True):
            print(f"[WARNING] Could not fetch 08_crops_{name}.pt -- skipping.")
            continue
        data = torch.load(crops_local, map_location="cpu", weights_only=False)
        crops = data["crops"]  # [N, 3, SEGMENT_LENGTH] int16
        group = SOURCE_TO_GROUP[name]
        for i in tqdm(range(crops.shape[0]), desc=f"[EMBED-POOL] {name}"):
            crop_embeddings = [embed_crop(model, crops[i, c]) for c in range(crops.shape[1])]
            embeddings.append(np.mean(crop_embeddings, axis=0))
            groups.append(group)
        del data, crops
        os.remove(crops_local)
        print(f"[SOURCE] {name}: {len(groups)} tracks embedded so far (cumulative).")

    embeddings = np.stack(embeddings).astype(np.float32)
    groups = np.array(groups)
    torch.save({"embeddings": embeddings, "groups": groups}, cache_local)
    copy_to_drive(cache_local, "13_pool_embeddings.pt")
    return embeddings, groups


def compute_fidelity_embeddings(model):
    """1 embedding/track (Program 12 stored a single fixed crop),
    tagged by the FINE-grained label, per tradition."""
    cache_local = os.path.join(LOCAL_ROOT, "13_fidelity_embeddings.pt")
    if copy_from_drive("13_fidelity_embeddings.pt", cache_local, skip_if_exists=True):
        cached = torch.load(cache_local, map_location="cpu", weights_only=False)
        print(f"[CACHE] Reused cached fidelity embeddings for "
              f"{list(cached.keys())}.")
        return cached

    result = {}
    for tradition, name_prefix in FIDELITY_CORPORA.items():
        crops_local = os.path.join(LOCAL_ROOT, f"fidelity_{name_prefix}.pt")
        if not copy_from_drive(f"{name_prefix}.pt", crops_local, skip_if_exists=True):
            print(f"[WARNING] Could not fetch {name_prefix}.pt -- skipping {tradition} fidelity.")
            continue
        data = torch.load(crops_local, map_location="cpu", weights_only=False)
        crops = data["crops"]  # [N, SEGMENT_LENGTH] int16
        labels = data["labels"]
        embeddings = []
        for i in tqdm(range(crops.shape[0]), desc=f"[EMBED-FIDELITY] {tradition}"):
            embeddings.append(embed_crop(model, crops[i]))
        embeddings = np.stack(embeddings).astype(np.float32)
        result[tradition] = {"embeddings": embeddings, "labels": np.array(labels)}
        os.remove(crops_local)
        print(f"[FIDELITY] {tradition}: {embeddings.shape[0]} tracks embedded.")

    torch.save(result, cache_local)
    copy_to_drive(cache_local, "13_fidelity_embeddings.pt")
    return result


def batched_topk_group_fractions(embeddings, groups, k=RETRIEVAL_K, batch_size=SIMILARITY_BATCH_SIZE):
    """For every track as a query, cosine-similarity top-K (self
    excluded) against the full pool; returns [N x len(GROUPS)] fractions
    -- row i, column g = fraction of query i's top-K belonging to group g."""
    emb_t = torch.tensor(embeddings, device=DEVICE)
    emb_norm = torch.nn.functional.normalize(emb_t, dim=1)
    n = emb_norm.shape[0]
    group_idx = np.array([GROUPS.index(g) for g in groups])
    group_idx_t = torch.tensor(group_idx, device=DEVICE)

    fractions = np.zeros((n, len(GROUPS)), dtype=np.float32)
    for start in tqdm(range(0, n, batch_size), desc="[RETRIEVAL] pool top-K"):
        end = min(start + batch_size, n)
        batch = emb_norm[start:end]  # [b, D]
        sims = batch @ emb_norm.T  # [b, N]
        # Exclude self.
        row_ids = torch.arange(start, end, device=DEVICE)
        sims[torch.arange(end - start), row_ids] = -float("inf")
        topk_idx = torch.topk(sims, k=k, dim=1).indices  # [b, k]
        topk_groups = group_idx_t[topk_idx]  # [b, k]
        for g_i in range(len(GROUPS)):
            fractions[start:end, g_i] = (topk_groups == g_i).float().mean(dim=1).cpu().numpy()
    return fractions


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
        values = fractions[:, col]  # ALL queries, regardless of source
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
        result["k_used"] = int(eff_k)
        results[tradition] = result
        print(f"[FIDELITY] {tradition}: precision@{eff_k}={result['observed']:.4f} "
              f"(chance={chance_level:.4f}, n_classes={n_classes}, n_tracks={n})")
    return results


def main():
    ensure_project_dir()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 2: copying trained backbone...")
    checkpoint_local = os.path.join(LOCAL_ROOT, "08_backbone_final.pt")
    if not copy_from_drive("08_backbone_final.pt", checkpoint_local, skip_if_exists=True):
        raise RuntimeError("Could not fetch 08_backbone_final.pt -- run Program 08 first.")
    model = load_trained_encoder(checkpoint_local)
    print(f"[MODEL] Loaded trained AudioEncoder, eval mode, device={DEVICE}.")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 3: computing pool + fidelity embeddings "
          f"(or reusing cached embeddings)...")
    pool_embeddings, pool_groups = compute_pool_embeddings(model)
    fidelity_data = compute_fidelity_embeddings(model)
    print(f"[POOL] {pool_embeddings.shape[0]} tracks, group distribution: "
          f"{pd.Series(pool_groups).value_counts().to_dict()}")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 4: pool-level retrieval metrics "
          f"(K={RETRIEVAL_K})...")
    fractions = batched_topk_group_fractions(pool_embeddings, pool_groups)
    pool_metrics = compute_pool_level_metrics(fractions, pool_groups)
    print("[SELF-CONSISTENCY]")
    for g, r in pool_metrics["self_consistency"].items():
        print(f"    {g:16s} {r['observed']:.4f}  95% CI=({r['ci_95_low']:.4f}, {r['ci_95_high']:.4f})")
    print("[EXPOSURE DISPARITY] (ratio to pool proportion; 1.0 = proportional)")
    for g, r in pool_metrics["exposure_disparity"].items():
        print(f"    {g:16s} ratio={r['disparity_ratio']:.3f}  "
              f"(pool share={r['pool_proportion']:.4f}, observed exposure={r['observed']:.4f})")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 5: intra-group retrieval fidelity...")
    fidelity_metrics = compute_intragroup_fidelity(fidelity_data)

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 6: saving results and figures...")
    local_results_path = os.path.join(LOCAL_ROOT, "13_bias_audit_results.json")
    elapsed_seconds = time.time() - RUN_START_TIME
    output = {
        "program": "13_e4_bias_audit.py",
        "retrieval_k": RETRIEVAL_K,
        "n_pool_tracks": int(pool_embeddings.shape[0]),
        "pool_level_metrics": pool_metrics,
        "intragroup_fidelity": fidelity_metrics,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "elapsed_hms": time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds)),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(local_results_path, "w") as f:
        json.dump(output, f, indent=2)

    # --- Figure 1: group x group retrieval heatmap ---
    matrix_vals = np.array([[pool_metrics["cross_tradition_retrievability_matrix"][gs][gt]["observed"]
                              for gt in GROUPS] for gs in GROUPS])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix_vals, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(GROUPS))); ax.set_xticklabels(GROUPS, rotation=30, ha="right")
    ax.set_yticks(range(len(GROUPS))); ax.set_yticklabels(GROUPS)
    ax.set_xlabel("Retrieved group"); ax.set_ylabel("Query group")
    ax.set_title(f"E4: Cross-tradition retrievability (K={RETRIEVAL_K})")
    for i in range(len(GROUPS)):
        for j in range(len(GROUPS)):
            ax.text(j, i, f"{matrix_vals[i, j]:.2f}", ha="center", va="center",
                     color="white" if matrix_vals[i, j] < 0.5 else "black")
    fig.colorbar(im, ax=ax, label="Fraction of top-K")
    fig.tight_layout()
    local_fig1 = os.path.join(LOCAL_ROOT, "fig_13_group_retrieval_heatmap.png")
    fig.savefig(local_fig1, dpi=300)
    plt.close(fig)

    # --- Figure 2: intra-group fidelity precision@K vs chance ---
    traditions = list(fidelity_metrics.keys())
    precisions = [fidelity_metrics[t]["observed"] for t in traditions]
    chances = [fidelity_metrics[t]["chance_level"] for t in traditions]
    x = np.arange(len(traditions))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, precisions, width, label=f"Precision@{RETRIEVAL_K}")
    ax.bar(x + width / 2, chances, width, label="Chance (1/n_classes)")
    ax.set_xticks(x); ax.set_xticklabels(traditions)
    ax.set_ylabel("Precision")
    ax.set_title("E4: Intra-group retrieval fidelity vs chance")
    ax.legend()
    fig.tight_layout()
    local_fig2 = os.path.join(LOCAL_ROOT, "fig_13_intragroup_fidelity.png")
    fig.savefig(local_fig2, dpi=300)
    plt.close(fig)

    copy_to_drive(local_results_path, "13_bias_audit_results.json")
    copy_to_drive(local_fig1, "fig_13_group_retrieval_heatmap.png")
    copy_to_drive(local_fig2, "fig_13_intragroup_fidelity.png")

    return output


if __name__ == "__main__":
    summary = main()
    elapsed = time.time() - RUN_START_TIME
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Program 13_e4_bias_audit.py finished.")
    print(f"[TOTAL RUNNING TIME] {time.strftime('%H:%M:%S', time.gmtime(elapsed))} ({elapsed:.1f}s)")