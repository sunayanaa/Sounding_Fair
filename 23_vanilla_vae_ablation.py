# =============================================================================
# Program      : 23_vanilla_vae_ablation.py
# Version      : 1.0
# GPU Required : NO (small MLP-based models throughout; GPU used
#                 automatically if present)
# Description  : The one additional experiment for the TAI Brief
#                 Manuscript: tests whether this project's specific
#                 methodological contribution -- semantically-grounded
#                 content/style disentanglement via auxiliary supervision
#                 (Program 18) -- was actually NECESSARY for the fidelity
#                 preservation observed throughout E6, or whether a
#                 VANILLA disentangled VAE (same architecture, same TC/
#                 HSIC independence penalty between z_c and z_s, but with
#                 NO auxiliary heads pushing them toward known content/
#                 style targets) would have preserved fidelity just as
#                 well. If it does, the paper's specific architectural
#                 choice isn't doing the work credited to it. If it
#                 doesn't -- expected, given BL5's own collapse under a
#                 similarly ungrounded HSIC penalty -- this becomes a
#                 clean, direct ablation substantiating the paper's
#                 actual technical claim, currently asserted rather than
#                 demonstrated.
#
#                 Isolates exactly ONE variable for a fair comparison:
#                 the VanillaVAE here uses the IDENTICAL encoder input
#                 (concat[E1 embedding, tonic-normalized content
#                 features], 335-dim -- Program 18 v1.2's fix) and
#                 identical hyperparameters (beta_KL=1.0, gamma_TC=1.0,
#                 matching v1.4's final values) as the real dVAE. The
#                 ONLY difference: no content_aux_head, no style_aux_head,
#                 no auxiliary loss terms -- z_c and z_s are shaped by
#                 reconstruction + KL + TC/HSIC alone, exactly the
#                 "statistical independence without semantic grounding"
#                 setup this whole project has argued is insufficient.
#
#                 Downstream pipeline is otherwise identical: same
#                 bidirectional, distance-quantile-band counterfactual
#                 generation (rho=0.15) from Program 18, then the SAME
#                 fixed-scale-normalized refinement network training from
#                 Program 19/21 (the bug-free version), at lambda_HSIC=1.5
#                 -- matching Program 21's sweep peak, so this ablation is
#                 compared against the REAL method's best-case result, not
#                 an arbitrary or unfavorable setting.
#
#                 All inputs and outputs for this program live in
#                 PROJECT_DIR on Google Drive. There is no separate
#                 cache/mirror directory: every checkpoint, results JSON,
#                 figure PNG, and .pt cache produced by any program in
#                 this project lives here, and every program reads its
#                 inputs from here too.
#
# INPUT:
#                  Google Drive (all under PROJECT_DIR):
#                    13_pool_embeddings.pt (Program 13 -- vanilla E1
#                      embeddings, tagged by group)
#                    18_aux_targets_cache.pt (Program 18 -- tonic-
#                      normalized content features + loudness labels,
#                      same order as the pool embeddings; reused as
#                      EXTRA ENCODER INPUT here, but NOT as a supervision
#                      target -- that's the whole point of the ablation)
#                    13_fidelity_embeddings.pt (Program 13 -- E1
#                      embeddings for the 4 labeled fidelity corpora)
#
# STEPS:
#                  Step 1  Mount Drive, ensure PROJECT_DIR exists
#                  Step 2  Load pool embeddings + auxiliary features
#                          (content features used as input only, not a
#                          supervision target) + fidelity embeddings
#                  Step 3  Train the VanillaVAE (no auxiliary heads);
#                          checkpoint every epoch
#                  Step 4  Encode the full pool; generate counterfactuals
#                          via the same distance-quantile donor selection
#                          as Program 18
#                  Step 5  Train the refinement network on this vanilla-
#                          VAE-based augmented corpus at lambda_HSIC=1.5
#                          (Program 19/21's fixed-scale-normalized
#                          training, unchanged)
#                  Step 6  Validate: self-consistency, Intra-Group
#                          Fidelity, loudness probe -- compared directly
#                          against E4 vanilla and Program 21's
#                          lambda=1.5 result (the real, semantically-
#                          grounded method)
#                  Step 7  Save results JSON + a 3-way comparison figure
#                          to PROJECT_DIR
#
# OUTPUT FILES : under PROJECT_DIR
#                  23_vanilla_vae_checkpoint.pt
#                  23_refinement_checkpoint.pt
#                  23_ablation_results.json
#                  fig_23_fidelity_comparison.png
#
# Hyperparameters : EMBEDDING_DIM=256, HIDDEN_DIM=512 (dVAE) / 256
#                    (refinement), LATENT_DIM=64, BETA_KL=1.0,
#                    GAMMA_TC=1.0 (matching Program 18 v1.4), N_EPOCHS=100
#                    (dVAE) / 50 (refinement), AUGMENTATION_RATIO=0.15,
#                    DONOR_PERCENTILE=[40,70], LAMBDA_HSIC_REFINE=1.5
#                    (Program 21's sweep peak), RETRIEVAL_K=10,
#                    N_BOOTSTRAP_ITERATIONS=10000, RANDOM_SEED=42.
# Dependencies : torch, numpy, pandas, tqdm, matplotlib, scikit-learn
#
# Change Log   :
#   v1.0  2026-09-25  Initial version
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
print(f"[{datetime.now().strftime('%H:%M:%S')}] Program 23_vanilla_vae_ablation.py started.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] Using {DEVICE} (GPU not required; used automatically if present).")

# --- Configuration ---
# PROJECT_DIR is the single source of truth for the pipeline: every
# checkpoint, results JSON, figure PNG, and .pt cache produced by any
# program in this project lives here, and every program reads its inputs
# from here too. No separate cache/mirror directory exists.
PROJECT_DIR = "/content/drive/MyDrive/paper/sounding-fair/"

EMBEDDING_DIM = 256
HIDDEN_DIM_VAE = 512
HIDDEN_DIM_REFINE = 256
LATENT_DIM = 64
BETA_KL = 1.0
GAMMA_TC = 1.0
DVAE_BATCH_SIZE = 512
DVAE_N_EPOCHS = 100
AUGMENTATION_RATIO = 0.15
DONOR_PERCENTILE_LOW = 40
DONOR_PERCENTILE_HIGH = 70
LAMBDA_ALIGN = 1.0
LAMBDA_HSIC_REFINE = 1.5  # Program 21's sweep peak -- comparing against the real method's best case
NORM_WARMUP_STEPS = 100
HSIC_BATCH_SIZE = 500
ALIGN_BATCH_SIZE = 256
REFINE_N_EPOCHS = 50
STEPS_PER_EPOCH = 100
LEARNING_RATE = 1e-3
RETRIEVAL_K = 10
N_BOOTSTRAP_ITERATIONS = 10000
RANDOM_SEED = 42
N_FOLDS = 5

GROUPS = ["Dominant", "Carnatic", "Hindustani", "Jingju", "ArabAndalusian"]
TRADITION_GROUPS = ["Carnatic", "Hindustani", "Jingju", "ArabAndalusian"]

LOCAL_ROOT = "/content/local_data_vanilla_vae_ablation"
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
    with open(src, "rb") as fsrc, open(local_filepath, "wb") as fdst, tqdm(
        total=total_size, unit="B", unit_scale=True, unit_divisor=1024,
        desc=f"[Drive] {remote_filename}",
    ) as pbar:
        while True:
            chunk = fsrc.read(chunk_size)
            if not chunk:
                break
            fdst.write(chunk)
            pbar.update(len(chunk))
    return True
# --- End Google Drive Helper Functions ---


class VanillaVAE(nn.Module):
    """Identical to Program 18's FairVAE EXCEPT: no content_aux_head, no
    style_aux_head, no auxiliary loss terms. z_c and z_s are shaped by
    reconstruction + KL + TC/HSIC alone -- statistical disentanglement
    with no semantic grounding toward known content/style targets. Same
    encoder input (concat[embedding, content features], 335-dim) as the
    real dVAE, so the content features are available to the encoder as
    raw information either way -- the ONLY difference under test is
    whether anything explicitly supervises z_c/z_s to USE them."""
    def __init__(self, encoder_input_dim, embedding_dim=EMBEDDING_DIM,
                 hidden_dim=HIDDEN_DIM_VAE, latent_dim=LATENT_DIM):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(encoder_input_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
        )
        self.content_net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU())
        self.content_mu = nn.Linear(hidden_dim, latent_dim)
        self.content_logvar = nn.Linear(hidden_dim, latent_dim)

        self.style_net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU())
        self.style_mu = nn.Linear(hidden_dim, latent_dim)
        self.style_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim), nn.ELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
        )
        # Deliberately NO content_aux_head / style_aux_head.

    def encode(self, encoder_input):
        h = self.shared(encoder_input)
        hc = self.content_net(h)
        hs = self.style_net(h)
        mu_c, logvar_c = self.content_mu(hc), self.content_logvar(hc)
        mu_s, logvar_s = self.style_mu(hs), self.style_logvar(hs)
        return mu_c, logvar_c, mu_s, logvar_s

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, encoder_input):
        mu_c, logvar_c, mu_s, logvar_s = self.encode(encoder_input)
        z_c = self.reparameterize(mu_c, logvar_c)
        z_s = self.reparameterize(mu_s, logvar_s)
        phi_recon = self.decoder(torch.cat([z_c, z_s], dim=1))
        return phi_recon, z_c, z_s, mu_c, logvar_c, mu_s, logvar_s


class RefinementNetwork(nn.Module):
    """Identical to Programs 19/20/21 -- unchanged."""
    def __init__(self, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM_REFINE):
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


def hsic_continuous(Z1, Z2):
    """HSIC between two continuous codes (TC penalty inside the dVAE)."""
    B = Z1.shape[0]
    K1 = rbf_kernel_median_heuristic(Z1)
    K2 = rbf_kernel_median_heuristic(Z2)
    H = torch.eye(B, device=Z1.device) - torch.ones(B, B, device=Z1.device) / B
    return torch.trace(K1 @ H @ K2 @ H) / ((B - 1) ** 2)


def hsic_categorical(Z, G_onehot):
    """HSIC between a continuous embedding and a categorical one-hot
    group label (refinement network's fairness penalty)."""
    B = Z.shape[0]
    K_Z = rbf_kernel_median_heuristic(Z)
    K_G = G_onehot @ G_onehot.T
    H = torch.eye(B, device=Z.device) - torch.ones(B, B, device=Z.device) / B
    return torch.trace(K_Z @ H @ K_G @ H) / ((B - 1) ** 2)


def kl_to_standard_normal(mu, logvar):
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


def stratified_batch_indices(group_to_indices, batch_size, n_groups, rng):
    per_group = batch_size // n_groups
    batch_idx = []
    for g in GROUPS:
        idx_pool = group_to_indices[g]
        replace = len(idx_pool) < per_group
        batch_idx.extend(rng.choice(idx_pool, size=per_group, replace=replace))
    return np.array(batch_idx)


def try_resume_checkpoint(remote_name, local_name):
    local_path = os.path.join(LOCAL_ROOT, local_name)
    if copy_from_drive(remote_name, local_path, skip_if_exists=True):
        checkpoint = torch.load(local_path, map_location=DEVICE, weights_only=False)
        print(f"[RESUME] Found {remote_name} at epoch {checkpoint['epoch']}. Resuming.")
        return checkpoint
    print(f"[RESUME] No checkpoint {remote_name} found. Starting fresh.")
    return None


def train_vanilla_vae(embeddings, groups, content_target):
    n = embeddings.shape[0]
    phi_t = torch.tensor(embeddings, dtype=torch.float32, device=DEVICE)

    content_mean = content_target.mean(axis=0, keepdims=True)
    content_std = content_target.std(axis=0, keepdims=True) + 1e-8
    content_t = torch.tensor((content_target - content_mean) / content_std,
                              dtype=torch.float32, device=DEVICE)
    encoder_input_t = torch.cat([phi_t, content_t], dim=1)  # 335-dim, SAME as real dVAE

    group_idx = np.array([GROUPS.index(g) for g in groups])
    group_to_indices = {g: np.where(group_idx == GROUPS.index(g))[0] for g in GROUPS}

    model = VanillaVAE(encoder_input_dim=encoder_input_t.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    rng = np.random.RandomState(RANDOM_SEED)

    start_epoch, history = 0, []
    checkpoint = try_resume_checkpoint("23_vanilla_vae_checkpoint.pt", "23_vanilla_vae_checkpoint.pt")
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        history = checkpoint["history"]

    steps_per_epoch = max(1, n // DVAE_BATCH_SIZE)
    for epoch in range(start_epoch, DVAE_N_EPOCHS):
        epoch_losses = {"recon": 0.0, "kl": 0.0, "tc": 0.0}
        for step in range(steps_per_epoch):
            batch_idx = stratified_batch_indices(group_to_indices, DVAE_BATCH_SIZE, len(GROUPS), rng)
            encoder_input_batch = encoder_input_t[batch_idx]
            phi_batch = phi_t[batch_idx]

            phi_recon, z_c, z_s, mu_c, logvar_c, mu_s, logvar_s = model(encoder_input_batch)
            l_recon = F.mse_loss(phi_recon, phi_batch)
            l_kl = kl_to_standard_normal(mu_c, logvar_c) + kl_to_standard_normal(mu_s, logvar_s)
            l_tc = hsic_continuous(z_c, z_s)
            loss = l_recon + BETA_KL * l_kl + GAMMA_TC * l_tc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses["recon"] += l_recon.item()
            epoch_losses["kl"] += l_kl.item()
            epoch_losses["tc"] += l_tc.item()

        avg_losses = {k: v / steps_per_epoch for k, v in epoch_losses.items()}
        avg_losses["epoch"] = epoch
        history.append(avg_losses)
        if epoch % 20 == 0 or epoch == DVAE_N_EPOCHS - 1:
            print(f"[VANILLA-VAE] epoch {epoch}/{DVAE_N_EPOCHS}  " +
                  "  ".join(f"{k}={v:.4f}" for k, v in avg_losses.items() if k != "epoch"))

        checkpoint_local = os.path.join(LOCAL_ROOT, "23_vanilla_vae_checkpoint.pt")
        torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch, "history": history}, checkpoint_local)
        copy_to_drive(checkpoint_local, "23_vanilla_vae_checkpoint.pt")

    model.eval()
    return model, history, content_mean, content_std


def encode_full_pool(model, embeddings, content_target, content_mean, content_std):
    content_target_std = (content_target - content_mean) / content_std
    encoder_input = np.concatenate([embeddings, content_target_std], axis=1)
    phi_t = torch.tensor(encoder_input, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        mu_c, _, mu_s, _ = model.encode(phi_t)
    return mu_c.cpu().numpy(), mu_s.cpu().numpy()


def generate_counterfactuals(model, mu_c, mu_s, groups, rng):
    """Identical bidirectional, distance-quantile donor selection to
    Program 18 -- unchanged."""
    n = mu_c.shape[0]
    group_idx = np.array([GROUPS.index(g) for g in groups])
    group_to_indices = {g: np.where(group_idx == GROUPS.index(g))[0] for g in GROUPS}

    n_augment = int(n * AUGMENTATION_RATIO)
    n_tradition_to_dominant = n_augment // 2
    n_dominant_to_tradition = n_augment - n_tradition_to_dominant

    cf_embeddings, cf_groups, source_indices = [], [], []
    mu_s_t = torch.tensor(mu_s, device=DEVICE)

    def sample_donor(src_idx, target_group):
        target_pool = group_to_indices[target_group]
        target_pool = target_pool[target_pool != src_idx]
        dists = torch.norm(mu_s_t[target_pool] - mu_s_t[src_idx], dim=1).cpu().numpy()
        low, high = np.percentile(dists, [DONOR_PERCENTILE_LOW, DONOR_PERCENTILE_HIGH])
        band_mask = (dists >= low) & (dists <= high)
        band_candidates = target_pool[band_mask]
        if len(band_candidates) == 0:
            band_candidates = target_pool
        return rng.choice(band_candidates)

    tradition_pool_all = np.concatenate([group_to_indices[g] for g in TRADITION_GROUPS])
    src_indices = rng.choice(tradition_pool_all, size=n_tradition_to_dominant, replace=True)
    for src_idx in tqdm(src_indices, desc="[CF-GEN] tradition->dominant"):
        donor_idx = sample_donor(src_idx, "Dominant")
        z_c_src = torch.tensor(mu_c[src_idx], device=DEVICE).unsqueeze(0)
        z_s_donor = torch.tensor(mu_s[donor_idx], device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            phi_cf = model.decoder(torch.cat([z_c_src, z_s_donor], dim=1)).squeeze(0).cpu().numpy()
        cf_embeddings.append(phi_cf)
        cf_groups.append(groups[src_idx])
        source_indices.append(int(src_idx))

    dominant_pool = group_to_indices["Dominant"]
    src_indices = rng.choice(dominant_pool, size=n_dominant_to_tradition, replace=True)
    target_cycle = [TRADITION_GROUPS[i % len(TRADITION_GROUPS)] for i in range(n_dominant_to_tradition)]
    for src_idx, target_group in tqdm(list(zip(src_indices, target_cycle)), desc="[CF-GEN] dominant->tradition"):
        donor_idx = sample_donor(src_idx, target_group)
        z_c_src = torch.tensor(mu_c[src_idx], device=DEVICE).unsqueeze(0)
        z_s_donor = torch.tensor(mu_s[donor_idx], device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            phi_cf = model.decoder(torch.cat([z_c_src, z_s_donor], dim=1)).squeeze(0).cpu().numpy()
        cf_embeddings.append(phi_cf)
        cf_groups.append(groups[src_idx])
        source_indices.append(int(src_idx))

    return np.stack(cf_embeddings).astype(np.float32), np.array(cf_groups), np.array(source_indices)


def train_refinement(all_embeddings, all_groups, cf_source_indices, n_original):
    """Program 19/21's fixed-scale-normalized training, UNCHANGED, at a
    single lambda_HSIC=1.5 (Program 21's sweep peak)."""
    torch.manual_seed(RANDOM_SEED)
    emb_t = torch.tensor(all_embeddings, dtype=torch.float32, device=DEVICE)
    group_idx_all = np.array([GROUPS.index(g) for g in all_groups])
    group_to_indices_all = {g: np.where(group_idx_all == GROUPS.index(g))[0] for g in GROUPS}

    n_cf = len(cf_source_indices)
    cf_global_indices = np.arange(n_original, n_original + n_cf)

    model = RefinementNetwork().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    rng = np.random.RandomState(RANDOM_SEED)

    start_epoch, history = 0, []
    align_scale, hsic_scale = None, None
    warmup_align_vals, warmup_hsic_vals = [], []
    global_step = 0
    checkpoint = try_resume_checkpoint("23_refinement_checkpoint.pt", "23_refinement_checkpoint.pt")
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

    for epoch in range(start_epoch, REFINE_N_EPOCHS):
        epoch_losses = {"align": 0.0, "hsic": 0.0}
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
                        print(f"[REFINE-ABLATION] Warmup complete: align_scale={align_scale:.6f}, "
                              f"hsic_scale={hsic_scale:.6f}")
                else:
                    align_scale_current, hsic_scale_current = align_scale, hsic_scale
                    global_step += 1

            l_align_norm = l_align / align_scale_current
            l_hsic_norm = l_hsic / hsic_scale_current
            loss = LAMBDA_ALIGN * l_align_norm + LAMBDA_HSIC_REFINE * l_hsic_norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses["align"] += align_val
            epoch_losses["hsic"] += hsic_val

        avg_losses = {k: v / STEPS_PER_EPOCH for k, v in epoch_losses.items()}
        avg_losses["epoch"] = epoch
        history.append(avg_losses)
        if epoch % 10 == 0 or epoch == REFINE_N_EPOCHS - 1:
            print(f"[REFINE-ABLATION] epoch {epoch}/{REFINE_N_EPOCHS}  "
                  f"align={avg_losses['align']:.6f}  hsic={avg_losses['hsic']:.6f}")

        checkpoint_local = os.path.join(LOCAL_ROOT, "23_refinement_checkpoint.pt")
        torch.save({
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch, "history": history, "align_scale": align_scale, "hsic_scale": hsic_scale,
            "warmup_align_vals": warmup_align_vals, "warmup_hsic_vals": warmup_hsic_vals,
            "global_step": global_step,
        }, checkpoint_local)
        copy_to_drive(checkpoint_local, "23_refinement_checkpoint.pt")

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


def compute_self_consistency(fractions, groups):
    group_idx = np.array([GROUPS.index(g) for g in groups])
    self_consistency = {}
    for g in GROUPS:
        mask = group_idx == GROUPS.index(g)
        if mask.sum() > 0:
            col = GROUPS.index(g)
            self_consistency[g] = bootstrap_mean_ci(fractions[mask, col])
    return self_consistency


def compute_intragroup_fidelity(fidelity_data, k=RETRIEVAL_K):
    results = {}
    for tradition, d in fidelity_data.items():
        embeddings, labels = d["embeddings"], d["labels"]
        n = embeddings.shape[0]
        eff_k = min(k, n - 1)
        emb_t = torch.tensor(embeddings, device=DEVICE)
        emb_norm = torch.nn.functional.normalize(emb_t, dim=1)
        sims = emb_norm @ emb_norm.T
        sims[torch.arange(n), torch.arange(n)] = -float("inf")
        topk_idx = torch.topk(sims, k=eff_k, dim=1).indices.cpu().numpy()
        per_query_precision = np.zeros(n, dtype=np.float32)
        for i in range(n):
            per_query_precision[i] = np.mean(labels[topk_idx[i]] == labels[i])
        results[tradition] = bootstrap_mean_ci(per_query_precision)
    return results


def loudness_probe_accuracy(embeddings, loudness_values):
    median_loudness = np.median(loudness_values)
    y = np.where(loudness_values >= median_loudness, "Loud", "Quiet")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    oof_preds = np.empty(len(y), dtype=object)
    for train_idx, test_idx in skf.split(embeddings, y):
        scaler = StandardScaler().fit(embeddings[train_idx])
        clf = LogisticRegression(max_iter=2000)
        clf.fit(scaler.transform(embeddings[train_idx]), y[train_idx])
        oof_preds[test_idx] = clf.predict(scaler.transform(embeddings[test_idx]))
    return accuracy_score(y, oof_preds)


def main():
    ensure_project_dir()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 2: loading pool embeddings, "
          f"auxiliary features, and fidelity embeddings...")
    pool_local = os.path.join(LOCAL_ROOT, "13_pool_embeddings.pt")
    if not copy_from_drive("13_pool_embeddings.pt", pool_local):
        raise RuntimeError("Could not fetch 13_pool_embeddings.pt.")
    pool_cache = torch.load(pool_local, map_location="cpu", weights_only=False)
    embeddings, groups = pool_cache["embeddings"], pool_cache["groups"]

    aux_local = os.path.join(LOCAL_ROOT, "18_aux_targets_cache.pt")
    if not copy_from_drive("18_aux_targets_cache.pt", aux_local):
        raise RuntimeError("Could not fetch 18_aux_targets_cache.pt.")
    aux_data = torch.load(aux_local, map_location="cpu", weights_only=False)
    content_target, loudness_values = aux_data["content_target"], aux_data["style_target"].reshape(-1)

    fidelity_local = os.path.join(LOCAL_ROOT, "13_fidelity_embeddings.pt")
    if not copy_from_drive("13_fidelity_embeddings.pt", fidelity_local):
        raise RuntimeError("Could not fetch 13_fidelity_embeddings.pt.")
    fidelity_data_vanilla = torch.load(fidelity_local, map_location="cpu", weights_only=False)

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 3: training the VanillaVAE "
          f"(no auxiliary content/style supervision)...")
    vae_model, vae_history, content_mean, content_std = train_vanilla_vae(embeddings, groups, content_target)

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 4: encoding pool + generating counterfactuals...")
    mu_c, mu_s = encode_full_pool(vae_model, embeddings, content_target, content_mean, content_std)
    rng = np.random.RandomState(RANDOM_SEED)
    cf_embeddings, cf_groups, cf_source_indices = generate_counterfactuals(vae_model, mu_c, mu_s, groups, rng)
    print(f"[CF-GEN] Generated {cf_embeddings.shape[0]} counterfactuals.")

    n_original = len(groups)
    all_embeddings = np.concatenate([embeddings, cf_embeddings], axis=0)
    all_groups = np.concatenate([groups, cf_groups])

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 5: training refinement network "
          f"(lambda_HSIC={LAMBDA_HSIC_REFINE})...")
    refine_model, refine_history = train_refinement(all_embeddings, all_groups, cf_source_indices, n_original)

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 6: validation...")
    with torch.no_grad():
        refined_pool = refine_model(torch.tensor(embeddings, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    refined_fidelity_data = {}
    for tradition, d in fidelity_data_vanilla.items():
        with torch.no_grad():
            refined_emb = refine_model(torch.tensor(d["embeddings"], dtype=torch.float32, device=DEVICE)).cpu().numpy()
        refined_fidelity_data[tradition] = {"embeddings": refined_emb, "labels": d["labels"]}

    fractions = batched_topk_group_fractions(refined_pool, groups)
    self_consistency_ablation = compute_self_consistency(fractions, groups)
    fidelity_ablation = compute_intragroup_fidelity(refined_fidelity_data)
    fidelity_vanilla_baseline = compute_intragroup_fidelity(fidelity_data_vanilla)
    loudness_acc_vanilla = loudness_probe_accuracy(embeddings, loudness_values)
    loudness_acc_ablation = loudness_probe_accuracy(refined_pool, loudness_values)

    print("[ABLATION] Self-consistency:")
    for g, r in self_consistency_ablation.items():
        print(f"    {g:16s} {r['observed']:.4f}")
    print("[ABLATION] Fidelity vs. vanilla baseline:")
    for t in fidelity_ablation:
        print(f"    {t:16s} ablation={fidelity_ablation[t]['observed']:.4f}  "
              f"vanilla={fidelity_vanilla_baseline[t]['observed']:.4f}")
    print(f"[ABLATION] Loudness accuracy: {loudness_acc_ablation:.4f} (vanilla: {loudness_acc_vanilla:.4f})")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 7: saving results and figure...")
    local_results_path = os.path.join(LOCAL_ROOT, "23_ablation_results.json")
    elapsed_seconds = time.time() - RUN_START_TIME
    output = {
        "program": "23_vanilla_vae_ablation.py",
        "lambda_hsic_refine": LAMBDA_HSIC_REFINE,
        "n_original_tracks": n_original, "n_counterfactuals": len(cf_source_indices),
        "vae_training_history_tail": vae_history[-5:],
        "refine_training_history_tail": refine_history[-5:],
        "self_consistency_ablation": self_consistency_ablation,
        "fidelity_ablation": fidelity_ablation,
        "fidelity_vanilla_baseline": fidelity_vanilla_baseline,
        "loudness_accuracy_ablation": loudness_acc_ablation,
        "loudness_accuracy_vanilla": loudness_acc_vanilla,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "elapsed_hms": time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds)),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(local_results_path, "w") as f:
        json.dump(output, f, indent=2)
    copy_to_drive(local_results_path, "23_ablation_results.json")

    traditions = list(fidelity_ablation.keys())
    x = np.arange(len(traditions))
    width = 0.3
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, [fidelity_vanilla_baseline[t]["observed"] for t in traditions],
           width, label="E4 Vanilla (E1 backbone)")
    ax.bar(x, [fidelity_ablation[t]["observed"] for t in traditions],
           width, label="Vanilla-VAE ablation (no aux supervision)")
    ax.set_xticks(x); ax.set_xticklabels(traditions)
    ax.set_ylabel("Precision@10")
    ax.set_title("Fidelity: vanilla backbone vs. VAE-without-semantic-grounding ablation")
    ax.legend()
    fig.tight_layout()
    local_fig = os.path.join(LOCAL_ROOT, "fig_23_fidelity_comparison.png")
    fig.savefig(local_fig, dpi=300)
    plt.close(fig)
    copy_to_drive(local_fig, "fig_23_fidelity_comparison.png")

    return output


if __name__ == "__main__":
    summary = main()
    elapsed = time.time() - RUN_START_TIME
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Program 23_vanilla_vae_ablation.py finished.")
    print(f"[TOTAL RUNNING TIME] {time.strftime('%H:%M:%S', time.gmtime(elapsed))} ({elapsed:.1f}s)")