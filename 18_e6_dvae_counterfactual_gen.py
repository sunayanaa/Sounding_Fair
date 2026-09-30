# =============================================================================
# Program      : 18_e6_dvae_counterfactual_gen.py
# Version      : 1.4
# GPU Required : NO (small MLP-based dVAE; GPU used automatically if present)
# Description  : Experiment 6, Part 1 -- trains the disentangled VAE
#                 (dVAE) that factorizes E1's 256-dim embedding into a
#                 content code z_c and a style code z_s, then generates
#                 the counterfactual-augmented training corpus that
#                 Program 19 will use for the fairness-regularized
#                 retraining.
#
#                 Architecture (ported from the original blueprint's
#                 dVAE spec, dimension-compatible since our phi(t) is
#                 already 256-dim): shared 2-layer MLP (ReLU+LayerNorm,
#                 ->512) feeding two separate 2-layer inference networks
#                 (ELU, ->64-dim mean+logvar each) for content and style,
#                 and a 3-layer decoder (ELU+LayerNorm) reconstructing
#                 the 256-dim embedding from [z_c; z_s].
#
#                 THE key fix from the locked design, and the direct
#                 answer to the cultural-erasure risk BL5 (Program 15)
#                 empirically demonstrated: content and style are NOT
#                 left to fall out of pure statistical independence
#                 (the failure mode of a TC penalty alone -- statistical
#                 independence is not semantic disentanglement). Instead
#                 both codes get explicit auxiliary supervision:
#                   - z_c is pushed toward tonic-normalized pitch-class
#                     content + a simple rhythm summary (tempo +
#                     onset-strength stats) via an auxiliary prediction
#                     head trained to reconstruct these FROM z_c. The
#                     tonic-normalized chroma-based phi(t) reuses E2's
#                     construction (Program 09) -- recomputed here since
#                     it was never cached at pool scale, only used
#                     in-memory for E2's aggregate results.
#                   - z_s is pushed toward the loudness confound (E3's
#                     exact label, Program 11) via a similar auxiliary
#                     head trained on z_s.
#                 This directly operationalizes "content = tonic-
#                 normalized pitch-class + rhythm; style = production/
#                 recording confounds only, never instrumentation or
#                 tuning" as a supervised target, not a hope.
#
#                 The TC (total correlation) independence penalty
#                 between z_c and z_s uses HSIC rather than an
#                 adversarial TC estimator -- consistent with this
#                 whole pipeline's established preference (BL3/BL5) for
#                 closed-form, single-pass, non-adversarial dependence
#                 penalties over unstable alternatives.
#
#                 Style-donor selection uses the LOCKED fix, not the
#                 naive version that review caught as trivial: for each
#                 source track, the donor is sampled from the 40th-70th
#                 percentile band of style-space distance from the
#                 source (within the target group) -- not the single
#                 nearest neighbor, which risks an already-near-identical
#                 donor making the transfer a no-op.
#
#                 Transfer is bidirectional at augmentation ratio
#                 rho=0.15: underrepresented-tradition tracks get a
#                 Dominant-pool style donor, and an equal share of
#                 Dominant-pool tracks get a donor from one of the 4
#                 traditions (round-robin across traditions, so each
#                 gets some inbound representation).
#
#                 All inputs and outputs for this program live in
#                 PROJECT_DIR on Google Drive. There is no separate
#                 cache/mirror directory: every checkpoint, results JSON,
#                 .pt cache, and .npy produced anywhere in this pipeline
#                 resides under PROJECT_DIR, and this program reads its
#                 inputs (Program 13's pool embeddings, crop caches for
#                 auxiliary targets, its own resume checkpoint) from
#                 that same location.
#
# INPUT:
#                  Google Drive (all under PROJECT_DIR):
#                    13_pool_embeddings.pt (Program 13 -- E1's 256-dim
#                      embeddings for the full pool, tagged by group;
#                      reused directly, no re-embedding needed)
#                    08_crops_fma.pt, 08_crops_carnatic.pt,
#                    08_crops_hindustani.pt, 08_crops_jingju_base.pt,
#                    08_crops_jacrc.pt, 08_crops_arab_andalusian.pt,
#                    08_crops_jamendo.pt (for auxiliary target computation)
#
# STEPS:
#                  Step 1  Mount Drive, ensure PROJECT_DIR exists
#                  Step 2  Load Program 13's cached pool embeddings
#                          directly (no re-embedding)
#                  Step 3  Compute auxiliary targets (tonic-normalized
#                          chroma-phi(t) + rhythm for content; loudness
#                          for style) from the pool crops, via the
#                          shared-local-cache -> PROJECT_DIR fetch chain
#                  Step 4  Train the dVAE: stratified batches (5 groups),
#                          reconstruction + KL + HSIC-TC + content-aux +
#                          style-aux loss; checkpoint after every epoch
#                  Step 5  Encode the full pool through the trained dVAE
#                          to get (mu_c, mu_s) per track
#                  Step 6  Distance-quantile donor selection + bidirectional
#                          counterfactual generation at rho=0.15
#                  Step 7  Save the trained dVAE + augmented corpus
#                          (original embeddings + counterfactual
#                          embeddings + labels) to PROJECT_DIR
#
# OUTPUT FILES : under PROJECT_DIR
#                  18_dvae_checkpoint.pt     (resumable training state)
#                  18_dvae_final_model.pt    (final dVAE weights)
#                  18_augmented_corpus.pt    {embeddings, groups,
#                                             is_counterfactual}
#                  18_training_history.json
#
# Hyperparameters : EMBEDDING_DIM=256, HIDDEN_DIM=512, LATENT_DIM=64
#                    (each for z_c and z_s), GAMMA_TC=6.0 (from spec),
#                    BETA_KL=1.0 (not given an exact value in the
#                    original spec -- standard default), DELTA_CONTENT=1.0,
#                    DELTA_STYLE=1.0, AUGMENTATION_RATIO=0.15 (rho, from
#                    spec), DONOR_PERCENTILE_LOW=40, DONOR_PERCENTILE_HIGH=70
#                    (from spec), BATCH_SIZE=512 (from spec, stratified
#                    ~102/group), N_EPOCHS=100 (from spec), LR=1e-3
#                    (from spec), RANDOM_SEED=42.
# Dependencies : torch, torchaudio, librosa, numpy, pandas, tqdm
#
# Change Log   :
#   v1.0  2026-09-24  Initial version
#   v1.1  2026-09-24  First run showed content_aux converging to R^2 ~
#                      -0.48 (worse than predicting the mean) and
#                      style_aux to only R^2 ~0.10 -- z_c and z_s were
#                      not learning the intended semantic split. Most
#                      likely cause: GAMMA_TC=6.0 dominated the
#                      DELTA_CONTENT/DELTA_STYLE=1.0 auxiliary terms,
#                      making "independence via mutual emptiness" a
#                      cheaper solution than genuine disentanglement.
#                      Reduced GAMMA_TC to 1.0, raised both DELTA terms
#                      to 5.0. Also clipped the tempo feature to
#                      [40,220] BPM -- a 4s crop is a short window for
#                      tempo estimation, and an occasional wild outlier
#                      could disproportionately inflate the standardized
#                      content-target MSE. NOTE: rerunning requires
#                      deleting the existing 18_dvae_checkpoint.pt AND
#                      18_aux_targets_cache.pt (both local and Drive) --
#                      otherwise the checkpoint silently skips retraining
#                      (already at epoch 99 of 100) and the cached
#                      auxiliary targets silently keep the un-clipped
#                      tempo values, defeating both fixes without error.
#   v1.2  2026-09-24  v1.1's fix helped style_aux (R^2 0.56) but
#                      content_aux was still negative R^2 (-0.09) --
#                      loss-weighting was not the real problem for
#                      content specifically. Root cause: z_c had to
#                      RECONSTRUCT tonic-normalized content from a
#                      compressed 256-dim E1 embedding that, per E4's
#                      near-chance Carnatic raga fidelity on this same
#                      embedding, likely never reliably captured that
#                      structure in the first place -- no amount of
#                      auxiliary-loss reweighting can recover information
#                      that was already discarded upstream. Fixed by
#                      changing the ENCODER INPUT to
#                      concat[E1 embedding (256), tonic-normalized content
#                      features (79)] = 335-dim, giving z_c direct access
#                      to genuine content signal. Decoder OUTPUT stays
#                      256-dim (reconstructs only the original embedding),
#                      so counterfactual embeddings remain in the same
#                      space Program 19 expects. NOTE: this changes the
#                      model architecture itself (encoder input dim), so
#                      the existing checkpoint (from v1.1's completed run)
#                      has incompatible tensor shapes and will crash
#                      load_state_dict if not deleted first -- the cached
#                      auxiliary targets, however, are still valid and do
#                      NOT need to be deleted this time.
#   v1.4  2026-09-24  v1.3's annealing completed exactly as designed
#                      (current_beta_kl/current_gamma_tc both ramped to
#                      1.0 by epoch 20 and held) but content_aux STILL
#                      converged to exactly 1.0 -- the posterior-collapse
#                      diagnosis does not hold up: re-checking the
#                      numbers, GAMMA_TC=1.0 times an HSIC value of
#                      ~0.0003 contributes essentially nothing to the
#                      total loss, so there was never a meaningful
#                      "collapse to escape the independence penalty"
#                      incentive to anneal away from in the first place --
#                      that hypothesis was wrong. More likely culprit:
#                      content_aux_head was a SINGLE LINEAR layer mapping
#                      64-dim z_c to a 79-dim, highly heterogeneous target
#                      (MFCC/chroma/contrast/Tonnetz/rhythm mixed) --
#                      plausibly too weak to learn that mapping regardless
#                      of whether z_c carries real signal. Changed to a
#                      small 2-layer MLP. Also added direct diagnostic
#                      instrumentation (mu_c_std, logvar_c_mean, and the
#                      same for z_s) to the training history, so a THIRD
#                      failure can be diagnosed from evidence (is z_c
#                      actually flat/collapsed, or does it have real
#                      variation the old linear head just couldn't read
#                      out) rather than another blind guess. NOTE: the
#                      content_aux_head's parameters changed shape, so
#                      the v1.3 checkpoint is incompatible again and must
#                      be deleted before rerunning.
# Running time:
# !pip install torch torchaudio librosa pandas tqdm
# =============================================================================

!pip install torch torchaudio librosa pandas tqdm

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

from google.colab import drive
drive.mount('/content/drive')

RUN_START_TIME = time.time()
print(f"[{datetime.now().strftime('%H:%M:%S')}] Program 18_e6_dvae_counterfactual_gen.py started.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] Using {DEVICE} (GPU not required; used automatically if present).")

# --- Configuration ---
# PROJECT_DIR is the single source of truth for the pipeline: every
# checkpoint, results JSON, figure PNG, .pt/.npy cache, and .json
# produced by any program in this project lives here, and every program
# reads its inputs from here too. No separate cache/mirror directory
# exists.
PROJECT_DIR = "/content/drive/MyDrive/paper/sounding-fair/"

SAMPLE_RATE = 22050
N_MFCC = 13
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
LATENT_DIM = 64  # for each of z_c, z_s
GAMMA_TC = 1.0  # reduced from 6.0 -- was likely dominating and crushing both codes
BETA_KL = 1.0  # ANNEALING TARGET -- ramped up from 0, not applied at full strength from epoch 0
KL_ANNEAL_EPOCHS = 20  # linear ramp for BETA_KL and GAMMA_TC over the first 20 epochs, then held
                       # at target -- standard fix (Bowman et al. 2016) for posterior collapse:
                       # v1.2 showed content_aux converge to EXACTLY 0.9998 (the trivial
                       # predict-the-mean baseline) while style_aux reached 0.11 -- an asymmetric
                       # collapse consistent with z_c's sampled draw becoming pure noise early,
                       # since noise trivially satisfies the independence penalty and the KL
                       # term has no reason not to push logvar_c up before the auxiliary/recon
                       # losses have shaped a meaningful mean. Ramping KL and TC in gradually
                       # gives those losses room to establish a real z_c first.
DELTA_CONTENT = 5.0  # increased from 1.0 -- content_aux R^2 was negative under the old weighting
DELTA_STYLE = 5.0  # increased from 1.0 -- style_aux R^2 was only ~0.10 under the old weighting
AUGMENTATION_RATIO = 0.15
DONOR_PERCENTILE_LOW = 40
DONOR_PERCENTILE_HIGH = 70
BATCH_SIZE = 512  # ~102 per group, stratified
N_EPOCHS = 100
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

POOL_SOURCES = ["fma", "carnatic", "hindustani", "jingju_base", "jacrc", "arab_andalusian", "jamendo"]
SOURCE_TO_GROUP = {
    "fma": "Dominant", "jamendo": "Dominant",
    "carnatic": "Carnatic", "hindustani": "Hindustani",
    "jingju_base": "Jingju", "jacrc": "Jingju",
    "arab_andalusian": "ArabAndalusian",
}
GROUPS = ["Dominant", "Carnatic", "Hindustani", "Jingju", "ArabAndalusian"]
TRADITION_GROUPS = ["Carnatic", "Hindustani", "Jingju", "ArabAndalusian"]

SHARED_CACHE_DIR = "/content/shared_crops_cache"
os.makedirs(SHARED_CACHE_DIR, exist_ok=True)
LOCAL_ROOT = "/content/local_data_e6_dvae"
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


def copy_from_drive_with_progress(src, dst):
    """Copy a file from Drive to local disk with a tqdm progress bar.
    Resumable: skipped entirely if dst already exists with the same size."""
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


def fetch_file(remote_filename, shared_path):
    """Tiered fetch: shared local cache first (free, best case after any
    prior program), then PROJECT_DIR on Google Drive (the single source
    of truth for the pipeline). A file is only accepted from the shared
    cache if its size matches the Drive copy's -- a stale partial file
    from an earlier interrupted run would otherwise be silently treated
    as complete."""
    drive_path = os.path.join(PROJECT_DIR, remote_filename)
    drive_size = os.path.getsize(drive_path) if os.path.exists(drive_path) else None

    if os.path.exists(shared_path) and os.path.getsize(shared_path) > 0:
        local_size = os.path.getsize(shared_path)
        if drive_size is not None and local_size != drive_size:
            print(f"[LOCAL] {remote_filename} present but size differs from Drive -- re-copying.")
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
# --- End Google Drive Helper Functions ---


def compute_tonic_normalized_phi_and_rhythm(waveform_np, sr):
    """Content auxiliary target: tonic-normalized chroma-based phi(t)
    (same construction as Program 09's tonic-normalized variant --
    MFCC + argmax-tonic-shifted chroma + contrast + Tonnetz, mean+std
    pooled) concatenated with a simple rhythm summary (tempo + onset-
    strength mean/std)."""
    mfcc = librosa.feature.mfcc(y=waveform_np, sr=sr, n_mfcc=N_MFCC)
    chroma = librosa.feature.chroma_stft(y=waveform_np, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=waveform_np, sr=sr)

    tonic_bin = int(np.argmax(chroma.sum(axis=1)))
    chroma = np.roll(chroma, -tonic_bin, axis=0)

    tonnetz = librosa.feature.tonnetz(chroma=chroma, sr=sr)

    def pool(feat):
        return np.concatenate([feat.mean(axis=1), feat.std(axis=1)])

    phi_tonic = np.concatenate([pool(mfcc), pool(chroma), pool(contrast), pool(tonnetz)])  # 76-dim

    onset_env = librosa.onset.onset_strength(y=waveform_np, sr=sr)
    try:
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        tempo = float(np.clip(tempo, 40.0, 220.0))  # standard music tempo range -- a 4s crop
                                                     # is a short window for tempo estimation,
                                                     # clipping guards against occasional wild
                                                     # outliers dominating the standardized MSE
    except Exception:
        tempo = 0.0
    rhythm = np.array([tempo, onset_env.mean(), onset_env.std()])  # 3-dim

    return np.concatenate([phi_tonic, rhythm])  # 79-dim total


def compute_loudness_db(waveform_np):
    rms = np.sqrt(np.mean(waveform_np ** 2) + 1e-10)
    return float(20 * np.log10(rms + 1e-10))


def compute_auxiliary_targets(cache_local, cache_remote):
    if copy_from_drive(cache_remote, cache_local, skip_if_exists=True):
        cached = torch.load(cache_local, map_location="cpu", weights_only=False)
        print(f"[CACHE] Reused cached auxiliary targets: {cached['content_target'].shape[0]} tracks.")
        return cached["content_target"], cached["style_target"], cached["groups"]

    content_list, style_list, groups = [], [], []
    for name in POOL_SOURCES:
        shared_path = os.path.join(SHARED_CACHE_DIR, f"08_crops_{name}.pt")
        if not fetch_file(f"08_crops_{name}.pt", shared_path):
            print(f"[WARNING] Could not fetch 08_crops_{name}.pt -- skipping.")
            continue
        data = torch.load(shared_path, map_location="cpu", weights_only=False)
        crops = data["crops"]  # [N, 3, SEGMENT_LENGTH] int16
        group = SOURCE_TO_GROUP[name]
        for i in tqdm(range(crops.shape[0]), desc=f"[AUX-TARGETS] {name}"):
            wf = (crops[i, 0].float() / 32767.0).numpy()
            content_list.append(compute_tonic_normalized_phi_and_rhythm(wf, SAMPLE_RATE))
            style_list.append(compute_loudness_db(wf))
            groups.append(group)
        del data, crops
        print(f"[SOURCE] {name}: {len(groups)} tracks processed so far (cumulative).")

    content_target = np.stack(content_list).astype(np.float32)
    style_target = np.array(style_list, dtype=np.float32).reshape(-1, 1)
    groups = np.array(groups)

    torch.save({"content_target": content_target, "style_target": style_target, "groups": groups},
               cache_local)
    copy_to_drive(cache_local, cache_remote)
    return content_target, style_target, groups


class FairVAE(nn.Module):
    def __init__(self, encoder_input_dim, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                 latent_dim=LATENT_DIM, content_aux_dim=79, style_aux_dim=1):
        super().__init__()
        # encoder_input_dim = embedding_dim + content_aux_dim (E1's embedding
        # concatenated with the tonic-normalized content features) -- z_c
        # gets DIRECT access to genuine content signal, rather than having
        # to reconstruct it from a compressed embedding that may already
        # have discarded it (as E4's near-chance Carnatic raga fidelity on
        # this same E1 embedding suggests it largely has).
        self.shared = nn.Sequential(
            nn.Linear(encoder_input_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
        )
        self.content_net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU())
        self.content_mu = nn.Linear(hidden_dim, latent_dim)
        self.content_logvar = nn.Linear(hidden_dim, latent_dim)

        self.style_net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU())
        self.style_mu = nn.Linear(hidden_dim, latent_dim)
        self.style_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder output stays at embedding_dim (256) -- it reconstructs
        # the ORIGINAL E1 embedding only, not the concatenated input, so
        # counterfactual embeddings generated later stay in the same
        # 256-dim space as vanilla E1 embeddings for Program 19's use.
        self.decoder = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim), nn.ELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # Content head is a small 2-layer MLP, not a single linear layer:
        # content_aux stuck at EXACTLY 1.0 across two different fixes
        # (loss reweighting, KL/TC annealing) suggests the bottleneck may
        # be that a single linear layer can't learn the likely-nonlinear
        # mapping from 64-dim z_c to the 79-dim heterogeneous content
        # target (MFCC/chroma/contrast/Tonnetz/rhythm mixed together) --
        # regardless of whether z_c itself carries real signal. Style
        # head kept linear since its 1-dim target already succeeded
        # (R^2 ~0.88) with the simpler architecture.
        self.content_aux_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2), nn.ELU(),
            nn.Linear(hidden_dim // 2, content_aux_dim),
        )
        self.style_aux_head = nn.Linear(latent_dim, style_aux_dim)

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
        content_pred = self.content_aux_head(z_c)
        style_pred = self.style_aux_head(z_s)
        return phi_recon, z_c, z_s, mu_c, logvar_c, mu_s, logvar_s, content_pred, style_pred


def rbf_kernel_median_heuristic(X):
    pairwise_sq_dists = torch.cdist(X, X) ** 2
    nonzero = pairwise_sq_dists[pairwise_sq_dists > 0]
    sigma_sq = torch.median(nonzero) + 1e-8 if nonzero.numel() > 0 else torch.tensor(1.0, device=X.device)
    return torch.exp(-pairwise_sq_dists / (2 * sigma_sq))


def hsic_independence(Z1, Z2):
    """HSIC between two CONTINUOUS latent codes (both RBF kernels), used
    here as the TC (total correlation) independence penalty between
    z_c and z_s -- consistent with this pipeline's established
    preference (BL3/BL5) for closed-form, non-adversarial dependence
    penalties over an adversarial TC discriminator."""
    B = Z1.shape[0]
    K1 = rbf_kernel_median_heuristic(Z1)
    K2 = rbf_kernel_median_heuristic(Z2)
    H = torch.eye(B, device=Z1.device) - torch.ones(B, B, device=Z1.device) / B
    return torch.trace(K1 @ H @ K2 @ H) / ((B - 1) ** 2)


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


def try_resume_checkpoint():
    checkpoint_local = os.path.join(LOCAL_ROOT, "18_dvae_checkpoint.pt")
    if copy_from_drive("18_dvae_checkpoint.pt", checkpoint_local, skip_if_exists=True):
        checkpoint = torch.load(checkpoint_local, map_location=DEVICE, weights_only=False)
        print(f"[RESUME] Found checkpoint at epoch {checkpoint['epoch']}. Resuming.")
        return checkpoint
    print("[RESUME] No checkpoint found. Starting fresh.")
    return None


def save_checkpoint(model, optimizer, epoch, history):
    checkpoint_local = os.path.join(LOCAL_ROOT, "18_dvae_checkpoint.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch, "history": history,
    }, checkpoint_local)
    copy_to_drive(checkpoint_local, "18_dvae_checkpoint.pt")


def train_dvae(embeddings, groups, content_target, style_target):
    n = embeddings.shape[0]
    phi_t = torch.tensor(embeddings, dtype=torch.float32, device=DEVICE)  # 256-dim, RECON TARGET only

    content_mean = content_target.mean(axis=0, keepdims=True)
    content_std = content_target.std(axis=0, keepdims=True) + 1e-8
    content_target_std = (content_target - content_mean) / content_std
    content_t = torch.tensor(content_target_std, dtype=torch.float32, device=DEVICE)  # 79-dim

    style_mean = style_target.mean(axis=0, keepdims=True)
    style_std = style_target.std(axis=0, keepdims=True) + 1e-8
    style_target_std = (style_target - style_mean) / style_std
    style_t = torch.tensor(style_target_std, dtype=torch.float32, device=DEVICE)

    # ENCODER INPUT: E1 embedding concatenated with the (standardized)
    # content features -- gives z_c direct access to content signal
    # instead of forcing it to be reconstructed from a 256-dim embedding
    # that may already have discarded it.
    encoder_input_t = torch.cat([phi_t, content_t], dim=1)

    group_idx = np.array([GROUPS.index(g) for g in groups])
    group_to_indices = {g: np.where(group_idx == GROUPS.index(g))[0] for g in GROUPS}
    for g, idx in group_to_indices.items():
        print(f"[dVAE] Group '{g}': {len(idx)} tracks available for stratified sampling.")

    model = FairVAE(encoder_input_dim=encoder_input_t.shape[1],
                     content_aux_dim=content_target.shape[1],
                     style_aux_dim=style_target.shape[1]).to(DEVICE)
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

    steps_per_epoch = max(1, n // BATCH_SIZE)
    for epoch in range(start_epoch, N_EPOCHS):
        anneal_frac = min(1.0, epoch / KL_ANNEAL_EPOCHS)
        current_beta_kl = BETA_KL * anneal_frac
        current_gamma_tc = GAMMA_TC * anneal_frac
        epoch_losses = {"recon": 0.0, "kl": 0.0, "tc": 0.0, "content_aux": 0.0, "style_aux": 0.0,
                         "mu_c_std": 0.0, "logvar_c_mean": 0.0, "mu_s_std": 0.0, "logvar_s_mean": 0.0}
        for step in range(steps_per_epoch):
            batch_idx = stratified_batch_indices(group_to_indices, BATCH_SIZE, len(GROUPS), rng)
            encoder_input_batch = encoder_input_t[batch_idx]
            phi_batch = phi_t[batch_idx]  # 256-dim, reconstruction target
            content_batch = content_t[batch_idx]
            style_batch = style_t[batch_idx]

            phi_recon, z_c, z_s, mu_c, logvar_c, mu_s, logvar_s, content_pred, style_pred = \
                model(encoder_input_batch)

            l_recon = F.mse_loss(phi_recon, phi_batch)
            l_kl = kl_to_standard_normal(mu_c, logvar_c) + kl_to_standard_normal(mu_s, logvar_s)
            l_tc = hsic_independence(z_c, z_s)
            l_content_aux = F.mse_loss(content_pred, content_batch)
            l_style_aux = F.mse_loss(style_pred, style_batch)

            loss = (l_recon + current_beta_kl * l_kl + current_gamma_tc * l_tc
                    + DELTA_CONTENT * l_content_aux + DELTA_STYLE * l_style_aux)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Diagnostic instrumentation: directly observe whether z_c (or
            # z_s) is collapsing, rather than only inferring it from the
            # auxiliary loss. mu_*_std near 0 = the encoder outputs nearly
            # the same mean regardless of input (collapsed). logvar_*_mean
            # very high (large positive) = the sampled z is dominated by
            # reparameterization noise even if mu carries real signal.
            epoch_losses["mu_c_std"] += mu_c.std().item()
            epoch_losses["logvar_c_mean"] += logvar_c.mean().item()
            epoch_losses["mu_s_std"] += mu_s.std().item()
            epoch_losses["logvar_s_mean"] += logvar_s.mean().item()

            epoch_losses["recon"] += l_recon.item()
            epoch_losses["kl"] += l_kl.item()
            epoch_losses["tc"] += l_tc.item()
            epoch_losses["content_aux"] += l_content_aux.item()
            epoch_losses["style_aux"] += l_style_aux.item()

        avg_losses = {k: v / steps_per_epoch for k, v in epoch_losses.items()}
        avg_losses["epoch"] = epoch
        avg_losses["current_beta_kl"] = current_beta_kl
        avg_losses["current_gamma_tc"] = current_gamma_tc
        history.append(avg_losses)
        if epoch % 10 == 0 or epoch == N_EPOCHS - 1:
            print(f"[dVAE] epoch {epoch}/{N_EPOCHS}  " +
                  "  ".join(f"{k}={v:.4f}" for k, v in avg_losses.items() if k != "epoch"))
        save_checkpoint(model, optimizer, epoch, history)

    model.eval()
    return model, history, content_mean, content_std


def encode_full_pool(model, embeddings, content_target, content_mean, content_std):
    """Builds the SAME concatenated [embedding, standardized content
    features] input used during training, via the training run's own
    content_mean/content_std -- so encoding here is on identical footing
    to what the model was actually trained on, not a fresh, inconsistent
    standardization."""
    content_target_std = (content_target - content_mean) / content_std
    encoder_input = np.concatenate([embeddings, content_target_std], axis=1)
    phi_t = torch.tensor(encoder_input, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        mu_c, _, mu_s, _ = model.encode(phi_t)
    return mu_c.cpu().numpy(), mu_s.cpu().numpy()


def generate_counterfactuals(model, mu_c, mu_s, groups, rng):
    """Bidirectional distance-quantile donor selection. For a rho
    fraction of the pool, generate a counterfactual embedding using this
    track's OWN content code and a DONOR's style code, where the donor
    is sampled from the 40th-70th percentile band of style-space
    distance from the source, within the target group -- not the
    nearest neighbor, which review flagged as a trivial, near-identity
    transfer risk."""
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
            band_candidates = target_pool  # fallback: shouldn't normally happen
        return rng.choice(band_candidates)

    print(f"[CF-GEN] Tradition-to-Dominant transfers: {n_tradition_to_dominant}")
    tradition_pool_all = np.concatenate([group_to_indices[g] for g in TRADITION_GROUPS])
    src_indices = rng.choice(tradition_pool_all, size=n_tradition_to_dominant, replace=True)
    for src_idx in tqdm(src_indices, desc="[CF-GEN] tradition->dominant"):
        donor_idx = sample_donor(src_idx, "Dominant")
        z_c_src = torch.tensor(mu_c[src_idx], device=DEVICE).unsqueeze(0)
        z_s_donor = torch.tensor(mu_s[donor_idx], device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            phi_cf = model.decoder(torch.cat([z_c_src, z_s_donor], dim=1)).squeeze(0).cpu().numpy()
        cf_embeddings.append(phi_cf)
        cf_groups.append(groups[src_idx])  # retains SOURCE's group label, not donor's
        source_indices.append(int(src_idx))

    print(f"[CF-GEN] Dominant-to-tradition transfers: {n_dominant_to_tradition}")
    dominant_pool = group_to_indices["Dominant"]
    src_indices = rng.choice(dominant_pool, size=n_dominant_to_tradition, replace=True)
    target_traditions_cycle = [TRADITION_GROUPS[i % len(TRADITION_GROUPS)] for i in range(n_dominant_to_tradition)]
    for src_idx, target_group in tqdm(list(zip(src_indices, target_traditions_cycle)),
                                       desc="[CF-GEN] dominant->tradition"):
        donor_idx = sample_donor(src_idx, target_group)
        z_c_src = torch.tensor(mu_c[src_idx], device=DEVICE).unsqueeze(0)
        z_s_donor = torch.tensor(mu_s[donor_idx], device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            phi_cf = model.decoder(torch.cat([z_c_src, z_s_donor], dim=1)).squeeze(0).cpu().numpy()
        cf_embeddings.append(phi_cf)
        cf_groups.append(groups[src_idx])  # retains SOURCE's (Dominant) group label
        source_indices.append(int(src_idx))

    return np.stack(cf_embeddings).astype(np.float32), np.array(cf_groups), np.array(source_indices)


def main():
    ensure_project_dir()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 2: loading Program 13's cached pool embeddings...")
    pool_cache_local = os.path.join(LOCAL_ROOT, "13_pool_embeddings.pt")
    if not copy_from_drive("13_pool_embeddings.pt", pool_cache_local, skip_if_exists=True):
        raise RuntimeError("Could not fetch 13_pool_embeddings.pt -- run Program 13 first.")
    cached = torch.load(pool_cache_local, map_location="cpu", weights_only=False)
    embeddings, groups = cached["embeddings"], cached["groups"]
    print(f"[EMBEDDINGS] Reusing {embeddings.shape[0]} cached E1 embeddings from Program 13.")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 3: computing auxiliary content/style "
          f"targets (or reusing cache)...")
    aux_cache_local = os.path.join(LOCAL_ROOT, "18_aux_targets_cache.pt")
    content_target, style_target, aux_groups = compute_auxiliary_targets(
        aux_cache_local, "18_aux_targets_cache.pt")
    assert np.array_equal(groups, aux_groups), \
        "Group order mismatch between pool embeddings and auxiliary targets -- staging order must match."
    print(f"[AUX] content_target shape: {content_target.shape}, style_target shape: {style_target.shape}")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 4: training the dVAE...")
    model, history, content_mean, content_std = train_dvae(embeddings, groups, content_target, style_target)

    local_final_model = os.path.join(LOCAL_ROOT, "18_dvae_final_model.pt")
    torch.save(model.state_dict(), local_final_model)
    copy_to_drive(local_final_model, "18_dvae_final_model.pt")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 5: encoding the full pool...")
    mu_c, mu_s = encode_full_pool(model, embeddings, content_target, content_mean, content_std)

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 6: generating counterfactuals "
          f"(distance-quantile donor selection, rho={AUGMENTATION_RATIO})...")
    rng = np.random.RandomState(RANDOM_SEED)
    cf_embeddings, cf_groups, cf_source_indices = generate_counterfactuals(model, mu_c, mu_s, groups, rng)
    print(f"[CF-GEN] Generated {cf_embeddings.shape[0]} counterfactual embeddings.")
    print(f"[CF-GEN] Counterfactual group distribution: {pd.Series(cf_groups).value_counts().to_dict()}")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 7: saving augmented corpus...")
    all_embeddings = np.concatenate([embeddings, cf_embeddings], axis=0)
    all_groups = np.concatenate([groups, cf_groups])
    is_counterfactual = np.concatenate([np.zeros(len(groups), dtype=bool),
                                         np.ones(len(cf_groups), dtype=bool)])

    local_corpus_path = os.path.join(LOCAL_ROOT, "18_augmented_corpus.pt")
    torch.save({
        "embeddings": all_embeddings, "groups": all_groups,
        "is_counterfactual": is_counterfactual, "cf_source_indices": cf_source_indices,
    }, local_corpus_path)
    copy_to_drive(local_corpus_path, "18_augmented_corpus.pt")

    local_history_path = os.path.join(LOCAL_ROOT, "18_training_history.json")
    elapsed_seconds = time.time() - RUN_START_TIME
    output = {
        "program": "18_e6_dvae_counterfactual_gen.py",
        "n_original_tracks": int(len(groups)), "n_counterfactuals": int(len(cf_groups)),
        "augmentation_ratio": AUGMENTATION_RATIO,
        "donor_percentile_band": [DONOR_PERCENTILE_LOW, DONOR_PERCENTILE_HIGH],
        "training_history_tail": history[-5:],
        "elapsed_seconds": round(elapsed_seconds, 1),
        "elapsed_hms": time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds)),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(local_history_path, "w") as f:
        json.dump(output, f, indent=2)
    copy_to_drive(local_history_path, "18_training_history.json")

    return output


if __name__ == "__main__":
    summary = main()
    elapsed = time.time() - RUN_START_TIME
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Program 18_e6_dvae_counterfactual_gen.py finished.")
    print(f"[TOTAL RUNNING TIME] {time.strftime('%H:%M:%S', time.gmtime(elapsed))} ({elapsed:.1f}s)")