# =============================================================================
# Program      : 08_e1_backbone_training.py
# Version      : 3.2
# GPU Required : YES (T4)
# Description  : Experiment 1 -- Backbone Training. Self-supervised
#                 contrastive audio embedding model (SimCLR-style),
#                 trained on the pooled catalog with NO tradition or
#                 demographic label in the loop.
#
#                 v3.0: v2.0 stored exactly ONE fixed 4-second crop per
#                 track, so all 20 epochs only ever saw different
#                 augmentation (gain/noise/masking) of the same underlying
#                 audio -- loss plateaued by epoch ~12 with only ~0.008
#                 further movement across the last 5 epochs, consistent
#                 with a model that had exhausted what this pool could
#                 teach it. v3.0 stores N_CROPS_PER_TRACK=3 crops per
#                 track at different temporal offsets (15s/60s/150s in,
#                 falling back to 0 for files shorter than that -- so
#                 short clips like FMA/Jamendo end up with near-duplicate
#                 crops, which is fine, while long concert recordings
#                 (Carnatic/Hindustani/Arab-Andalusian) get genuinely
#                 different segments). The two SimCLR views per training
#                 step are now drawn from INDEPENDENTLY sampled crop
#                 positions (same track, possibly different offset), each
#                 further augmented -- teaching temporal-position
#                 invariance in addition to production-confound
#                 invariance, a materially stronger signal for the long
#                 recordings where v2.0's single fixed crop offered no
#                 diversity to learn from at all.
#
#                 This changes the stored crop tensor's shape
#                 ([N, SEGMENT_LENGTH] -> [N, N_CROPS_PER_TRACK,
#                 SEGMENT_LENGTH]), so every v2.0 cache (7 per-source
#                 files, the pooled file, and the trained checkpoint) is a
#                 different, incompatible format. A format_version tag is
#                 now saved in every cache dict and checked on load,
#                 alongside the existing row-count floor, so this kind of
#                 incompatibility is caught automatically rather than
#                 silently loaded wrong (as the v1.0/v2.0 torchaudio.info
#                 bug was).
#
#                 v3.1 (this version) consolidates three further fixes
#                 made after v3.0's first real run:
#                   - RAM crash fix: both the per-source crop loop and the
#                     pooling step used to build a Python list and then
#                     torch.stack/cat it, which briefly holds the list AND
#                     the new tensor simultaneously (~2x peak memory --
#                     this is what crashed FMA's ~4.2GB crop set into
#                     OOM). Both now pre-allocate the final tensor and
#                     fill it in place, so peak memory never doubles.
#                   - Google Drive resume fix: download_from_drive used to open the
#                     local file in "wb" (truncate) mode on every retry,
#                     so a timeout near the end of a multi-GB transfer
#                     discarded everything already received. Now resumes
#                     by appending from the saved byte offset.
#                   - Metadata-first cache check: checking whether a
#                     cached file was stale used to require downloading
#                     the WHOLE file first (e.g. the full 2.78GB pooled
#                     cache) just to read its format_version tag. Now a
#                     tiny companion *_meta.json is checked first, so a
#                     stale cache is detected without any large download.
#
#                 CMD/HMD remain excluded (held out for audit, E4/E9).
#
# INPUT:
#                  Google Drive (7 sources):
#                    datasets/fma_metadata.zip, datasets/FMA-small.zip
#                    datasets/Saraga/saraga1.5_carnatic.zip
#                    datasets/Saraga/saraga1.5_hindustani.zip
#                    datasets/Jingju/wav_mono-1.zip
#                    datasets/Jingju/JaCRC/JaCRC-recordings.zip
#                    datasets/Arab_Andalusian/ArabAndalusianDataset.tar.gz
#                    datasets/Jamendo/mtg-jamendo-dataset/data/raw_30s_cleantags_50artists.tsv
#                    datasets/Jamendo/audio_data/ (partial: 18,486/55,609)
#
# STEPS:
#                  Step 1  Mount Drive, GPU check, ensure PROJECT_DIR
#                  Step 2  Check pooled crops cache (format + count
#                          validated); if absent/stale, process each of
#                          the 7 sources in sequence (stage -> extract
#                          N_CROPS_PER_TRACK crops/track -> cache to Google
#                          Drive -> delete raw), each itself cache-checked first
#                  Step 3  Concatenate all sources' crops, save + copy
#                          the pooled cache
#                  Step 4  Resume training from Google Drive checkpoint if present
#                  Step 5  Train: two independently-sampled crop positions
#                          per track, each augmented, NT-Xent loss,
#                          checkpoint to Google Drive after every epoch
#                  Step 6  Save final model + training-history JSON
#
# OUTPUT FILES : under PROJECT_DIR
#                  08_crops_<source>.pt        (7 files, per-source cache)
#                  08_pooled_crops.pt          (pooled cache, all sources)
#                  08_backbone_checkpoint.pt   (resumable training state)
#                  08_backbone_final.pt
#                  08_training_history.json
#
# Hyperparameters : SAMPLE_RATE=22050, SEGMENT_SECONDS=4, N_MELS=64,
#                    EMBEDDING_DIM=256, PROJECTION_DIM=128, BATCH_SIZE=64,
#                    LEARNING_RATE=3e-4, TEMPERATURE=0.5, TOTAL_EPOCHS=50,
#                    RANDOM_SEED=42, JAMENDO_SAMPLE_SIZE=8000,
#                    NUM_WORKERS=0, N_CROPS_PER_TRACK=3,
#                    CROP_OFFSET_GUESSES_SECONDS=[15, 60, 150].
# Dependencies : torch, torchaudio, pandas, tqdm,
#                 zipfile (stdlib), tarfile (stdlib)
#
# Change Log   :
#   v1.0  2026-09-16  Initial consolidated version.
#   v2.0  2026-09-16  Redesign after a disk-full crash: process one
#                      source at a time, cache compact crops to Google
#                      Drive, delete raw audio immediately after each source.
#                      Fixed num_workers>0 crashing on Colab's small
#                      /dev/shm (set to 0).
#   v2.1  2026-09-16  (in-conversation patches, not a separate file
#                      version) Fixed torchaudio.info() being unavailable
#                      in this torchaudio build, which had silently
#                      produced 0 crops for 6/7 sources; fixed a stale
#                      0-byte Google Drive-stub bug in download_from_drive; added
#                      cache-validity floors.
#   v3.0  2026-09-16  Multi-crop-per-track redesign (see Description) to
#                      fix the loss plateau observed with v2.0's single
#                      fixed crop per track. Adds format_version tagging
#                      to all caches.
#   v3.1  2026-09-16  Consolidated fixes after v3.0's first run: RAM
#                      doubling in the per-source crop loop and in pooling
#                      (pre-allocate + fill in place instead of
#                      list+stack/cat); Google Drive download resume by appending
#                      instead of truncating on every retry; metadata-
#                      first cache validity check so a stale cache is
#                      detected without downloading the full file.
#   v3.2  2026-09-16  The v3.1 pooling fix still crashed RAM: it pre-
#                      allocated the FULL final tensor (~8.7GB) upfront,
#                      then reloaded each source's tensor (up to ~4.2GB)
#                      on top of it to copy in -- peak ~13GB, over
#                      Colab's ceiling. Removed the single-pooled-tensor
#                      step entirely: SSLCropsDataset now indexes across
#                      a LIST of the 7 per-source tensors directly, never
#                      concatenated. Total resident memory is unchanged
#                      (~8.7GB, unavoidable), but nothing ever transiently
#                      overlaps on top of that anymore. Also removes the
#                      now-redundant 08_pooled_crops.pt cache layer --
#                      the 7 per-source caches already provide full
#                      resumability, as confirmed working in practice.
# Running time:
# !pip install pandas tqdm torch torchaudio
# =============================================================================

!pip install pandas tqdm torch torchaudio

import os
import sys
import time
import json
import shutil
import zipfile
import tarfile
from datetime import datetime

import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader

from google.colab import drive
drive.mount('/content/drive')

# --- GPU check ---
if not torch.cuda.is_available():
    print("\n[ERROR] GPU not detected!")
    print("This script requires a GPU")
    print("Please switch your Colab runtime to a T4 GPU and rerun.")
    sys.exit(1)
print("CUDA available: True. Proceeding...")
DEVICE = torch.device("cuda")

RUN_START_TIME = time.time()
print(f"[{datetime.now().strftime('%H:%M:%S')}] Program 08_e1_backbone_training.py (v3.0) started.")

# --- Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/sounding-fair/"

torch.manual_seed(42)

SAMPLE_RATE = 22050
SEGMENT_SECONDS = 4
SEGMENT_LENGTH = SAMPLE_RATE * SEGMENT_SECONDS
N_MELS = 64
EMBEDDING_DIM = 256
PROJECTION_DIM = 128
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
TEMPERATURE = 0.5
TOTAL_EPOCHS = 50
RANDOM_SEED = 42
JAMENDO_SAMPLE_SIZE = 8000
NUM_WORKERS = 0  # >0 crashes on Colab's small /dev/shm
N_CROPS_PER_TRACK = 3
CROP_OFFSET_GUESSES_SECONDS = [15, 60, 150]
CROP_FORMAT_VERSION = 2  # bump whenever the stored crop tensor's shape/meaning changes

MIN_EXPECTED_CROPS = {
    "fma": 1000, "carnatic": 50, "hindustani": 30, "jingju_base": 10,
    "jacrc": 50, "arab_andalusian": 50, "jamendo": 1000,
}

LOCAL_ROOT = "/content/local_data"
HISTORY_LOCAL = os.path.join(LOCAL_ROOT, "08_training_history.json")
CHECKPOINT_LOCAL = os.path.join(LOCAL_ROOT, "08_backbone_checkpoint.pt")
CHECKPOINT_REMOTE = "08_backbone_checkpoint.pt"
FINAL_MODEL_LOCAL = os.path.join(LOCAL_ROOT, "08_backbone_final.pt")
FINAL_MODEL_REMOTE = "08_backbone_final.pt"
HISTORY_REMOTE = "08_training_history.json"

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


def print_disk_usage(path="/content"):
    total, used, free = shutil.disk_usage(path)
    print(f"[DISK] {path}: {used / 1e9:.1f} GB used, {free / 1e9:.1f} GB free, "
          f"{total / 1e9:.1f} GB total.")


def upload_meta(name_prefix, format_version, extra=None):
    meta = {"format_version": format_version}
    if extra:
        meta.update(extra)
    meta_local = os.path.join(LOCAL_ROOT, f"{name_prefix}_meta.json")
    with open(meta_local, "w") as f:
        json.dump(meta, f)
    copy_to_drive(meta_local, f"{name_prefix}_meta.json")


def remote_cache_is_valid(name_prefix, min_rows=1, expected_counts=None):
    """Read only the tiny metadata file, not the full cache, to
    decide whether the full multi-GB file is even worth copying."""
    meta_local = os.path.join(LOCAL_ROOT, f"{name_prefix}_meta_check.json")
    if os.path.exists(meta_local):
        os.remove(meta_local)
    found = copy_from_drive(f"{name_prefix}_meta.json", meta_local, skip_if_exists=False, verbose=False)
    if not found:
        return False
    with open(meta_local) as f:
        meta = json.load(f)
    if meta.get("format_version") != CROP_FORMAT_VERSION:
        return False
    if expected_counts is not None:
        counts = meta.get("counts", {})
        return all(counts.get(n, 0) >= c for n, c in expected_counts.items())
    return meta.get("n_rows", 0) >= min_rows


# --- Generic staging helpers ---

def copy_from_drive_with_progress(src, dst):
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
        print(f"[COPY] {dst} already present locally, skipping copy.")
        return
    total_size = os.path.getsize(src)
    chunk_size = 1024 * 1024 * 8
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst, tqdm(
        total=total_size, unit="B", unit_scale=True, unit_divisor=1024,
        desc=f"[COPY] {os.path.basename(src)}",
    ) as pbar:
        while True:
            chunk = fsrc.read(chunk_size)
            if not chunk:
                break
            fdst.write(chunk)
            pbar.update(len(chunk))


def is_junk(name):
    if "__MACOSX" in name or name.endswith("/"):
        return True
    basename = os.path.basename(name)
    return basename == ".DS_Store" or basename.startswith("._")


def extract_zip_selective(zip_path, extract_dir, member_predicate, expected_min_files=None):
    if expected_min_files is not None and os.path.isdir(extract_dir):
        existing = sum(len(files) for _, _, files in os.walk(extract_dir))
        if existing >= expected_min_files:
            print(f"[EXTRACT] {extract_dir} already has {existing} files, skipping.")
            return
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        wanted = [m for m in zf.namelist() if member_predicate(m)]
        for member in tqdm(wanted, desc=f"[EXTRACT] {os.path.basename(zip_path)}"):
            target = os.path.join(extract_dir, member)
            if os.path.exists(target):
                continue
            zf.extract(member, extract_dir)


def extract_targz(targz_path, extract_dir, expected_min_files):
    if os.path.isdir(extract_dir):
        existing = sum(len(files) for _, _, files in os.walk(extract_dir))
        if existing >= expected_min_files:
            print(f"[EXTRACT] {extract_dir} already has {existing} files, skipping.")
            return
    os.makedirs(extract_dir, exist_ok=True)
    print(f"[EXTRACT] Extracting {targz_path} (no per-file progress for tar.gz)...")
    with tarfile.open(targz_path, "r:gz") as tf:
        tf.extractall(extract_dir, filter="data")


def find_audio_files(root_dir, extensions):
    found = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if any(fn.lower().endswith(ext) for ext in extensions):
                found.append(os.path.join(dirpath, fn))
    return found


# --- Per-source staging (returns list of raw audio paths; caller deletes
# the source's raw directory after crop conversion) ---

def stage_fma():
    root = os.path.join(LOCAL_ROOT, "fma")
    os.makedirs(root, exist_ok=True)
    meta_zip = os.path.join(root, "fma_metadata.zip")
    small_zip = os.path.join(root, "FMA-small.zip")
    copy_from_drive_with_progress("/content/drive/MyDrive/datasets/fma_metadata.zip", meta_zip)
    copy_from_drive_with_progress("/content/drive/MyDrive/datasets/FMA-small.zip", small_zip)
    extract_zip_selective(small_zip, os.path.join(root, "small_extracted"),
                           lambda n: not is_junk(n), expected_min_files=8000)
    audio_root = os.path.join(root, "small_extracted", "fma_small", "fma_small")
    if not os.path.isdir(audio_root):
        audio_root = os.path.join(root, "small_extracted")
    return find_audio_files(audio_root, [".mp3"])


def stage_carnatic():
    root = os.path.join(LOCAL_ROOT, "carnatic")
    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, "saraga1.5_carnatic.zip")
    copy_from_drive_with_progress("/content/drive/MyDrive/datasets/Saraga/saraga1.5_carnatic.zip", zip_path)

    def is_mix_or_needed(name):
        if is_junk(name):
            return False
        basename = os.path.basename(name)
        return basename.endswith(".mp3") and "multitrack" not in basename

    extract_zip_selective(zip_path, os.path.join(root, "extracted"), is_mix_or_needed,
                           expected_min_files=249)
    return find_audio_files(os.path.join(root, "extracted"), [".mp3"])


def stage_hindustani():
    root = os.path.join(LOCAL_ROOT, "hindustani")
    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, "saraga1.5_hindustani.zip")
    copy_from_drive_with_progress("/content/drive/MyDrive/datasets/Saraga/saraga1.5_hindustani.zip", zip_path)
    extract_zip_selective(zip_path, os.path.join(root, "extracted"),
                           lambda n: not is_junk(n), expected_min_files=108)
    return find_audio_files(os.path.join(root, "extracted"), [".mp3"])


def stage_jingju_base():
    root = os.path.join(LOCAL_ROOT, "jingju_base")
    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, "wav_mono-1.zip")
    copy_from_drive_with_progress("/content/drive/MyDrive/datasets/Jingju/wav_mono-1.zip", zip_path)
    extract_zip_selective(zip_path, os.path.join(root, "extracted"),
                           lambda n: not is_junk(n), expected_min_files=50)
    return find_audio_files(os.path.join(root, "extracted"), [".wav"])


def stage_jacrc():
    root = os.path.join(LOCAL_ROOT, "jacrc")
    os.makedirs(root, exist_ok=True)
    rec_zip = os.path.join(root, "JaCRC-recordings.zip")
    copy_from_drive_with_progress(
        "/content/drive/MyDrive/datasets/Jingju/JaCRC/JaCRC-recordings.zip", rec_zip)
    extract_zip_selective(rec_zip, os.path.join(root, "extracted"),
                           lambda n: not is_junk(n), expected_min_files=300)
    return find_audio_files(os.path.join(root, "extracted"), [".wav"])


def stage_arab_andalusian():
    root = os.path.join(LOCAL_ROOT, "arab_andalusian")
    os.makedirs(root, exist_ok=True)
    targz = os.path.join(root, "ArabAndalusianDataset.tar.gz")
    copy_from_drive_with_progress(
        "/content/drive/MyDrive/datasets/Arab_Andalusian/ArabAndalusianDataset.tar.gz", targz)
    extract_targz(targz, os.path.join(root, "extracted"), expected_min_files=164 * 4)
    return find_audio_files(os.path.join(root, "extracted"), [".mp3"])


def read_jamendo_tsv(path):
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        fixed_cols = header[:5]
        rows = []
        for line in f:
            parts = line.rstrip("\n").split("\t")
            fixed = parts[:5]
            fixed.append(" ".join(parts[5:]))
            rows.append(fixed)
    return pd.DataFrame(rows, columns=fixed_cols + ["TAGS"])


def stage_jamendo():
    root = os.path.join(LOCAL_ROOT, "jamendo")
    audio_dir = os.path.join(root, "audio_subset")
    os.makedirs(audio_dir, exist_ok=True)

    tsv_path = "/content/drive/MyDrive/datasets/Jamendo/mtg-jamendo-dataset/data/raw_30s_cleantags_50artists.tsv"
    audio_root = "/content/drive/MyDrive/datasets/Jamendo/audio_data"
    tsv = read_jamendo_tsv(tsv_path)

    available = set()
    for dirpath, _, filenames in os.walk(audio_root):
        folder = os.path.basename(dirpath)
        for fn in filenames:
            available.add(f"{folder}/{fn}")

    tsv_available = tsv[tsv["PATH"].isin(available)].reset_index(drop=True)
    sample = tsv_available.sample(n=min(JAMENDO_SAMPLE_SIZE, len(tsv_available)),
                                   random_state=RANDOM_SEED)

    paths = []
    for rel_path in tqdm(sample["PATH"], desc="[JAMENDO] copying"):
        src = os.path.join(audio_root, rel_path)
        dst = os.path.join(audio_dir, rel_path)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            paths.append(dst)
            continue
        if not os.path.isfile(src):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        paths.append(dst)
    return paths


SOURCES = [
    ("fma", stage_fma),
    ("carnatic", stage_carnatic),
    ("hindustani", stage_hindustani),
    ("jingju_base", stage_jingju_base),
    ("jacrc", stage_jacrc),
    ("arab_andalusian", stage_arab_andalusian),
    ("jamendo", stage_jamendo),
]


def path_to_crops(path, n_crops=N_CROPS_PER_TRACK):
    """Extract n_crops short, temporally-spread windows from one file
    without needing to know its duration in advance (torchaudio.info is
    unavailable in this build -- confirmed by direct probe). Tries a few
    fixed offset guesses (assuming a common ~44.1kHz native rate); any
    offset beyond the file's real length naturally returns 0 frames and
    falls back to offset 0. Short files (FMA/Jamendo's 30s clips) end up
    with near-duplicate crops from the 0-fallback -- harmless, since
    there isn't much temporal diversity to extract from a 30s clip
    anyway. Long files (Carnatic/Hindustani/Arab-Andalusian concert
    recordings) get genuinely different segments."""
    guess_sr = 44100
    orig_segment_guess = guess_sr * SEGMENT_SECONDS
    offsets = (CROP_OFFSET_GUESSES_SECONDS + [CROP_OFFSET_GUESSES_SECONDS[-1] + 300] *
               max(0, n_crops - len(CROP_OFFSET_GUESSES_SECONDS)))[:n_crops]

    crops = []
    for offset_s in offsets:
        offset_frames = guess_sr * offset_s
        waveform, sr = None, None
        for frame_offset in (offset_frames, 0):
            try:
                wf, s = torchaudio.load(path, frame_offset=frame_offset,
                                         num_frames=orig_segment_guess)
            except Exception:
                continue
            if wf.shape[1] > 0:
                waveform, sr = wf, s
                break
        if waveform is None:
            continue
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        if waveform.shape[1] < SEGMENT_LENGTH:
            waveform = F.pad(waveform, (0, SEGMENT_LENGTH - waveform.shape[1]))
        elif waveform.shape[1] > SEGMENT_LENGTH:
            waveform = waveform[:, :SEGMENT_LENGTH]
        waveform = torch.clamp(waveform, -1.0, 1.0)
        crops.append((waveform * 32767).to(torch.int16).squeeze(0))  # [SEGMENT_LENGTH]

    if not crops:
        return None
    while len(crops) < n_crops:
        crops.append(crops[-1].clone())  # pad by repeating the last successful crop
    return torch.stack(crops[:n_crops], dim=0)  # [n_crops, SEGMENT_LENGTH]


def process_source(name, stage_fn):
    """Full per-source pipeline: check Google Drive cache (format + count
    validated) -> stage raw audio if needed -> convert to crops -> cache
    to Google Drive -> delete raw audio."""
    crops_local = os.path.join(LOCAL_ROOT, f"crops_{name}.pt")
    crops_remote = f"08_crops_{name}.pt"

    meta_prefix = f"08_crops_{name}"
    if remote_cache_is_valid(meta_prefix, min_rows=MIN_EXPECTED_CROPS.get(name, 1)):
        if copy_from_drive(crops_remote, crops_local, skip_if_exists=True):
            data = torch.load(crops_local, map_location="cpu")
            print(f"[SOURCE] {name}: reused cached crops from Google Drive "
                  f"({data['crops'].shape[0]} tracks, {data['crops'].shape[1]} crops/track), "
                  f"no raw staging needed.")
            return data["crops"]
    else:
        print(f"[SOURCE] {name}: remote cache missing or stale (checked via lightweight "
              f"metadata -- no multi-GB download needed to find out) -- reprocessing from raw audio.")
        if os.path.exists(crops_local):
            os.remove(crops_local)

    print(f"[SOURCE] {name}: not cached (or stale), staging raw audio...")
    raw_paths = stage_fn()
    print(f"[SOURCE] {name}: {len(raw_paths)} raw audio files found, "
          f"converting to {N_CROPS_PER_TRACK} crops/track...")

    # Pre-allocate and fill in-place rather than building a Python list
    # and stacking at the end -- torch.stack briefly holds both the list
    # and the new tensor simultaneously, doubling peak memory (this is
    # what crashed FMA's ~4.2GB crop set into OOM). Failed files are left
    # as all-zero placeholder rows (silence) instead of filtered out, to
    # avoid a second doubling copy at the end -- a few silent placeholders
    # among thousands is harmless for contrastive training.
    crops_tensor = torch.zeros(len(raw_paths), N_CROPS_PER_TRACK, SEGMENT_LENGTH, dtype=torch.int16)
    n_failed = 0
    for i, path in enumerate(tqdm(raw_paths, desc=f"[CROPS] {name}")):
        c = path_to_crops(path, N_CROPS_PER_TRACK)
        if c is not None:
            crops_tensor[i] = c
        else:
            n_failed += 1
    print(f"[SOURCE] {name}: {len(raw_paths) - n_failed}/{len(raw_paths)} converted "
          f"({n_failed} left as silent placeholder rows).")

    torch.save({"source": name, "crops": crops_tensor, "format_version": CROP_FORMAT_VERSION}, crops_local)
    copy_to_drive(crops_local, crops_remote)
    upload_meta(meta_prefix, CROP_FORMAT_VERSION, {"n_rows": int(crops_tensor.shape[0])})

    source_root = os.path.join(LOCAL_ROOT, name)
    if os.path.isdir(source_root):
        shutil.rmtree(source_root)
        print(f"[CLEANUP] Removed {source_root} to free disk.")
    print_disk_usage()

    return crops_tensor


class SSLCropsDataset(Dataset):
    def __init__(self, crops_list):
        """crops_list: a LIST of per-source tensors, each
        [N_i, n_crops, SEGMENT_LENGTH] int16 -- deliberately never
        concatenated into one contiguous tensor. Building one combined
        tensor requires a pre-allocated full-size destination plus, at
        some point, a freshly reloaded source tensor coexisting with it
        (peak ~13GB for this pool, which crashed the previous version).
        Indexing across a list instead keeps total memory at exactly the
        sum of the per-source tensors (~8.7GB, unavoidable either way)
        with no transient overlap on top of that."""
        self.crops_list = crops_list
        self.n_crops = crops_list[0].shape[1]
        self.sizes = [t.shape[0] for t in crops_list]
        self.cumulative = [0]
        for s in self.sizes:
            self.cumulative.append(self.cumulative[-1] + s)
        self.total = self.cumulative[-1]
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=N_MELS)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return self.total

    def _locate(self, idx):
        # linear scan is fine with only 7 sources
        for i in range(len(self.sizes)):
            if idx < self.cumulative[i + 1]:
                return i, idx - self.cumulative[i]
        raise IndexError(idx)

    def _augment_view(self, waveform):
        gain_db = (torch.rand(1).item() * 12) - 6
        waveform = waveform * (10 ** (gain_db / 20))
        waveform = waveform + torch.randn_like(waveform) * 0.005
        mel = self.amplitude_to_db(self.mel_transform(waveform))
        t = mel.shape[-1]
        mask_len = max(1, t // 8)
        start = torch.randint(0, max(1, t - mask_len), (1,)).item()
        mel[..., start:start + mask_len] = mel.mean()
        f_dim = mel.shape[-2]
        fmask_len = max(1, f_dim // 8)
        fstart = torch.randint(0, max(1, f_dim - fmask_len), (1,)).item()
        mel[..., fstart:fstart + fmask_len, :] = mel.mean()
        return mel

    def __getitem__(self, idx):
        source_idx, local_idx = self._locate(idx)
        crops = self.crops_list[source_idx]
        # two INDEPENDENTLY sampled crop positions -- same track, possibly
        # different offset -- teaches temporal-position invariance on top
        # of the augmentation-based production-confound invariance below
        crop_idx1 = torch.randint(0, self.n_crops, (1,)).item()
        crop_idx2 = torch.randint(0, self.n_crops, (1,)).item()
        wf1 = (crops[local_idx, crop_idx1].float() / 32767.0).unsqueeze(0)
        wf2 = (crops[local_idx, crop_idx2].float() / 32767.0).unsqueeze(0)
        return self._augment_view(wf1), self._augment_view(wf2)


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


def nt_xent_loss(z, temperature):
    n = z.shape[0] // 2
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temperature
    sim.fill_diagonal_(-1e9)
    targets = torch.arange(2 * n, device=z.device)
    targets = (targets + n) % (2 * n)
    return F.cross_entropy(sim, targets)


def save_checkpoint(model, optimizer, epoch, cumulative_seconds, history):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cumulative_seconds": cumulative_seconds,
        "history": history,
        "format_version": CROP_FORMAT_VERSION,
    }, CHECKPOINT_LOCAL)
    copy_to_drive(CHECKPOINT_LOCAL, CHECKPOINT_REMOTE)


def try_resume(model, optimizer):
    found = copy_from_drive(CHECKPOINT_REMOTE, CHECKPOINT_LOCAL, skip_if_exists=False)
    if not found:
        print("[RESUME] No checkpoint found on Google Drive -- starting from epoch 0.")
        return 0, 0.0, []
    ckpt = torch.load(CHECKPOINT_LOCAL, map_location=DEVICE)
    if ckpt.get("format_version") != CROP_FORMAT_VERSION:
        print(f"[RESUME] Checkpoint format_version={ckpt.get('format_version')} does not match "
              f"current {CROP_FORMAT_VERSION} -- discarding, starting from epoch 0.")
        return 0, 0.0, []
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = ckpt["epoch"] + 1
    print(f"[RESUME] Found checkpoint at epoch {ckpt['epoch']}, resuming from epoch {start_epoch}.")
    return start_epoch, ckpt.get("cumulative_seconds", 0.0), ckpt.get("history", [])


def main():
    ensure_project_dir()
    print_disk_usage()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 2-3: assembling pooled crops "
          f"(kept as separate per-source tensors, never concatenated into one tensor -- "
          f")...")
    counts = {}
    crops_list = []
    for name, stage_fn in SOURCES:
        crops = process_source(name, stage_fn)
        counts[name] = int(crops.shape[0])
        crops_list.append(crops)

    print(f"[POOL] Per-source counts: {counts}")
    print(f"[POOL] Pooled total: {sum(counts.values())} tracks, {N_CROPS_PER_TRACK} crops/track "
          f"(held as {len(crops_list)} separate tensors).")
    print_disk_usage()

    dataset = SSLCropsDataset(crops_list)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=NUM_WORKERS, drop_last=True)

    model = AudioEncoder().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 4: checking for existing checkpoint...")
    start_epoch, cumulative_seconds, history = try_resume(model, optimizer)

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 5: training "
          f"(epochs {start_epoch + 1}-{TOTAL_EPOCHS})...")
    for epoch in range(start_epoch, TOTAL_EPOCHS):
        epoch_start_time = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] epoch={epoch + 1}/{TOTAL_EPOCHS} starting")
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{TOTAL_EPOCHS}")
        for view1, view2 in pbar:
            view1, view2 = view1.to(DEVICE), view2.to(DEVICE)
            batch = torch.cat([view1, view2], dim=0)
            optimizer.zero_grad()
            _, proj = model(batch)
            loss = nt_xent_loss(proj, TEMPERATURE)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / max(n_batches, 1)
        epoch_seconds = time.time() - epoch_start_time
        cumulative_seconds += epoch_seconds
        history.append({"epoch": epoch, "avg_loss": avg_loss, "epoch_seconds": round(epoch_seconds, 1)})
        print(f"[{datetime.now().strftime('%H:%M:%S')}] epoch={epoch + 1}/{TOTAL_EPOCHS}  "
              f"loss={avg_loss:.4f}  epoch_time={time.strftime('%H:%M:%S', time.gmtime(epoch_seconds))}  "
              f"cumulative_time={time.strftime('%H:%M:%S', time.gmtime(cumulative_seconds))}")

        save_checkpoint(model, optimizer, epoch, cumulative_seconds, history)

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 6: saving final model...")
    torch.save(model.state_dict(), FINAL_MODEL_LOCAL)
    copy_to_drive(FINAL_MODEL_LOCAL, FINAL_MODEL_REMOTE)
    with open(HISTORY_LOCAL, "w") as f:
        json.dump({"history": history, "cumulative_training_seconds": cumulative_seconds}, f, indent=2)
    copy_to_drive(HISTORY_LOCAL, HISTORY_REMOTE)

    return cumulative_seconds


if __name__ == "__main__":
    cumulative_training_seconds = main()
    elapsed = time.time() - RUN_START_TIME
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Program 08_e1_backbone_training.py finished.")
    print(f"[THIS SESSION RUNNING TIME] {time.strftime('%H:%M:%S', time.gmtime(elapsed))} ({elapsed:.1f}s)")
    print(f"[CUMULATIVE TRAINING TIME, ALL SESSIONS] "
          f"{time.strftime('%H:%M:%S', time.gmtime(cumulative_training_seconds))}")