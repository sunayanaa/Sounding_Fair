# =============================================================================
# Program      : 07_e0_dataextract_jamendo.py
# Version      : 1.0
# GPU Required : NO
# Description  : Experiment 0 -- Data Extraction (MTG-Jamendo dominant pool,
#                 RANDOM SUBSET). Full audio_data/ on Drive is 152GB -- far
#                 too large for free-tier local disk alongside everything
#                 else already staged. This program copies only a random
#                 sample of N tracks, by individual file path, not a
#                 zip/tar extraction like Programs 01-06.
#
#                 Sampling is a plain random draw (seed=42), NOT stratified
#                 by genre. Jamendo is the dominant/mainstream pool and its
#                 natural (skewed) genre distribution needs to stay intact
#                 -- Exposure Disparity's catalog-share normalization
#                 assumes real proportions, not an artificially balanced
#                 sample (same reasoning already applied to why fma_small's
#                 8-genre balance is flagged as non-representative).
#
#                 Source manifest is raw_30s_cleantags_50artists.tsv
#                 (55,609 tracks, 3,566 artists, confirmed artist-balanced
#                 enough to need no extra correction). autotagging.tsv in
#                 the same folder is a SYMLINK to this file -- read the
#                 real file directly, not the symlink, to avoid the
#                 placeholder-text problem seen during cataloging.
#
# INPUT:
#                  Google Drive:
#                    /content/drive/MyDrive/datasets/Jamendo/mtg-jamendo-dataset/
#                      data/raw_30s_cleantags_50artists.tsv
#                    /content/drive/MyDrive/datasets/Jamendo/audio_data/
#                      (individual track files, path given by the TSV)
#
# STEPS:
#                  Step 1  Mount Drive, ensure PROJECT_DIR exists
#                  Step 2  Read the TSV, draw a fixed-seed random sample
#                          of SAMPLE_SIZE tracks
#                  Step 3  Copy each sampled track's audio file from Drive
#                          to local disk individually (resumable: a track
#                          already present locally with a non-zero size is
#                          not re-copied)
#                  Step 4  Build manifest: track_id, artist_id, album_id,
#                          duration, tags, local_relpath, audio_exists
#                  Step 5  Save manifest + summary JSON locally, copy
#                          both to Google Drive
#
# OUTPUT FILES : under PROJECT_DIR
#                  07_jamendo_subset_manifest.csv
#                  07_summary.json
#
# Hyperparameters : SAMPLE_SIZE=15000, RANDOM_SEED=42
# Dependencies : pandas, tqdm
#
# Change Log   :
#   v1.0  2026-09-16  Initial version
# Running time: 2 hrs
# !pip install pandas tqdm
# =============================================================================

!pip install pandas tqdm

import os
import sys
import time
import json
import shutil
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from google.colab import drive
drive.mount('/content/drive')

RUN_START_TIME = time.time()
print(f"[{datetime.now().strftime('%H:%M:%S')}] Program 07_e0_dataextract_jamendo.py started.")

# --- Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/sounding-fair/"

DRIVE_JAMENDO_TSV = "/content/drive/MyDrive/datasets/Jamendo/mtg-jamendo-dataset/data/raw_30s_cleantags_50artists.tsv"
DRIVE_AUDIO_ROOT = "/content/drive/MyDrive/datasets/Jamendo/audio_data"

LOCAL_ROOT = "/content/local_data/jamendo"
LOCAL_AUDIO_DIR = os.path.join(LOCAL_ROOT, "audio_subset")

SAMPLE_SIZE = 8000
RANDOM_SEED = 42

os.makedirs(LOCAL_ROOT, exist_ok=True)
os.makedirs(LOCAL_AUDIO_DIR, exist_ok=True)

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
    print(f"[DISK] {path}: {used / 1e9:.1f} GB used, "
          f"{free / 1e9:.1f} GB free, {total / 1e9:.1f} GB total.")

def scan_available_paths(audio_root):
    """One-time metadata-only walk of audio_data to find which relative
    paths actually exist on Drive. audio_data was found to be a PARTIAL
    download (18,486 of the full 55,609-track catalog) -- this avoids
    wasting a ~0.5s Drive round-trip on every guaranteed-to-fail lookup."""
    available = set()
    for dirpath, _, filenames in os.walk(audio_root):
        folder = os.path.basename(dirpath)
        for fn in filenames:
            available.add(f"{folder}/{fn}")
    return available


def scan_already_local(local_audio_dir):
    """Relative paths already copied locally from a previous (interrupted
    or partial) run -- these cost nothing to keep."""
    existing = set()
    for dirpath, _, filenames in os.walk(local_audio_dir):
        folder = os.path.basename(dirpath)
        for fn in filenames:
            if os.path.getsize(os.path.join(dirpath, fn)) > 0:
                existing.add(f"{folder}/{fn}")
    return existing

def copy_one_file(src, dst):
    """Individual small-file copy, no progress bar (would be too noisy at
    15,000 calls) -- overall progress is tracked by the outer tqdm loop
    instead. Resumable: skipped if dst already exists with size > 0."""
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return True
    if not os.path.isfile(src):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    return True

def read_jamendo_tsv(path):
    """MTG-Jamendo's raw TSV is not fixed-width: each additional tag past
    the first is its own extra tab-separated field, so row length varies
    with tag count. Read manually and fold everything past the 5 fixed
    columns into one space-joined TAGS string."""
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        fixed_cols = header[:5]
        rows = []
        for line in f:
            parts = line.rstrip("\n").split("\t")
            fixed = parts[:5]
            tags = parts[5:]
            fixed.append(" ".join(tags))
            rows.append(fixed)
    return pd.DataFrame(rows, columns=fixed_cols + ["TAGS"])

def main():
    ensure_project_dir()
    print_disk_usage()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 2: reading manifest and "
          f"scanning what's actually available/already copied...")
    tsv = read_jamendo_tsv(DRIVE_JAMENDO_TSV)
    print(f"[SAMPLE] Full manifest: {len(tsv)} tracks.")

    available_paths = scan_available_paths(DRIVE_AUDIO_ROOT)
    print(f"[SAMPLE] {len(available_paths)} of {len(tsv)} tracks actually present under audio_data "
          f"(partial download confirmed).")

    already_local_paths = scan_already_local(LOCAL_AUDIO_DIR)
    print(f"[SAMPLE] {len(already_local_paths)} tracks already copied locally from a prior run.")

    tsv_available = tsv[tsv["PATH"].isin(available_paths)].reset_index(drop=True)

    already_local_rows = tsv_available[tsv_available["PATH"].isin(already_local_paths)]
    remaining_pool = tsv_available[~tsv_available["PATH"].isin(already_local_paths)]

    n_already = len(already_local_rows)
    n_needed = max(0, SAMPLE_SIZE - n_already)
    print(f"[SAMPLE] {n_already} usable rows already local, need {n_needed} more to reach "
          f"target {SAMPLE_SIZE}.")

    newly_sampled = remaining_pool.sample(
        n=min(n_needed, len(remaining_pool)), random_state=RANDOM_SEED
    ) if n_needed > 0 else remaining_pool.iloc[0:0]

    sample = pd.concat([already_local_rows, newly_sampled], ignore_index=True)
    print(f"[SAMPLE] Final sample: {len(sample)} tracks "
          f"({n_already} reused + {len(newly_sampled)} newly selected).")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 3: copying newly selected audio files "
          f"({len(newly_sampled)} to copy; {n_already} already on disk, skipped entirely)...")
    n_copied = 0
    n_missing_source = 0
    local_relpaths = []
    for _, r in tqdm(sample.iterrows(), total=len(sample), desc="[COPY] Jamendo subset"):
        rel_path = r["PATH"]
        src = os.path.join(DRIVE_AUDIO_ROOT, rel_path)
        dst = os.path.join(LOCAL_AUDIO_DIR, rel_path)
        ok = copy_one_file(src, dst)
        if ok:
            n_copied += 1
            local_relpaths.append(rel_path)
        else:
            n_missing_source += 1
            local_relpaths.append(None)
    print(f"[COPY] {n_copied}/{len(sample)} present on disk after this run "
          f"({n_missing_source} unexpectedly missing -- should be 0 since both pools were "
          f"pre-filtered to confirmed-available paths).")
    print_disk_usage()


    # --- Step 4: build manifest ---
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 4: building manifest...")
    sample["local_relpath"] = local_relpaths
    sample["audio_exists"] = sample["local_relpath"].notna() & sample["local_relpath"].apply(
        lambda p: os.path.isfile(os.path.join(LOCAL_AUDIO_DIR, p)) if p else False
    )
    manifest = sample.rename(columns={
        "TRACK_ID": "track_id", "ARTIST_ID": "artist_id", "ALBUM_ID": "album_id",
        "PATH": "drive_relpath", "DURATION": "duration", "TAGS": "tags",
    })

    # --- Step 5: save and copy to Google Drive ---
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 5: saving and copying outputs...")
    local_manifest_path = os.path.join(LOCAL_ROOT, "07_jamendo_subset_manifest.csv")
    local_summary_path = os.path.join(LOCAL_ROOT, "07_summary.json")

    manifest.to_csv(local_manifest_path, index=False)

    elapsed_seconds = time.time() - RUN_START_TIME
    summary = {
        "program": "07_e0_dataextract_jamendo.py",
        "sample_size_requested": SAMPLE_SIZE,
        "random_seed": RANDOM_SEED,
        "manifest_rows": int(len(manifest)),
        "audio_copied": n_copied,
        "audio_missing_source": n_missing_source,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "elapsed_hms": time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds)),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(local_summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    copy_to_drive(local_manifest_path, "07_jamendo_subset_manifest.csv")
    copy_to_drive(local_summary_path, "07_summary.json")

    return summary


if __name__ == "__main__":
    summary = main()
    elapsed = time.time() - RUN_START_TIME
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Program 07_e0_dataextract_jamendo.py finished.")
    print(f"[TOTAL RUNNING TIME] {time.strftime('%H:%M:%S', time.gmtime(elapsed))} "
          f"({elapsed:.1f} seconds)")