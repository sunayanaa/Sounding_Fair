# =============================================================================
# Program      : 01_e0_dataextract_fma.py
# Version      : 1.0
# GPU Required : NO
# Description  : Experiment 0 -- Data Extraction (FMA dominant-pool corpus).
#
#                First stage of the Sounding Fair data pipeline. Copies the
#                FMA metadata and FMA-small audio archives from Google Drive
#                to local Colab disk (Drive I/O is slow and rate-limited),
#                extracts them, and builds a clean, corrected track-level
#                manifest for the FMA side of the dominant/mainstream pool.
#
#                Handles every FMA-specific gotcha identified during dataset
#                cataloging:
#                  - tracks.csv has a 3-row header where the 3rd row is
#                    blank placeholder text; parsed with header=[0,1] (the
#                    3rd row's index-name is absorbed automatically).
#                  - features.csv has a genuinely 3-level header (feature
#                    group / statistic / coefficient number); parsed with
#                    header=[0,1,2] in full -- the opposite convention from
#                    tracks.csv, not assumed to be the same.
#                  - set.subset ('small'/'medium'/'large') is an ADDITIVE
#                    tier, not a mutually-exclusive label (small=8000,
#                    medium=17000, large=81574, summing to 106574 with no
#                    overlap). Only subset == 'small' is valid as a plain
#                    equality filter; medium/large would need a cumulative
#                    .isin() filter. This program only needs 'small', where
#                    plain equality happens to be the correct case.
#                  - FMA-small.zip extracts to a doubled folder path,
#                    fma_small/fma_small/<3-digit>/<6-digit>.mp3 -- verified
#                    programmatically here rather than assumed, in case a
#                    future archive rebuild flattens it.
#
# INPUT:
#                  Google Drive:
#                    /content/drive/MyDrive/datasets/fma_metadata.zip
#                    /content/drive/MyDrive/datasets/FMA-small.zip
#
# STEPS:
#                  Step 1  Mount Drive, create PROJECT_DIR on Google Drive
#                  Step 2  Copy fma_metadata.zip + FMA-small.zip to local
#                          Colab disk (skipped if already present locally
#                          with matching size -- resumable after disconnects)
#                  Step 3  Extract both archives to local disk (skipped if
#                          expected file counts are already present)
#                  Step 4  Parse tracks.csv (header=[0,1]), filter
#                          subset == 'small', build manifest with genre_top,
#                          genres, genres_all, artist name/location, license,
#                          duration, bit_rate
#                  Step 5  Verify manifest track_ids against real mp3 files
#                          found on disk; report any mismatches
#                  Step 6  Parse features.csv (header=[0,1,2]), subset to
#                          the same 8,000 track_ids
#                  Step 7  Save manifest + features subset + run summary
#                          locally, then copy all three to Google Drive
#
# OUTPUT FILES : under PROJECT_DIR
#                  01_fma_small_manifest.csv
#                      One row per track: track_id, genre_top, genres,
#                      genres_all, artist_name, artist_location, license,
#                      duration, bit_rate, audio_relpath, audio_exists
#                  01_fma_small_features.csv
#                      features.csv rows restricted to the manifest's
#                      8,000 track_ids; the original 3-level column header
#                      is flattened to a single "group|stat|coef" string
#                      per column so the file is plain 2-D CSV.
#                  01_summary.json
#                      Row counts, mismatch counts, resolved local paths,
#                      and elapsed running time for this run.
#
# Hyperparameters : N/A -- data extraction step. No model training, no
#                    stochastic operations, nothing to seed.
# Dependencies : pandas, tqdm, zipfile (stdlib)
#
# Change Log   :
#   v1.0  2026-09-16  Initial version
# Running time: 00:06:42
# !pip install pandas tqdm
# =============================================================================

!pip install pandas tqdm

import os
import sys
import time
import json
import shutil
import zipfile
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from google.colab import drive
drive.mount('/content/drive')

# --- Run timing: note start time as early as possible ---
RUN_START_TIME = time.time()
print(f"[{datetime.now().strftime('%H:%M:%S')}] Program 01_e0_dataextract_fma.py started.")

# --- Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/sounding-fair/"

DRIVE_FMA_METADATA_ZIP = "/content/drive/MyDrive/datasets/fma_metadata.zip"
DRIVE_FMA_SMALL_ZIP = "/content/drive/MyDrive/datasets/FMA-small.zip"

LOCAL_ROOT = "/content/local_data/fma"
LOCAL_METADATA_ZIP = os.path.join(LOCAL_ROOT, "fma_metadata.zip")
LOCAL_SMALL_ZIP = os.path.join(LOCAL_ROOT, "FMA-small.zip")
LOCAL_METADATA_DIR = os.path.join(LOCAL_ROOT, "fma_metadata_extracted")
LOCAL_SMALL_DIR = os.path.join(LOCAL_ROOT, "fma_small_extracted")

EXPECTED_SMALL_TRACK_COUNT = 8000

os.makedirs(LOCAL_ROOT, exist_ok=True)

# --- Begin Google Drive Helper Functions ---
def ensure_project_dir():
    """Create PROJECT_DIR on Google Drive if it doesn't already exist.
    Must run before any copy_to_drive() call elsewhere in this
    program."""
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


def copy_from_drive_with_progress(src, dst):
    """Copy a file from Drive to local disk with a tqdm progress bar.
    Resumable across reruns: skipped entirely if dst already exists with
    the same size as src, since Drive I/O is the slow, rate-limited part
    we're trying to avoid repeating."""
    if os.path.exists(dst):
        src_size = os.path.getsize(src)
        dst_size = os.path.getsize(dst)
        if src_size == dst_size:
            print(f"[COPY] {dst} already present locally "
                  f"({dst_size / 1e9:.2f} GB), skipping copy.")
            return
        print(f"[COPY] {dst} exists but size differs from source "
              f"({dst_size} vs {src_size} bytes) -- re-copying.")

    total_size = os.path.getsize(src)
    chunk_size = 1024 * 1024 * 8  # 8 MB
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst, tqdm(
        total=total_size, unit="B", unit_scale=True, unit_divisor=1024,
        desc=f"[COPY] Drive -> local: {os.path.basename(src)}",
    ) as pbar:
        while True:
            chunk = fsrc.read(chunk_size)
            if not chunk:
                break
            fdst.write(chunk)
            pbar.update(len(chunk))


def extract_zip_with_progress(zip_path, extract_dir, expected_min_files=None):
    """Extract a zip file to extract_dir. Resumable: if extract_dir already
    contains at least expected_min_files files, extraction is skipped.
    (Zip extraction, not tar -- no filter= parameter applies here.)"""
    if expected_min_files is not None and os.path.isdir(extract_dir):
        existing_count = sum(len(files) for _, _, files in os.walk(extract_dir))
        if existing_count >= expected_min_files:
            print(f"[EXTRACT] {extract_dir} already has {existing_count} files "
                  f"(>= expected {expected_min_files}), skipping extraction.")
            return

    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        for member in tqdm(members, desc=f"[EXTRACT] {os.path.basename(zip_path)}"):
            zf.extract(member, extract_dir)


def print_disk_usage(path="/content"):
    total, used, free = shutil.disk_usage(path)
    print(f"[DISK] {path}: {used / 1e9:.1f} GB used, "
          f"{free / 1e9:.1f} GB free, {total / 1e9:.1f} GB total.")


def tid_to_audio_relpath(track_id):
    """FMA convention: zero-padded 6-digit track id, first 3 digits are
    the subfolder name. e.g. 2 -> '000/000002.mp3'."""
    tid_str = f"{int(track_id):06d}"
    return os.path.join(tid_str[:3], f"{tid_str}.mp3")


def main():
    ensure_project_dir()
    print_disk_usage()

    # --- Step 2: copy archives from Drive to local disk ---
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 2: copying archives from Drive...")
    copy_from_drive_with_progress(DRIVE_FMA_METADATA_ZIP, LOCAL_METADATA_ZIP)
    copy_from_drive_with_progress(DRIVE_FMA_SMALL_ZIP, LOCAL_SMALL_ZIP)
    print_disk_usage()

    # --- Step 3: extract archives ---
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 3: extracting archives...")
    extract_zip_with_progress(LOCAL_METADATA_ZIP, LOCAL_METADATA_DIR, expected_min_files=11)
    extract_zip_with_progress(LOCAL_SMALL_ZIP, LOCAL_SMALL_DIR, expected_min_files=EXPECTED_SMALL_TRACK_COUNT)
    print_disk_usage()

    # Locate the real audio root by checking, not assuming, whether the
    # doubled fma_small/fma_small/ nesting is present.
    candidate_roots = [
        os.path.join(LOCAL_SMALL_DIR, "fma_small", "fma_small"),
        os.path.join(LOCAL_SMALL_DIR, "fma_small"),
        LOCAL_SMALL_DIR,
    ]
    audio_root = None
    for candidate in candidate_roots:
        if os.path.isdir(candidate):
            mp3_count = sum(
                1 for _, _, files in os.walk(candidate)
                for f in files if f.endswith(".mp3")
            )
            if mp3_count > 0:
                audio_root = candidate
                print(f"[AUDIO] Resolved audio root: {audio_root} ({mp3_count} mp3 files found)")
                break
    if audio_root is None:
        print("[ERROR] Could not locate extracted FMA-small audio directory. Aborting.")
        sys.exit(1)

    # --- Step 4: parse tracks.csv ---
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 4: parsing tracks.csv...")
    tracks_csv_path = os.path.join(LOCAL_METADATA_DIR, "fma_metadata", "tracks.csv")
    if not os.path.exists(tracks_csv_path):
        alt_path = os.path.join(LOCAL_METADATA_DIR, "tracks.csv")
        tracks_csv_path = alt_path if os.path.exists(alt_path) else tracks_csv_path
    tracks = pd.read_csv(tracks_csv_path, index_col=0, header=[0, 1])

    subset_col = ("set", "subset")
    small = tracks[tracks[subset_col] == "small"].copy()
    print(f"[MANIFEST] {len(small)} tracks with subset == 'small' "
          f"(expected {EXPECTED_SMALL_TRACK_COUNT}).")
    if len(small) != EXPECTED_SMALL_TRACK_COUNT:
        print(f"[WARNING] subset=='small' count does not match the expected "
              f"{EXPECTED_SMALL_TRACK_COUNT} -- verify tracks.csv version before trusting this manifest.")

    manifest = pd.DataFrame(index=small.index)
    manifest["genre_top"] = small[("track", "genre_top")]
    manifest["genres"] = small[("track", "genres")]
    manifest["genres_all"] = small[("track", "genres_all")]
    manifest["duration"] = small[("track", "duration")]
    manifest["bit_rate"] = small[("track", "bit_rate")]
    manifest["license"] = small[("track", "license")]
    manifest["artist_name"] = small[("artist", "name")]
    manifest["artist_location"] = small[("artist", "location")]
    manifest["audio_relpath"] = [tid_to_audio_relpath(tid) for tid in manifest.index]

    # --- Step 5: verify manifest track_ids against real files on disk ---
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 5: verifying audio files on disk...")
    exists_flags = []
    for relpath in tqdm(manifest["audio_relpath"], desc="[VERIFY] checking mp3 files"):
        exists_flags.append(os.path.exists(os.path.join(audio_root, relpath)))
    manifest["audio_exists"] = exists_flags

    n_missing = int((~manifest["audio_exists"]).sum())
    if n_missing > 0:
        print(f"[WARNING] {n_missing} manifest tracks have no corresponding "
              f"mp3 file on disk under {audio_root}.")
    else:
        print(f"[VERIFY] All {len(manifest)} manifest tracks have a matching mp3 file.")

    manifest.index.name = "track_id"
    manifest = manifest.reset_index()

    # --- Step 6: parse features.csv, subset to the same track_ids ---
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 6: parsing features.csv...")
    features_csv_path = os.path.join(LOCAL_METADATA_DIR, "fma_metadata", "features.csv")
    if not os.path.exists(features_csv_path):
        alt_path = os.path.join(LOCAL_METADATA_DIR, "features.csv")
        features_csv_path = alt_path if os.path.exists(alt_path) else features_csv_path
    features = pd.read_csv(features_csv_path, index_col=0, header=[0, 1, 2])
    features_small = features.loc[features.index.isin(manifest["track_id"])].copy()
    # Flatten the 3-level column header into a single readable string per
    # column so the saved CSV is plain 2-D, not a MultiIndex round-trip.
    features_small.columns = ["|".join(map(str, c)) for c in features_small.columns]
    features_small.index.name = "track_id"
    features_small = features_small.reset_index()
    print(f"[FEATURES] {features_small.shape[0]} rows x {features_small.shape[1]} cols "
          f"(expected {len(manifest)} rows).")

    # --- Step 7: save locally, then copy to Google Drive ---
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 7: saving and copying outputs...")
    local_manifest_path = os.path.join(LOCAL_ROOT, "01_fma_small_manifest.csv")
    local_features_path = os.path.join(LOCAL_ROOT, "01_fma_small_features.csv")
    local_summary_path = os.path.join(LOCAL_ROOT, "01_summary.json")

    manifest.to_csv(local_manifest_path, index=False)
    features_small.to_csv(local_features_path, index=False)

    elapsed_seconds = time.time() - RUN_START_TIME
    summary = {
        "program": "01_e0_dataextract_fma.py",
        "manifest_rows": int(len(manifest)),
        "manifest_missing_audio": n_missing,
        "features_rows": int(features_small.shape[0]),
        "features_cols": int(features_small.shape[1]),
        "audio_root_resolved": audio_root,
        "expected_small_track_count": EXPECTED_SMALL_TRACK_COUNT,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "elapsed_hms": time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds)),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(local_summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    copy_to_drive(local_manifest_path, "01_fma_small_manifest.csv")
    copy_to_drive(local_features_path, "01_fma_small_features.csv")
    copy_to_drive(local_summary_path, "01_summary.json")

    return summary


if __name__ == "__main__":
    summary = main()
    elapsed = time.time() - RUN_START_TIME
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Program 01_e0_dataextract_fma.py finished.")
    print(f"[TOTAL RUNNING TIME] {time.strftime('%H:%M:%S', time.gmtime(elapsed))} "
          f"({elapsed:.1f} seconds)")