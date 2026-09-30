# =============================================================================
# Program      : 04_e0_dataextract_jingju_base.py
# Version      : 1.0
# GPU Required : NO
# Description  : Experiment 0 -- Data Extraction (Jingju base corpus:
#                 catalogue-dan.csv + catalogue-laosheng.csv + wav_mono-1.zip).
#                 JaCRC is a SEPARATE, later program -- not handled here.
#
#                 Critical parsing rule from cataloging: the 'Legend' column
#                 in each catalogue CSV is NOT a per-row decoder -- each
#                 non-empty Legend cell defines a role+shengqiang code that
#                 shows up in a DIFFERENT, later row's filename, not its
#                 own row. This program scans every non-empty Legend cell
#                 across BOTH catalogues into one lookup dict, independent
#                 of row position, rather than trying to read row N's
#                 legend as describing row N's own filename.
#
#                 Filename convention: '<role+shengqiang code>-<aria>-
#                 <play>-<source>[-T|-S#|-jh].wav'. The leading code (before
#                 the first hyphen) is looked up in the Legend dict built
#                 above. Not every code will resolve -- catalogues are not
#                 guaranteed to define every code they use; unresolved
#                 codes are reported, not guessed at.
#
#                 wav_mono-1.zip has no isolated stems -- full extraction,
#                 junk excluded.
#
# INPUT:
#                  Google Drive:
#                    /content/drive/MyDrive/datasets/Jingju/catalogue-dan.csv
#                    /content/drive/MyDrive/datasets/Jingju/catalogue-laosheng.csv
#                    /content/drive/MyDrive/datasets/Jingju/wav_mono-1.zip
#
# STEPS:
#                  Step 1  Mount Drive, ensure PROJECT_DIR exists
#                  Step 2  Copy wav_mono-1.zip to local disk (resumable)
#                  Step 3  Extract, junk excluded (resumable per-file)
#                  Step 4  Read both catalogue CSVs; build the Legend
#                          code->description lookup from every non-empty
#                          Legend cell across both files
#                  Step 5  Parse each row's leading filename code against
#                          the Legend lookup; locate matching wav file
#                          under wav_left/{laosheng,danAll}/
#                  Step 6  Save manifest + legend lookup + summary JSON
#                          locally, copy all three to Google Drive
#
# OUTPUT FILES : under PROJECT_DIR
#                  04_jingju_base_manifest.csv
#                      file_name, work, phonetically_annotated (bool),
#                      details, catalogue_source (dan/laosheng), code,
#                      code_description (from Legend, or None if
#                      unresolved), audio_relpath, audio_exists
#                  04_jingju_legend_lookup.json
#                      Every code->description pair found, for reuse in
#                      later Jingju-related programs (e.g. JaCRC)
#                  04_summary.json
#
# Hyperparameters : N/A -- data extraction step.
# Dependencies : pandas, tqdm, zipfile (stdlib)
#
# Change Log   :
#   v1.0  2026-09-16  Initial version
# Running time: 27 secs
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

RUN_START_TIME = time.time()
print(f"[{datetime.now().strftime('%H:%M:%S')}] Program 04_e0_dataextract_jingju_base.py started.")

# --- Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/sounding-fair/"

DRIVE_CATALOGUE_DAN = "/content/drive/MyDrive/datasets/Jingju/catalogue-dan.csv"
DRIVE_CATALOGUE_LAOSHENG = "/content/drive/MyDrive/datasets/Jingju/catalogue-laosheng.csv"
DRIVE_WAV_MONO_ZIP = "/content/drive/MyDrive/datasets/Jingju/wav_mono-1.zip"

LOCAL_ROOT = "/content/local_data/jingju_base"
LOCAL_ZIP = os.path.join(LOCAL_ROOT, "wav_mono-1.zip")
LOCAL_EXTRACT_DIR = os.path.join(LOCAL_ROOT, "wav_mono_extracted")

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


def copy_from_drive_with_progress(src, dst):
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
    chunk_size = 1024 * 1024 * 8
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


def is_wanted_member(name):
    if "__MACOSX" in name:
        return False
    if name.endswith("/"):
        return False
    basename = os.path.basename(name)
    if basename == ".DS_Store" or basename.startswith("._"):
        return False
    return True


def selective_extract_zip_with_progress(zip_path, extract_dir, member_predicate,
                                         expected_min_files=None):
    if expected_min_files is not None and os.path.isdir(extract_dir):
        existing_count = sum(len(files) for _, _, files in os.walk(extract_dir))
        if existing_count >= expected_min_files:
            print(f"[EXTRACT] {extract_dir} already has {existing_count} files "
                  f"(>= expected {expected_min_files}), skipping extraction.")
            return

    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        all_members = zf.namelist()
        wanted = [m for m in all_members if member_predicate(m)]
        print(f"[EXTRACT] {len(wanted)} of {len(all_members)} archive entries selected "
              f"(junk excluded).")
        n_skipped_existing = 0
        for member in tqdm(wanted, desc=f"[EXTRACT] {os.path.basename(zip_path)}"):
            target_path = os.path.join(extract_dir, member)
            if os.path.exists(target_path):
                n_skipped_existing += 1
                continue
            zf.extract(member, extract_dir)
        if n_skipped_existing:
            print(f"[EXTRACT] {n_skipped_existing} files already present, skipped re-extraction.")


def print_disk_usage(path="/content"):
    total, used, free = shutil.disk_usage(path)
    print(f"[DISK] {path}: {used / 1e9:.1f} GB used, "
          f"{free / 1e9:.1f} GB free, {total / 1e9:.1f} GB total.")


def build_legend_lookup(*dataframes):
    """Scan every non-empty 'Legend' cell across all given catalogue
    dataframes into one code->description dict. Legend cells look like
    'daeh - role-type: dan, shenqiang: er huang' -- the code is the text
    before the first ' - '."""
    lookup = {}
    for df in dataframes:
        for cell in df["Legend"].dropna():
            cell = str(cell).strip()
            if not cell:
                continue
            if " - " in cell:
                code, desc = cell.split(" - ", 1)
                lookup[code.strip()] = desc.strip()
            else:
                # malformed legend cell -- keep the raw text under itself
                # so nothing is silently dropped
                lookup[cell] = cell
    return lookup


def leading_code(file_name):
    """The role+shengqiang code is everything before the first hyphen in
    the filename (without extension)."""
    stem = os.path.splitext(file_name)[0]
    return stem.split("-", 1)[0]


def find_audio_path(wav_root, file_name_stem):
    """Search both known subfolders for a matching wav file, since the
    catalogue doesn't itself say which subfolder a row's audio lives in."""
    for subfolder in ("laosheng", "danAll"):
        candidate = os.path.join(wav_root, "wav_left", subfolder, f"{file_name_stem}.wav")
        if os.path.isfile(candidate):
            return candidate
    return None


def main():
    ensure_project_dir()
    print_disk_usage()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 2: copying wav_mono-1.zip from Drive...")
    copy_from_drive_with_progress(DRIVE_WAV_MONO_ZIP, LOCAL_ZIP)
    print_disk_usage()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 3: extracting (junk excluded)...")
    selective_extract_zip_with_progress(LOCAL_ZIP, LOCAL_EXTRACT_DIR, is_wanted_member)
    print_disk_usage()

    wav_root = LOCAL_EXTRACT_DIR
    if not os.path.isdir(os.path.join(wav_root, "wav_left")):
        # handle possible single-level extra nesting defensively
        for entry in os.listdir(wav_root):
            candidate = os.path.join(wav_root, entry)
            if os.path.isdir(os.path.join(candidate, "wav_left")):
                wav_root = candidate
                break

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 4: reading catalogues, building Legend lookup...")
    df_dan = pd.read_csv(DRIVE_CATALOGUE_DAN)
    df_dan["catalogue_source"] = "dan"
    df_laosheng = pd.read_csv(DRIVE_CATALOGUE_LAOSHENG)
    df_laosheng["catalogue_source"] = "laosheng"

    legend_lookup = build_legend_lookup(df_dan, df_laosheng)
    print(f"[LEGEND] {len(legend_lookup)} code->description pairs found across both catalogues.")

    catalogue = pd.concat([df_dan, df_laosheng], ignore_index=True)
    print(f"[CATALOGUE] {len(catalogue)} total rows ({len(df_dan)} dan + {len(df_laosheng)} laosheng).")

    # --- Step 5: resolve codes and audio paths ---
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 5: resolving codes and audio files...")
    rows = []
    n_code_resolved = 0
    n_audio_found = 0
    for _, r in tqdm(catalogue.iterrows(), total=len(catalogue), desc="[RESOLVE] rows"):
        file_name = str(r["File name"]).strip()
        code = leading_code(file_name)
        code_description = legend_lookup.get(code)
        if code_description is not None:
            n_code_resolved += 1

        audio_path = find_audio_path(wav_root, file_name)
        audio_exists = audio_path is not None
        if audio_exists:
            n_audio_found += 1
        audio_relpath = os.path.relpath(audio_path, wav_root) if audio_exists else None

        phon_annotated = str(r.get("Phonetically annotated", "")).strip() == "*"

        rows.append({
            "file_name": file_name,
            "work": r.get("Work"),
            "phonetically_annotated": phon_annotated,
            "details": r.get("Details"),
            "catalogue_source": r.get("catalogue_source"),
            "code": code,
            "code_description": code_description,
            "audio_relpath": audio_relpath,
            "audio_exists": audio_exists,
        })

    manifest = pd.DataFrame(rows)
    n = max(len(manifest), 1)
    print(f"[RESOLVE] code resolved: {n_code_resolved}/{n} "
          f"({100 * n_code_resolved / n:.1f}%)")
    print(f"[RESOLVE] audio found: {n_audio_found}/{n} "
          f"({100 * n_audio_found / n:.1f}%)")
    if n_audio_found < n:
        unresolved = manifest.loc[~manifest["audio_exists"], "file_name"].tolist()
        print(f"[WARNING] {n - n_audio_found} rows have no matching wav file. "
              f"First few: {unresolved[:5]}")
    if n_code_resolved < n:
        unresolved_codes = sorted(set(manifest.loc[manifest["code_description"].isna(), "code"]))
        print(f"[WARNING] {n - n_code_resolved} rows have an unresolved code. "
              f"Distinct unresolved codes: {unresolved_codes}")

    # --- Step 6: save and copy to Google Drive ---
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 6: saving and copying outputs...")
    local_manifest_path = os.path.join(LOCAL_ROOT, "04_jingju_base_manifest.csv")
    local_legend_path = os.path.join(LOCAL_ROOT, "04_jingju_legend_lookup.json")
    local_summary_path = os.path.join(LOCAL_ROOT, "04_summary.json")

    manifest.to_csv(local_manifest_path, index=False)
    with open(local_legend_path, "w", encoding="utf-8") as f:
        json.dump(legend_lookup, f, indent=2, ensure_ascii=False)

    elapsed_seconds = time.time() - RUN_START_TIME
    summary = {
        "program": "04_e0_dataextract_jingju_base.py",
        "manifest_rows": int(len(manifest)),
        "n_dan_rows": int(len(df_dan)),
        "n_laosheng_rows": int(len(df_laosheng)),
        "legend_entries": len(legend_lookup),
        "code_resolved": n_code_resolved,
        "audio_found": n_audio_found,
        "wav_root_resolved": wav_root,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "elapsed_hms": time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds)),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(local_summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    copy_to_drive(local_manifest_path, "04_jingju_base_manifest.csv")
    copy_to_drive(local_legend_path, "04_jingju_legend_lookup.json")
    copy_to_drive(local_summary_path, "04_summary.json")

    return summary


if __name__ == "__main__":
    summary = main()
    elapsed = time.time() - RUN_START_TIME
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Program 04_e0_dataextract_jingju_base.py finished.")
    print(f"[TOTAL RUNNING TIME] {time.strftime('%H:%M:%S', time.gmtime(elapsed))} "
          f"({elapsed:.1f} seconds)")