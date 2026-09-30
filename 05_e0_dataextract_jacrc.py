# =============================================================================
# Program      : 05_e0_dataextract_jacrc.py
# Version      : 1.0
# GPU Required : NO
# Description  : Experiment 0 -- Data Extraction (JaCRC: Jingju a Cappella
#                 Recordings Collection). Separate from the base Jingju
#                 corpus (Program 04).
#
#                 Known structure from cataloging:
#                   - 3 physical audio locations under JaCRC-recordings/:
#                     top level (239 main-pool), .../JaCRC-students-recordings/
#                     (75), .../JaCRC-accompaniment-recordings/ (76 jinghu).
#                   - 15 "commercial" rows in the main info CSV have NO
#                     audio at all (metadata/segmentation only) -- this
#                     program does not hardcode a 'source == commercial'
#                     check; audio_exists is determined by actually looking
#                     for the file, which naturally yields False for these
#                     15 without guessing the exact source-string casing.
#                   - Both info CSVs are SEMICOLON-separated, not comma.
#                   - 'role type' values are finer-grained than the 5 top-
#                     level categories (e.g. 'nandan' seen in real data) --
#                     stored RAW here, not normalized. Normalizing to a
#                     clean 5-class set is a metric-computation decision
#                     left for the E4 script, same principle as Program
#                     02's Ragamalika handling.
#                   - Accompaniment pairing (main CSV's accompaniment=='yes'
#                     rows) is resolved by matching the first 3 hyphen-
#                     separated filename tokens (role+shengqiang code, aria
#                     prefix, play title) against the accompaniment folder,
#                     NOT by guessing an exact '-jh' suffix construction we
#                     never confirmed. Match count (0/1/>1) is recorded
#                     explicitly so an ambiguous or failed pairing is
#                     visible, not silently wrong.
#                   - Teacher flag ('-T' as the final filename token) and
#                     student number ('-S<n>' as the final token) are
#                     parsed from the filename per the documented naming
#                     convention.
#
# INPUT:
#                  Google Drive:
#                    /content/drive/MyDrive/datasets/Jingju/JaCRC/JaCRC-recordings.zip
#                    /content/drive/MyDrive/datasets/Jingju/JaCRC/JaCRC-annotations.zip
#                    /content/drive/MyDrive/datasets/Jingju/JaCRC/JaCRC-recordings-info.csv
#                    /content/drive/MyDrive/datasets/Jingju/JaCRC/JaCRC-students-recordings-info.csv
#
# STEPS:
#                  Step 1  Mount Drive, ensure PROJECT_DIR exists
#                  Step 2  Copy both zip archives to local disk (resumable)
#                  Step 3  Extract both, junk excluded (resumable per-file)
#                  Step 4  Read both info CSVs (semicolon-separated)
#                  Step 5  Build manifest: locate audio for main-pool and
#                          student rows, resolve accompaniment pairing,
#                          parse teacher/student-number flags
#                  Step 6  Save manifest + summary JSON locally, copy
#                          both to Google Drive
#
# OUTPUT FILES : under PROJECT_DIR
#                  05_jacrc_manifest.csv
#                  05_summary.json
#
# Hyperparameters : N/A -- data extraction step.
# Dependencies : pandas, tqdm, zipfile (stdlib), re (stdlib)
#
# Change Log   :
#   v1.0  2026-09-16  Initial version
# Running time: 254.7 secs
# !pip install pandas tqdm
# =============================================================================

!pip install pandas tqdm

import os
import re
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
print(f"[{datetime.now().strftime('%H:%M:%S')}] Program 05_e0_dataextract_jacrc.py started.")

# --- Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/sounding-fair/"

JACRC_DIR = "/content/drive/MyDrive/datasets/Jingju/JaCRC"
DRIVE_RECORDINGS_ZIP = os.path.join(JACRC_DIR, "JaCRC-recordings.zip")
DRIVE_ANNOTATIONS_ZIP = os.path.join(JACRC_DIR, "JaCRC-annotations.zip")
DRIVE_MAIN_INFO_CSV = os.path.join(JACRC_DIR, "JaCRC-recordings-info.csv")
DRIVE_STUDENT_INFO_CSV = os.path.join(JACRC_DIR, "JaCRC-students-recordings-info.csv")

LOCAL_ROOT = "/content/local_data/jacrc"
LOCAL_RECORDINGS_ZIP = os.path.join(LOCAL_ROOT, "JaCRC-recordings.zip")
LOCAL_ANNOTATIONS_ZIP = os.path.join(LOCAL_ROOT, "JaCRC-annotations.zip")
LOCAL_RECORDINGS_DIR = os.path.join(LOCAL_ROOT, "recordings_extracted")
LOCAL_ANNOTATIONS_DIR = os.path.join(LOCAL_ROOT, "annotations_extracted")

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


def prefix_key(stem):
    """First 3 hyphen-separated tokens: role+shengqiang code, aria prefix,
    play title -- the stable portion of the naming convention shared by a
    vocal recording and its jinghu accompaniment."""
    parts = stem.split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else stem


def find_accompaniment(accompaniment_dir, vocal_stem):
    if not os.path.isdir(accompaniment_dir):
        return None, 0
    key = prefix_key(vocal_stem)
    candidates = [f for f in os.listdir(accompaniment_dir)
                  if prefix_key(os.path.splitext(f)[0]) == key]
    if len(candidates) == 1:
        return os.path.join(accompaniment_dir, candidates[0]), 1
    return None, len(candidates)


def parse_suffix_flags(stem):
    """Teacher flag: final token == 'T'. Student number: final token
    matches 'S<digits>' at the start (may have a trailing take number in
    brackets, e.g. 'S1(2)', which the regex still captures correctly)."""
    last = stem.split("-")[-1]
    is_teacher = (last == "T")
    m = re.match(r"^S(\d+)", last)
    student_number = int(m.group(1)) if m else None
    return is_teacher, student_number


def main():
    ensure_project_dir()
    print_disk_usage()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 2: copying archives from Drive...")
    copy_from_drive_with_progress(DRIVE_RECORDINGS_ZIP, LOCAL_RECORDINGS_ZIP)
    copy_from_drive_with_progress(DRIVE_ANNOTATIONS_ZIP, LOCAL_ANNOTATIONS_ZIP)
    print_disk_usage()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 3: extracting (junk excluded)...")
    selective_extract_zip_with_progress(LOCAL_RECORDINGS_ZIP, LOCAL_RECORDINGS_DIR, is_wanted_member)
    selective_extract_zip_with_progress(LOCAL_ANNOTATIONS_ZIP, LOCAL_ANNOTATIONS_DIR, is_wanted_member)
    print_disk_usage()

    recordings_root = LOCAL_RECORDINGS_DIR
    if not os.path.isdir(os.path.join(recordings_root, "JaCRC-recordings")):
        for entry in os.listdir(recordings_root):
            candidate = os.path.join(recordings_root, entry)
            if os.path.isdir(candidate) and "JaCRC-recordings" in os.listdir(candidate):
                pass
    main_pool_dir = None
    student_dir = None
    accompaniment_dir = None
    for dirpath, dirnames, _ in os.walk(recordings_root):
        if os.path.basename(dirpath) == "JaCRC-recordings":
            main_pool_dir = dirpath
        if os.path.basename(dirpath) == "JaCRC-students-recordings":
            student_dir = dirpath
        if os.path.basename(dirpath) == "JaCRC-accompaniment-recordings":
            accompaniment_dir = dirpath
    print(f"[PATHS] main_pool_dir={main_pool_dir}")
    print(f"[PATHS] student_dir={student_dir}")
    print(f"[PATHS] accompaniment_dir={accompaniment_dir}")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 4: reading info CSVs (semicolon-separated)...")
    df_main = pd.read_csv(DRIVE_MAIN_INFO_CSV, sep=";")
    df_main["pool"] = "main"
    df_student = pd.read_csv(DRIVE_STUDENT_INFO_CSV, sep=";")
    df_student["pool"] = "student"
    print(f"[CSV] main info: {len(df_main)} rows, student info: {len(df_student)} rows.")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 5: building manifest...")
    rows = []
    n_audio_found = 0
    n_accompaniment_resolved = 0
    n_accompaniment_ambiguous = 0
    n_teacher = 0
    n_student_numbered = 0

    for _, r in tqdm(df_main.iterrows(), total=len(df_main), desc="[MANIFEST] main pool"):
        file_name = str(r["file name"]).strip()
        stem = os.path.splitext(file_name)[0]
        audio_path = os.path.join(main_pool_dir, file_name) if main_pool_dir else None
        audio_exists = audio_path is not None and os.path.isfile(audio_path)
        if audio_exists:
            n_audio_found += 1
        audio_relpath = os.path.relpath(audio_path, recordings_root) if audio_exists else None

        accompaniment_relpath, acc_match_count = None, 0
        if str(r.get("accompaniment", "")).strip().lower() == "yes" and accompaniment_dir:
            acc_path, acc_match_count = find_accompaniment(accompaniment_dir, stem)
            if acc_path:
                accompaniment_relpath = os.path.relpath(acc_path, recordings_root)
                n_accompaniment_resolved += 1
            elif acc_match_count > 1:
                n_accompaniment_ambiguous += 1

        is_teacher, student_number = parse_suffix_flags(stem)
        if is_teacher:
            n_teacher += 1
        if student_number is not None:
            n_student_numbered += 1

        rows.append({
            "file_name": file_name, "pool": "main", "source": r.get("source"),
            "role_type_raw": r.get("role type"), "shengqiang": r.get("shengqiang"),
            "banshi": r.get("banshi"), "work": r.get("work"), "artist": r.get("artist"),
            "level": r.get("level"), "accompaniment_flag": r.get("accompaniment"),
            "date": r.get("date"), "recording_setting": r.get("recording setting"),
            "related_score": r.get("related score"), "note": r.get("note"),
            "textgrid": r.get("TextGrid"), "phrase": r.get("phrase"),
            "phrase_zh": r.get("phrase_ZH"), "syllable": r.get("syllable"),
            "phoneme": r.get("phoneme"), "lyrics": r.get("lyrics"),
            "is_teacher": is_teacher, "student_number": student_number,
            "audio_relpath": audio_relpath, "audio_exists": audio_exists,
            "accompaniment_relpath": accompaniment_relpath,
            "accompaniment_match_count": acc_match_count,
        })

    for _, r in tqdm(df_student.iterrows(), total=len(df_student), desc="[MANIFEST] students"):
        file_name = str(r["file name"]).strip()
        stem = os.path.splitext(file_name)[0]
        audio_path = os.path.join(student_dir, file_name) if student_dir else None
        audio_exists = audio_path is not None and os.path.isfile(audio_path)
        if audio_exists:
            n_audio_found += 1
        audio_relpath = os.path.relpath(audio_path, recordings_root) if audio_exists else None

        is_teacher, student_number = parse_suffix_flags(stem)
        if is_teacher:
            n_teacher += 1
        if student_number is not None:
            n_student_numbered += 1

        rows.append({
            "file_name": file_name, "pool": "student", "source": r.get("source"),
            "role_type_raw": r.get("role type"), "shengqiang": r.get("shengqiang"),
            "banshi": r.get("banshi"), "work": r.get("work"), "artist": r.get("artist"),
            "level": None, "accompaniment_flag": None,
            "date": r.get("date"), "recording_setting": r.get("recording setting"),
            "related_score": r.get("related score"), "note": None,
            "textgrid": r.get("TextGrid"), "phrase": r.get("phrase"),
            "phrase_zh": r.get("phrase_ZH"), "syllable": r.get("syllable"),
            "phoneme": r.get("phoneme"), "lyrics": None,
            "is_teacher": is_teacher, "student_number": student_number,
            "audio_relpath": audio_relpath, "audio_exists": audio_exists,
            "accompaniment_relpath": None, "accompaniment_match_count": 0,
        })

    manifest = pd.DataFrame(rows)
    n = len(manifest)
    n_expected_accompaniment = int(
        (df_main.get("accompaniment", pd.Series(dtype=str)).astype(str).str.strip().str.lower() == "yes").sum()
    )
    role_type_counts = manifest["role_type_raw"].value_counts(dropna=False).to_dict()

    print(f"[SUMMARY] manifest rows: {n} ({len(df_main)} main + {len(df_student)} student)")
    print(f"[SUMMARY] audio found: {n_audio_found}/{n} ({100 * n_audio_found / max(n,1):.1f}%) "
          f"-- 15-ish commercial-source main rows are expected to have no audio")
    print(f"[SUMMARY] accompaniment expected (flag=='yes'): {n_expected_accompaniment}, "
          f"resolved: {n_accompaniment_resolved}, ambiguous(>1 match): {n_accompaniment_ambiguous}")
    print(f"[SUMMARY] is_teacher: {n_teacher}, student_number parsed: {n_student_numbered}")
    print(f"[SUMMARY] distinct role_type_raw values: {role_type_counts}")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 6: saving and copying outputs...")
    local_manifest_path = os.path.join(LOCAL_ROOT, "05_jacrc_manifest.csv")
    local_summary_path = os.path.join(LOCAL_ROOT, "05_summary.json")

    manifest.to_csv(local_manifest_path, index=False)

    elapsed_seconds = time.time() - RUN_START_TIME
    summary = {
        "program": "05_e0_dataextract_jacrc.py",
        "manifest_rows": n,
        "main_rows": int(len(df_main)),
        "student_rows": int(len(df_student)),
        "audio_found": n_audio_found,
        "accompaniment_expected": n_expected_accompaniment,
        "accompaniment_resolved": n_accompaniment_resolved,
        "accompaniment_ambiguous": n_accompaniment_ambiguous,
        "is_teacher_count": n_teacher,
        "student_number_parsed": n_student_numbered,
        "role_type_raw_counts": {str(k): int(v) for k, v in role_type_counts.items()},
        "elapsed_seconds": round(elapsed_seconds, 1),
        "elapsed_hms": time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds)),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(local_summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    copy_to_drive(local_manifest_path, "05_jacrc_manifest.csv")
    copy_to_drive(local_summary_path, "05_summary.json")

    return summary


if __name__ == "__main__":
    summary = main()
    elapsed = time.time() - RUN_START_TIME
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Program 05_e0_dataextract_jacrc.py finished.")
    print(f"[TOTAL RUNNING TIME] {time.strftime('%H:%M:%S', time.gmtime(elapsed))} "
          f"({elapsed:.1f} seconds)")