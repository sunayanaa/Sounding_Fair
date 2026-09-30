# =============================================================================
# Program      : 03_e0_dataextract_hindustani.py
# Version      : 1.0
# GPU Required : NO
# Description  : Experiment 0 -- Data Extraction (Hindustani tradition
#                 corpus, Saraga Hindustani). No isolated stems exist in
#                 this archive (confirmed during cataloging: 0 multitrack
#                 files) -- full extraction, junk excluded.
#
#                 Schema differs from Saraga Carnatic: fields are 'raags'/
#                 'taals' (plural, {uuid, name, common_name} dicts), plus
#                 'layas' and 'forms' not present in the Carnatic schema.
#                 Concert info is under 'release', not 'concert'.
#
#                 Known corpus caveats, carried forward from cataloging and
#                 NOT resolved by this program (left for the E4 audit step):
#                   - raags: 100% coverage (108/108), but only 21 of 61
#                     classes have >=2 recordings -- clean but thin.
#                   - taals: 57/108 recordings list MULTIPLE taals (mid-
#                     performance transitions -- the taal-side analogue of
#                     Carnatic's Ragamalika problem) and 40/108 list NONE;
#                     only 11/108 have exactly one. This manifest stores
#                     taal_count and taal_single (name only when count==1)
#                     so the ambiguous/absent cases are never silently
#                     collapsed into a false single value.
#                   - Applying the Carnatic lesson defensively: mix-file
#                     naming is checked for both '<name>.mp3.mp3' and plain
#                     '<name>.mp3', not assumed to be one pattern.
#
#                 .ctonic.txt (manually annotated tonic, Hz) is parsed where
#                 present -- a real head start on the tonic-normalized pitch
#                 descriptor, no algorithmic tonic detection needed for
#                 these recordings.
#
# INPUT:
#                  Google Drive:
#                    /content/drive/MyDrive/datasets/Saraga/saraga1.5_hindustani.zip
#
# STEPS:
#                  Step 1  Mount Drive, ensure PROJECT_DIR exists
#                  Step 2  Copy saraga1.5_hindustani.zip to local disk
#                          (resumable -- skipped if already present with
#                          matching size)
#                  Step 3  Extract, excluding __MACOSX/ and .DS_Store junk
#                          only (no stems exist here to filter). Resumable
#                          per-file.
#                  Step 4  Parse every extracted per-recording JSON
#                  Step 5  Build manifest: mbid, title, raga (common_name),
#                          taal_count, taal_single, laya, form, tonic_hz,
#                          lead artist name+mbid, release title, length,
#                          audio_relpath, audio_exists
#                  Step 6  Save manifest + summary JSON locally, copy
#                          both to Google Drive
#
# OUTPUT FILES : under PROJECT_DIR
#                  03_saraga_hindustani_manifest.csv
#                  03_summary.json
#
# Hyperparameters : N/A -- data extraction step.
# Dependencies : pandas, tqdm, zipfile (stdlib), json (stdlib)
#
# Change Log   :
#   v1.0  2026-09-16  Initial version
# Running time: 141 secs
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
print(f"[{datetime.now().strftime('%H:%M:%S')}] Program 03_e0_dataextract_hindustani.py started.")

# --- Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/sounding-fair/"

DRIVE_SARAGA_HINDUSTANI_ZIP = "/content/drive/MyDrive/datasets/Saraga/saraga1.5_hindustani.zip"

LOCAL_ROOT = "/content/local_data/hindustani"
LOCAL_ZIP = os.path.join(LOCAL_ROOT, "saraga1.5_hindustani.zip")
LOCAL_EXTRACT_DIR = os.path.join(LOCAL_ROOT, "saraga_hindustani_extracted")

EXPECTED_RECORDING_COUNT = 108  # confirmed during dataset cataloging

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
    """No stems exist in this archive -- keep everything except __MACOSX/
    and .DS_Store/AppleDouble junk."""
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


def find_first_common_name(lst, default=None):
    if lst:
        return lst[0].get("common_name") or lst[0].get("name")
    return default


def extract_lead_artist(entry):
    for a in entry.get("artists", []):
        if a.get("lead"):
            artist = a.get("artist", {})
            return artist.get("name"), artist.get("mbid")
    album_artists = entry.get("album_artists", [])
    if album_artists:
        return album_artists[0].get("name"), album_artists[0].get("mbid")
    return None, None


def resolve_mix_path(comp_dir, comp_name):
    """Applying the Saraga Carnatic lesson defensively: check both the
    double-extension and single-extension mix filename patterns rather
    than assuming one holds for the whole archive."""
    double_ext_path = os.path.join(comp_dir, f"{comp_name}.mp3.mp3")
    if os.path.isfile(double_ext_path):
        return double_ext_path
    single_ext_path = os.path.join(comp_dir, f"{comp_name}.mp3")
    if os.path.isfile(single_ext_path):
        return single_ext_path
    return None


def read_tonic(comp_dir, comp_name):
    tonic_path = os.path.join(comp_dir, f"{comp_name}.ctonic.txt")
    if not os.path.isfile(tonic_path):
        return None
    try:
        with open(tonic_path, "r") as f:
            return float(f.read().strip())
    except (ValueError, OSError):
        return None


def main():
    ensure_project_dir()
    print_disk_usage()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 2: copying archive from Drive...")
    copy_from_drive_with_progress(DRIVE_SARAGA_HINDUSTANI_ZIP, LOCAL_ZIP)
    print_disk_usage()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 3: extracting (junk excluded)...")
    selective_extract_zip_with_progress(
        LOCAL_ZIP, LOCAL_EXTRACT_DIR, is_wanted_member,
        expected_min_files=2 * EXPECTED_RECORDING_COUNT,
    )
    print_disk_usage()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 4: parsing per-recording JSON files...")
    saraga_root = os.path.join(LOCAL_EXTRACT_DIR, "saraga1.5_hindustani")
    if not os.path.isdir(saraga_root):
        saraga_root = LOCAL_EXTRACT_DIR

    json_paths = []
    for dirpath, _, filenames in os.walk(saraga_root):
        for fn in filenames:
            if fn.endswith(".json"):
                json_paths.append(os.path.join(dirpath, fn))
    print(f"[PARSE] Found {len(json_paths)} recording JSON files "
          f"(expected {EXPECTED_RECORDING_COUNT}).")
    if len(json_paths) != EXPECTED_RECORDING_COUNT:
        print(f"[WARNING] JSON count does not match the expected "
              f"{EXPECTED_RECORDING_COUNT} -- verify archive version before trusting this manifest.")

    rows = []
    n_raga = n_taal_single = n_taal_multi = n_taal_absent = n_laya = n_form = n_tonic = 0
    for json_path in tqdm(json_paths, desc="[PARSE] reading JSON"):
        with open(json_path, "r", encoding="utf-8") as f:
            entry = json.load(f)

        comp_dir = os.path.dirname(json_path)
        comp_name = os.path.splitext(os.path.basename(json_path))[0]
        mix_path = resolve_mix_path(comp_dir, comp_name)
        audio_exists = mix_path is not None
        audio_relpath = os.path.relpath(mix_path, LOCAL_EXTRACT_DIR) if audio_exists else None

        raga = find_first_common_name(entry.get("raags", []))
        if raga is not None:
            n_raga += 1

        taals = entry.get("taals", [])
        taal_count = len(taals)
        if taal_count == 1:
            taal_single = taals[0].get("common_name") or taals[0].get("name")
            n_taal_single += 1
        elif taal_count > 1:
            taal_single = None
            n_taal_multi += 1
        else:
            taal_single = None
            n_taal_absent += 1

        laya = find_first_common_name(entry.get("layas", []))
        if laya is not None:
            n_laya += 1
        form = find_first_common_name(entry.get("forms", []))
        if form is not None:
            n_form += 1

        tonic_hz = read_tonic(comp_dir, comp_name)
        if tonic_hz is not None:
            n_tonic += 1

        lead_artist_name, lead_artist_mbid = extract_lead_artist(entry)
        release = entry.get("release", [])
        release_title = release[0].get("title") if release else None

        rows.append({
            "mbid": entry.get("mbid"),
            "title": entry.get("title"),
            "raga": raga,
            "taal_count": taal_count,
            "taal_single": taal_single,
            "laya": laya,
            "form": form,
            "tonic_hz": tonic_hz,
            "lead_artist_name": lead_artist_name,
            "lead_artist_mbid": lead_artist_mbid,
            "release_title": release_title,
            "length_ms": entry.get("length"),
            "audio_relpath": audio_relpath,
            "audio_exists": audio_exists,
        })

    manifest = pd.DataFrame(rows)
    n_missing_audio = int((~manifest["audio_exists"]).sum())
    if n_missing_audio > 0:
        print(f"[WARNING] {n_missing_audio} recordings have a JSON but no "
              f"matching mix mp3 on disk (checked both naming patterns).")
    else:
        print(f"[VERIFY] All {len(manifest)} recordings have a matching mix mp3.")

    n = max(len(manifest), 1)
    print(f"[COVERAGE] raga: {n_raga}/{n} ({100 * n_raga / n:.1f}%), "
          f"taal single: {n_taal_single}/{n}, taal multi: {n_taal_multi}/{n}, "
          f"taal absent: {n_taal_absent}/{n}, "
          f"laya: {n_laya}/{n}, form: {n_form}/{n}, tonic: {n_tonic}/{n}.")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 6: saving and copying outputs...")
    local_manifest_path = os.path.join(LOCAL_ROOT, "03_saraga_hindustani_manifest.csv")
    local_summary_path = os.path.join(LOCAL_ROOT, "03_summary.json")

    manifest.to_csv(local_manifest_path, index=False)

    elapsed_seconds = time.time() - RUN_START_TIME
    summary = {
        "program": "03_e0_dataextract_hindustani.py",
        "manifest_rows": int(len(manifest)),
        "manifest_missing_audio": n_missing_audio,
        "expected_recording_count": EXPECTED_RECORDING_COUNT,
        "raga_coverage": n_raga,
        "taal_single": n_taal_single,
        "taal_multi": n_taal_multi,
        "taal_absent": n_taal_absent,
        "laya_coverage": n_laya,
        "form_coverage": n_form,
        "tonic_coverage": n_tonic,
        "saraga_root_resolved": saraga_root,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "elapsed_hms": time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds)),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(local_summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    copy_to_drive(local_manifest_path, "03_saraga_hindustani_manifest.csv")
    copy_to_drive(local_summary_path, "03_summary.json")

    return summary


if __name__ == "__main__":
    summary = main()
    elapsed = time.time() - RUN_START_TIME
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Program 03_e0_dataextract_hindustani.py finished.")
    print(f"[TOTAL RUNNING TIME] {time.strftime('%H:%M:%S', time.gmtime(elapsed))} "
          f"({elapsed:.1f} seconds)")