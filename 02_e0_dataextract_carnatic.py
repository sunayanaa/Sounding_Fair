# =============================================================================
# Program      : 02_e0_dataextract_carnatic.py
# Version      : 1.1
# GPU Required : NO
# Description  : Experiment 0 -- Data Extraction (Carnatic tradition corpus,
#                 Saraga Carnatic only; CMR is not used here -- deprioritized
#                 earlier for raga/melody purposes, and CMD is the dedicated
#                 held-out audit-time source for raga fidelity, so nothing
#                 in this program touches raga-class balance decisions).
#
#                 Saraga Carnatic ships 5 audio files per composition (one
#                 full mix + 4 isolated multitrack stems: vocal, violin,
#                 mridangam-left, mridangam-right). The backbone (E1) only
#                 needs the full mix -- the isolated stems matter later,
#                 specifically for E13's content-preservation stress test,
#                 which will do its own narrow extraction when it runs.
#                 This program therefore extracts SELECTIVELY: only the
#                 full-mix mp3, the per-composition metadata JSON, and the
#                 top-level file_paths.csv index are ever decompressed --
#                 the 4 stem files per composition are never read at all,
#                 not just deleted after extraction. This is expected to
#                 cut both the disk footprint and the extraction time to
#                 roughly 30% of a full extract (estimated ~4-5GB actual
#                 vs. the archive's 14GB nominal size, based on per-track
#                 mix-vs-stem byte ratios sampled during earlier cataloging).
#
#                 Known corpus caveats, carried forward from cataloging and
#                 NOT resolved by this program (left for the E4 audit step):
#                   - raaga/taala/form coverage is only ~74% (184/249
#                     compositions); the rest ship empty lists.
#                   - Even where raaga is present, per-class counts are too
#                     thin for a reliable fidelity metric on their own (43
#                     usable classes, mostly 2-3 recordings each), and the
#                     'Rāgamālika' entry is a multi-raga composition form,
#                     not a real single-raga class -- this manifest stores
#                     raaga/taala/form RAW and unfiltered; any exclusion of
#                     Rāgamālika-type entries is a metric-computation
#                     decision that belongs in the E4 script, not here.
#                   - taala is comparatively stronger but imbalanced (Adi
#                     alone is ~66% of taala-labeled compositions).
#
# INPUT:
#                  Google Drive:
#                    /content/drive/MyDrive/datasets/Saraga/saraga1.5_carnatic.zip
#
# STEPS:
#                  Step 1  Mount Drive, ensure PROJECT_DIR exists
#                  Step 2  Copy saraga1.5_carnatic.zip to local Colab disk
#                          (skipped if already present locally with
#                          matching size -- resumable after disconnects)
#                  Step 3  Selectively extract: full-mix mp3s, per-
#                          composition JSONs, and file_paths.csv only.
#                          __MACOSX/ and .DS_Store junk excluded. Resumable
#                          per-file -- an interrupted extraction can be
#                          rerun and will only fetch what's still missing.
#                  Step 4  Parse file_paths.csv (mbid index) and every
#                          extracted per-composition JSON
#                  Step 5  Build manifest: mbid, title, raaga/taala/form
#                          (raw), lead artist name + mbid, concert title,
#                          length, audio_relpath, audio_exists
#                  Step 6  Save manifest + summary JSON locally, copy
#                          both to Google Drive
#
# OUTPUT FILES : under PROJECT_DIR
#                  02_saraga_carnatic_manifest.csv
#                      One row per composition: mbid, title, raaga, taala,
#                      form, raaga_present, taala_present, form_present,
#                      lead_artist_name, lead_artist_mbid, concert_title,
#                      length_ms, audio_relpath, audio_exists
#                  02_summary.json
#                      Row counts, coverage stats, resolved local paths,
#                      and elapsed running time for this run.
#
# Hyperparameters : N/A -- data extraction step. No model training, no
#                    stochastic operations, nothing to seed.
# Dependencies : pandas, tqdm, zipfile (stdlib), json (stdlib)
#
# Change Log   :
#   v1.0  2026-09-16  Initial version
#   v1.1  2026-09-16  Fixed mix-audio path resolution: some concerts (e.g.
#                      "Live at Kamarajar Hall", "Sanjay at Sastri Hall")
#                      ship the full mix as a single '<name>.mp3', not the
#                      double '<name>.mp3.mp3' pattern seen in the initially
#                      sampled concert. v1.0 hardcoded the double-extension
#                      form only, which caused 81/249 compositions to be
#                      misreported as missing audio when the files were
#                      actually present on disk the whole time. Now checks
#                      the double-extension form first, then falls back to
#                      the single-extension form.
# Running time: 310 secs
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
print(f"[{datetime.now().strftime('%H:%M:%S')}] Program 02_e0_dataextract_carnatic.py started.")

# --- Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/sounding-fair/"

DRIVE_SARAGA_CARNATIC_ZIP = "/content/drive/MyDrive/datasets/Saraga/saraga1.5_carnatic.zip"

LOCAL_ROOT = "/content/local_data/carnatic"
LOCAL_ZIP = os.path.join(LOCAL_ROOT, "saraga1.5_carnatic.zip")
LOCAL_EXTRACT_DIR = os.path.join(LOCAL_ROOT, "saraga_carnatic_mixonly")

EXPECTED_COMPOSITION_COUNT = 249  # confirmed during dataset cataloging

os.makedirs(LOCAL_ROOT, exist_ok=True)

# --- Begin Google Drive Helper Functions ---
def ensure_project_dir():
    """Create PROJECT_DIR on Google Drive if it doesn't already exist.
    Idempotent -- safe to call even if Program 01 already created it."""
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
    Resumable: skipped entirely if dst already exists with the same size."""
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


def is_wanted_saraga_member(name):
    """Selective-extraction predicate for the Saraga Carnatic archive.
    Keeps: file_paths.csv, every per-composition JSON, and every full-mix
    mp3 (either the '<name>.mp3.mp3' or plain '<name>.mp3' pattern -- both
    exist across different concerts in this archive -- as long as it does
    NOT contain 'multitrack'). Drops: the 4 isolated stems per composition,
    __MACOSX/ resource forks, and .DS_Store junk."""
    if "__MACOSX" in name:
        return False
    if name.endswith("/"):
        return False  # directory entry, nothing to extract
    basename = os.path.basename(name)
    if basename == ".DS_Store" or basename.startswith("._"):
        return False
    if basename == "file_paths.csv":
        return True
    if basename.endswith(".json"):
        return True
    if basename.endswith(".mp3") and "multitrack" not in basename:
        return True
    return False


def selective_extract_zip_with_progress(zip_path, extract_dir, member_predicate,
                                         expected_min_files=None):
    """Extract only members of zip_path for which member_predicate(name) is
    True. Resumable per-file: a member already present on disk (by path) is
    not re-extracted."""
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
        print(f"[EXTRACT] {len(wanted)} of {len(all_members)} archive entries "
              f"match the mix-only selection (stems and junk excluded).")
        n_skipped_existing = 0
        for member in tqdm(wanted, desc=f"[EXTRACT] {os.path.basename(zip_path)} (mix-only)"):
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


def find_first(lst, default=None):
    """Return the 'name' field of the first entry in a raaga/taala/form
    list, or default if the list is empty. Values are stored RAW here --
    no filtering of multi-value composition forms (e.g. Ragamalika) is
    done at extraction time; that's an E4 metric-computation decision."""
    if lst:
        return lst[0].get("name")
    return default


def extract_lead_artist(entry):
    """Prefer the artist flagged lead=True in the 'artists' list; fall
    back to album_artists[0] if no lead flag is found."""
    for a in entry.get("artists", []):
        if a.get("lead"):
            artist = a.get("artist", {})
            return artist.get("name"), artist.get("mbid")
    album_artists = entry.get("album_artists", [])
    if album_artists:
        return album_artists[0].get("name"), album_artists[0].get("mbid")
    return None, None


def resolve_mix_path(comp_dir, comp_name):
    """A composition's full mix ships under one of two naming patterns
    depending on which concert it came from: '<name>.mp3.mp3' (seen in
    concerts like 'Sumitra Nitin at Arkay') or plain '<name>.mp3' (seen
    in concerts like 'Live at Kamarajar Hall', 'Sanjay at Sastri Hall').
    Check the double-extension form first, then fall back to the single-
    extension form, so both are found correctly rather than assuming one
    pattern holds for the whole archive."""
    double_ext_path = os.path.join(comp_dir, f"{comp_name}.mp3.mp3")
    if os.path.isfile(double_ext_path):
        return double_ext_path
    single_ext_path = os.path.join(comp_dir, f"{comp_name}.mp3")
    if os.path.isfile(single_ext_path):
        return single_ext_path
    return None


def main():
    ensure_project_dir()
    print_disk_usage()

    # --- Step 2: copy archive from Drive to local disk ---
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 2: copying archive from Drive...")
    copy_from_drive_with_progress(DRIVE_SARAGA_CARNATIC_ZIP, LOCAL_ZIP)
    print_disk_usage()

    # --- Step 3: selective mix-only extraction ---
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 3: selective (mix-only) extraction...")
    # expected_min_files is a loose floor, not exact: file_paths.csv + one
    # json + one mix mp3 per composition = roughly 2*249 + 1
    selective_extract_zip_with_progress(
        LOCAL_ZIP, LOCAL_EXTRACT_DIR, is_wanted_saraga_member,
        expected_min_files=2 * EXPECTED_COMPOSITION_COUNT,
    )
    print_disk_usage()

    # --- Step 4: locate the extracted root and parse JSONs ---
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 4: parsing per-composition JSON files...")
    # The archive's internal paths already start with 'saraga1.5_carnatic/',
    # so that's where extraction lands under LOCAL_EXTRACT_DIR.
    saraga_root = os.path.join(LOCAL_EXTRACT_DIR, "saraga1.5_carnatic")
    if not os.path.isdir(saraga_root):
        saraga_root = LOCAL_EXTRACT_DIR  # fall back if the nesting differs

    json_paths = []
    for dirpath, _, filenames in os.walk(saraga_root):
        for fn in filenames:
            if fn.endswith(".json"):
                json_paths.append(os.path.join(dirpath, fn))
    print(f"[PARSE] Found {len(json_paths)} composition JSON files "
          f"(expected {EXPECTED_COMPOSITION_COUNT}).")
    if len(json_paths) != EXPECTED_COMPOSITION_COUNT:
        print(f"[WARNING] JSON count does not match the expected "
              f"{EXPECTED_COMPOSITION_COUNT} -- verify archive version before trusting this manifest.")

    # --- Step 5: build the manifest ---
    rows = []
    n_raaga = n_taala = n_form = 0
    for json_path in tqdm(json_paths, desc="[PARSE] reading JSON"):
        with open(json_path, "r", encoding="utf-8") as f:
            entry = json.load(f)

        comp_dir = os.path.dirname(json_path)
        comp_name = os.path.splitext(os.path.basename(json_path))[0]
        mix_path = resolve_mix_path(comp_dir, comp_name)
        audio_exists = mix_path is not None
        audio_relpath = os.path.relpath(mix_path, LOCAL_EXTRACT_DIR) if audio_exists else None

        raaga = find_first(entry.get("raaga", []))
        taala = find_first(entry.get("taala", []))
        form = find_first(entry.get("form", []))
        if raaga is not None:
            n_raaga += 1
        if taala is not None:
            n_taala += 1
        if form is not None:
            n_form += 1

        lead_artist_name, lead_artist_mbid = extract_lead_artist(entry)
        concert = entry.get("concert", [])
        concert_title = concert[0].get("title") if concert else None

        rows.append({
            "mbid": entry.get("mbid"),
            "title": entry.get("title"),
            "raaga": raaga,
            "taala": taala,
            "form": form,
            "raaga_present": raaga is not None,
            "taala_present": taala is not None,
            "form_present": form is not None,
            "lead_artist_name": lead_artist_name,
            "lead_artist_mbid": lead_artist_mbid,
            "concert_title": concert_title,
            "length_ms": entry.get("length"),
            "audio_relpath": audio_relpath,
            "audio_exists": audio_exists,
        })

    manifest = pd.DataFrame(rows)
    n_missing_audio = int((~manifest["audio_exists"]).sum())
    if n_missing_audio > 0:
        print(f"[WARNING] {n_missing_audio} compositions have a JSON but no "
              f"matching mix mp3 on disk (checked both naming patterns).")
    else:
        print(f"[VERIFY] All {len(manifest)} compositions have a matching mix mp3.")

    print(f"[COVERAGE] raaga: {n_raaga}/{len(manifest)} "
          f"({100 * n_raaga / max(len(manifest), 1):.1f}%), "
          f"taala: {n_taala}/{len(manifest)} "
          f"({100 * n_taala / max(len(manifest), 1):.1f}%), "
          f"form: {n_form}/{len(manifest)} "
          f"({100 * n_form / max(len(manifest), 1):.1f}%).")

    # --- Step 6: save locally, then copy to Google Drive ---
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 6: saving and copying outputs...")
    local_manifest_path = os.path.join(LOCAL_ROOT, "02_saraga_carnatic_manifest.csv")
    local_summary_path = os.path.join(LOCAL_ROOT, "02_summary.json")

    manifest.to_csv(local_manifest_path, index=False)

    elapsed_seconds = time.time() - RUN_START_TIME
    summary = {
        "program": "02_e0_dataextract_carnatic.py",
        "manifest_rows": int(len(manifest)),
        "manifest_missing_audio": n_missing_audio,
        "expected_composition_count": EXPECTED_COMPOSITION_COUNT,
        "raaga_coverage": n_raaga,
        "taala_coverage": n_taala,
        "form_coverage": n_form,
        "saraga_root_resolved": saraga_root,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "elapsed_hms": time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds)),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(local_summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    copy_to_drive(local_manifest_path, "02_saraga_carnatic_manifest.csv")
    copy_to_drive(local_summary_path, "02_summary.json")

    return summary


if __name__ == "__main__":
    summary = main()
    elapsed = time.time() - RUN_START_TIME
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Program 02_e0_dataextract_carnatic.py finished.")
    print(f"[TOTAL RUNNING TIME] {time.strftime('%H:%M:%S', time.gmtime(elapsed))} "
          f"({elapsed:.1f} seconds)")