# =============================================================================
# Program      : 06_e0_dataextract_arab_andalusian.py
# Version      : 1.0
# GPU Required : NO
# Description  : Experiment 0 -- Data Extraction (Arab-Andalusian corpus).
#                 This is a .tar.gz, not a .zip -- uses tarfile with
#                 filter='data' as required. No stems to exclude; every
#                 file in the archive is used (mp3, xml score, two
#                 precomputed-analysis JSONs, and a raw pitch-track txt
#                 per recording, plus 6 root-level lookup files).
#
#                 Categorization happens at the SECTION level, not once
#                 per recording (confirmed during cataloging): nawba/tab
#                 are effectively stable per recording (vary in only 1/164
#                 each), but mizan (141/164) and form (143/164) genuinely
#                 change mid-recording. This program therefore produces
#                 TWO manifests, not one:
#                   - a recording-level manifest (164 rows) with a
#                     "primary" nawba/tab (first section's value) plus
#                     explicit *_varies flags, never silently collapsing
#                     the ~1 recording where even nawba/tab vary
#                   - a section-level manifest (847 rows) with the real
#                     per-section nawba/mizan/tab/form, needed for any
#                     mizan/form-based metric later
#
#                 Root-level lookup files are SIBLINGS of documents/, not
#                 nested inside it. andalusian_recording.json is a TITLE
#                 lookup only (mbid/title), NOT the category join table --
#                 andalusian_description.json is the real join table, and
#                 is parsed directly here rather than re-deriving anything
#                 from andalusian_recording.json.
#
#                 License for this specific archive is NOT yet confirmed
#                 (the parent Zenodo record's CC BY 4.0 almost certainly
#                 applies, but this program does not assert that as fact).
#
# INPUT:
#                  Google Drive:
#                    /content/drive/MyDrive/datasets/Arab_Andalusian/ArabAndalusianDataset.tar.gz
#
# STEPS:
#                  Step 1  Mount Drive, ensure PROJECT_DIR exists
#                  Step 2  Copy the tar.gz to local disk (resumable)
#                  Step 3  Extract with filter='data' (whole-archive call,
#                          not per-member -- tar.gz doesn't support cheap
#                          random-access seeking the way zip does, so a
#                          coarse "skip if already extracted" check is used
#                          instead of per-file resumability)
#                  Step 4  Parse the 4 category lookup files (nawba, mizan,
#                          tab, form), andalusian_recording.json (titles),
#                          and andalusian_description.json (the real
#                          section-level join table)
#                  Step 5  Build the recording-level manifest (164 rows)
#                          and the section-level manifest (847 rows)
#                  Step 6  Save both manifests + summary JSON locally,
#                          copy all three to Google Drive
#
# OUTPUT FILES : under PROJECT_DIR
#                  06_arab_andalusian_recordings_manifest.csv
#                      mbid, title, archive_url, primary_nawba,
#                      primary_tab, nawba_varies, tab_varies, n_sections,
#                      mizan_varies, form_varies, lead_artist_name,
#                      lead_artist_id, audio_relpath, audio_exists,
#                      xml_relpath, xml_exists
#                  06_arab_andalusian_sections_manifest.csv
#                      mbid, section_index, start_time, end_time,
#                      nawba_id, nawba_name, mizan_id, mizan_name,
#                      tab_id, tab_name, form_id, form_name
#                  06_summary.json
#
# Hyperparameters : N/A -- data extraction step.
# Dependencies : pandas, tqdm, tarfile (stdlib), json (stdlib)
#
# Change Log   :
#   v1.0  2026-09-16  Initial version
# Running time:
# !pip install pandas tqdm
# =============================================================================

!pip install pandas tqdm

import os
import sys
import time
import json
import shutil
import tarfile
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from google.colab import drive
drive.mount('/content/drive')

RUN_START_TIME = time.time()
print(f"[{datetime.now().strftime('%H:%M:%S')}] Program 06_e0_dataextract_arab_andalusian.py started.")

# --- Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/sounding-fair/"

DRIVE_TARGZ = "/content/drive/MyDrive/datasets/Arab_Andalusian/ArabAndalusianDataset.tar.gz"

LOCAL_ROOT = "/content/local_data/arab_andalusian"
LOCAL_TARGZ = os.path.join(LOCAL_ROOT, "ArabAndalusianDataset.tar.gz")
LOCAL_EXTRACT_DIR = os.path.join(LOCAL_ROOT, "extracted")

EXPECTED_RECORDING_COUNT = 164
# floor for "already extracted" check: mp3 + metadata + pitch_distribution
# + pitch_filtered per recording (xml is present for only 158, so not
# counted in the floor)
EXPECTED_MIN_FILES = EXPECTED_RECORDING_COUNT * 4

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


def extract_targz(targz_path, extract_dir, expected_min_files):
    """tar.gz doesn't support cheap random-access seeking the way zip
    does, so resumability here is coarse: skip the WHOLE extraction if
    the directory already looks complete, rather than per-file skipping."""
    if os.path.isdir(extract_dir):
        existing_count = sum(len(files) for _, _, files in os.walk(extract_dir))
        if existing_count >= expected_min_files:
            print(f"[EXTRACT] {extract_dir} already has {existing_count} files "
                  f"(>= expected floor {expected_min_files}), skipping extraction.")
            return

    os.makedirs(extract_dir, exist_ok=True)
    print(f"[EXTRACT] Extracting {targz_path} (this may take a while, no "
          f"per-file progress for tar.gz)...")
    with tarfile.open(targz_path, "r:gz") as tf:
        tf.extractall(extract_dir, filter="data")
    print(f"[EXTRACT] Done.")


def print_disk_usage(path="/content"):
    total, used, free = shutil.disk_usage(path)
    print(f"[DISK] {path}: {used / 1e9:.1f} GB used, "
          f"{free / 1e9:.1f} GB free, {total / 1e9:.1f} GB total.")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, str):
        # audioanalysis--metadata.json files are double-encoded: the
        # top-level JSON value is itself a JSON string, not an object.
        data = json.loads(data)
    return data


def build_id_lookup(entries):
    """nawba/mizan/tab/form lookup files share the schema
    {id, name, transliterated_name}."""
    return {e["id"]: e for e in entries}


def extract_lead_artist(meta):
    recording = meta.get("recording", meta)
    arts = recording.get("artist-relation-list", [])
    for a in arts:
        if "lead vocals" in a.get("attribute-list", []):
            artist = a.get("artist", {})
            return artist.get("name"), artist.get("id")
    for a in arts:
        if a.get("type") == "vocal":
            artist = a.get("artist", {})
            return artist.get("name"), artist.get("id")
    if arts:
        artist = arts[0].get("artist", {})
        return artist.get("name"), artist.get("id")
    return None, None


def main():
    ensure_project_dir()
    print_disk_usage()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 2: copying archive from Drive...")
    copy_from_drive_with_progress(DRIVE_TARGZ, LOCAL_TARGZ)
    print_disk_usage()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 3: extracting (filter='data')...")
    extract_targz(LOCAL_TARGZ, LOCAL_EXTRACT_DIR, EXPECTED_MIN_FILES)
    print_disk_usage()

    root = os.path.join(LOCAL_EXTRACT_DIR, "ArabAndalusianDataset")
    if not os.path.isdir(root):
        root = LOCAL_EXTRACT_DIR
    documents_root = os.path.join(root, "documents")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 4: parsing lookup files...")
    nawba_lookup = build_id_lookup(load_json(os.path.join(root, "andalusian_nawba.json"))["results"])
    mizan_lookup = build_id_lookup(load_json(os.path.join(root, "andalusian_mizan.json"))["results"])
    tab_lookup = build_id_lookup(load_json(os.path.join(root, "andalusian_tab.json"))["results"])
    form_lookup = build_id_lookup(load_json(os.path.join(root, "andalusian_form.json"))["results"])
    recording_titles = {
        r["mbid"]: r for r in load_json(os.path.join(root, "andalusian_recording.json"))["results"]
    }
    description = load_json(os.path.join(root, "andalusian_description.json"))
    print(f"[LOOKUPS] nawba={len(nawba_lookup)}, mizan={len(mizan_lookup)}, "
          f"tab={len(tab_lookup)}, form={len(form_lookup)}, "
          f"titles={len(recording_titles)}, description_entries={len(description)}")

    # --- Step 5: build both manifests ---
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 5: building manifests...")
    recording_rows = []
    section_rows = []
    n_audio_found = 0
    n_xml_found = 0

    n_metadata_corrupt = [0]  # mutable counter, closed over inside the loop
    for entry in tqdm(description, desc="[MANIFEST] recordings"):
        mbid = entry.get("mbid")
        sections = entry.get("sections", [])
        doc_dir = os.path.join(documents_root, mbid)

        audio_path = os.path.join(doc_dir, f"{mbid}.mp3")
        audio_exists = os.path.isfile(audio_path)
        if audio_exists:
            n_audio_found += 1
        xml_path = os.path.join(doc_dir, f"{mbid}.xml")
        xml_exists = os.path.isfile(xml_path)
        if xml_exists:
            n_xml_found += 1


        meta_path = os.path.join(doc_dir, "audioanalysis--metadata.json")
        lead_artist_name, lead_artist_id = (None, None)
        if os.path.isfile(meta_path):
            try:
                lead_artist_name, lead_artist_id = extract_lead_artist(load_json(meta_path))
            except json.JSONDecodeError:
                n_metadata_corrupt[0] += 1


        nawba_ids, tab_ids, mizan_ids, form_ids = set(), set(), set(), set()
        for i, sec in enumerate(sections):
            nawba_id = sec.get("nawba", {}).get("id")
            mizan_id = sec.get("mizan", {}).get("id")
            tab_id = sec.get("tab", {}).get("id")
            form_id = sec.get("form", {}).get("id")
            nawba_ids.add(nawba_id)
            tab_ids.add(tab_id)
            mizan_ids.add(mizan_id)
            form_ids.add(form_id)
            section_rows.append({
                "mbid": mbid, "section_index": i,
                "start_time": sec.get("start_time"), "end_time": sec.get("end_time"),
                "nawba_id": nawba_id, "nawba_name": nawba_lookup.get(nawba_id, {}).get("transliterated_name"),
                "mizan_id": mizan_id, "mizan_name": mizan_lookup.get(mizan_id, {}).get("transliterated_name"),
                "tab_id": tab_id, "tab_name": tab_lookup.get(tab_id, {}).get("transliterated_name"),
                "form_id": form_id, "form_name": form_lookup.get(form_id, {}).get("transliterated_name"),
            })

        primary_nawba_id = sections[0].get("nawba", {}).get("id") if sections else None
        primary_tab_id = sections[0].get("tab", {}).get("id") if sections else None

        recording_rows.append({
            "mbid": mbid,
            "title": recording_titles.get(mbid, {}).get("title"),
            "transliterated_title": recording_titles.get(mbid, {}).get("transliterated_title"),
            "archive_url": entry.get("archive_url"),
            "primary_nawba": nawba_lookup.get(primary_nawba_id, {}).get("transliterated_name"),
            "primary_tab": tab_lookup.get(primary_tab_id, {}).get("transliterated_name"),
            "nawba_varies": len(nawba_ids) > 1,
            "tab_varies": len(tab_ids) > 1,
            "n_sections": len(sections),
            "mizan_varies": len(mizan_ids) > 1,
            "form_varies": len(form_ids) > 1,
            "lead_artist_name": lead_artist_name,
            "lead_artist_id": lead_artist_id,
            "audio_relpath": os.path.relpath(audio_path, root) if audio_exists else None,
            "audio_exists": audio_exists,
            "xml_relpath": os.path.relpath(xml_path, root) if xml_exists else None,
            "xml_exists": xml_exists,
        })

    recordings_manifest = pd.DataFrame(recording_rows)
    sections_manifest = pd.DataFrame(section_rows)

    n_rec = len(recordings_manifest)
    n_nawba_varies = int(recordings_manifest["nawba_varies"].sum())
    n_tab_varies = int(recordings_manifest["tab_varies"].sum())
    n_mizan_varies = int(recordings_manifest["mizan_varies"].sum())
    n_form_varies = int(recordings_manifest["form_varies"].sum())

    print(f"[SUMMARY] metadata.json corrupt (unescaped quote in source data): "
          f"{n_metadata_corrupt[0]}/{n_rec}")
    print(f"[SUMMARY] recordings: {n_rec} (expected {EXPECTED_RECORDING_COUNT})")
    print(f"[SUMMARY] audio found: {n_audio_found}/{n_rec}, xml found: {n_xml_found}/{n_rec}")
    print(f"[SUMMARY] nawba_varies: {n_nawba_varies}, tab_varies: {n_tab_varies}, "
          f"mizan_varies: {n_mizan_varies}, form_varies: {n_form_varies} "
          f"(expected ~1, ~1, ~141, ~143)")
    print(f"[SUMMARY] total sections: {len(sections_manifest)} (expected 847)")

    # --- Step 6: save and copy to Google Drive ---
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 6: saving and copying outputs...")
    local_rec_path = os.path.join(LOCAL_ROOT, "06_arab_andalusian_recordings_manifest.csv")
    local_sec_path = os.path.join(LOCAL_ROOT, "06_arab_andalusian_sections_manifest.csv")
    local_summary_path = os.path.join(LOCAL_ROOT, "06_summary.json")

    recordings_manifest.to_csv(local_rec_path, index=False)
    sections_manifest.to_csv(local_sec_path, index=False)

    elapsed_seconds = time.time() - RUN_START_TIME
    summary = {
        "program": "06_e0_dataextract_arab_andalusian.py",
        "recording_rows": n_rec,
        "section_rows": len(sections_manifest),
        "audio_found": n_audio_found,
        "xml_found": n_xml_found,
        "nawba_varies": n_nawba_varies,
        "tab_varies": n_tab_varies,
        "mizan_varies": n_mizan_varies,
        "form_varies": n_form_varies,
        "license_confirmed": False,
        "root_resolved": root,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "elapsed_hms": time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds)),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(local_summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    copy_to_drive(local_rec_path, "06_arab_andalusian_recordings_manifest.csv")
    copy_to_drive(local_sec_path, "06_arab_andalusian_sections_manifest.csv")
    copy_to_drive(local_summary_path, "06_summary.json")

    return summary


if __name__ == "__main__":
    summary = main()
    elapsed = time.time() - RUN_START_TIME
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Program 06_e0_dataextract_arab_andalusian.py finished.")
    print(f"[TOTAL RUNNING TIME] {time.strftime('%H:%M:%S', time.gmtime(elapsed))} "
          f"({elapsed:.1f} seconds)")