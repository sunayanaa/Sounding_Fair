# =============================================================================
# Program      : 12_e4_fidelity_staging.py
# Version      : 1.0
# GPU Required : NO
# Description  : Experiment 4, Part 1 -- staging the 4 LABELED fidelity
#                 corpora needed for Intra-Group Retrieval Fidelity. This
#                 is separate from E1's 7 anonymous crop caches (which
#                 have no per-track filename/label linkage at all -- fine
#                 for tradition-level audit metrics, useless for a
#                 per-track fidelity metric) and separate from Program 06
#                 (which produced a nawba manifest but never cached the
#                 actual audio).
#
#                 Per the locked design, CMD/HMD -- not Saraga's own
#                 raga/taala -- is the real raga-fidelity source for BOTH
#                 Carnatic and Hindustani (that's the whole reason it was
#                 fetched: Saraga's raga classes were too thin to be
#                 usable alone). This is the FIRST program in the entire
#                 pipeline that stages CMD/HMD audio at all -- it was
#                 deliberately held out of E1's training pool to avoid
#                 measuring memorization instead of genuine retrieval
#                 fidelity, so this is exactly the point where it should
#                 first appear.
#
#                 Fidelity axis per tradition (from the locked design):
#                   - Carnatic:        CMD raga  (folder-UUID label,
#                                       opaque but usable as a class ID --
#                                       no name mapping exists in this
#                                       archive, confirmed during
#                                       cataloging)
#                   - Hindustani:      HMD raga  (same UUID scheme)
#                   - Jingju:          role-type (dan/laosheng/jing/
#                                       xiaosheng/chou), from the base
#                                       corpus's Legend-decoded filename
#                                       codes AND JaCRC's role_type_raw,
#                                       both normalized to this 5-class
#                                       scheme (JaCRC's finer labels --
#                                       nandan, huadan, nvjing,
#                                       nvlaosheng -- collapse to their
#                                       parent role type)
#                   - Arab-Andalusian: nawba (13 classes, stable per
#                                       recording -- confirmed varies in
#                                       only 1/164 recordings)
#
#                 One fixed crop per track (not E1's 3 -- fidelity
#                 evaluation wants a stable, reproducible embedding per
#                 track, not augmentation diversity), same bounded-load
#                 approach as E1 (torchaudio.info() unavailable in this
#                 build; fixed offset guesses instead, so memory never
#                 scales with a file's true length regardless of how long
#                 CMD/HMD's concert-style recordings run).
#
# INPUT:
#                  Google Drive:
#                    datasets/RagaRecognition/Indian_Art_Music_Raga_Recognition_Dataset-audio.zip
#                    datasets/Jingju/catalogue-dan.csv, catalogue-laosheng.csv, wav_mono-1.zip
#                    datasets/Jingju/JaCRC/JaCRC-recordings.zip,
#                      JaCRC-recordings-info.csv, JaCRC-students-recordings-info.csv
#                    datasets/Arab_Andalusian/ArabAndalusianDataset.tar.gz
#
# STEPS:
#                  Step 1  Ensure PROJECT_DIR exists
#                  Step 2  Stage CMD (Carnatic raga) -- selective zip
#                          extraction (mp3 only), raga-UUID from folder path
#                  Step 3  Stage HMD (Hindustani raga) -- same archive,
#                          Hindustani subtree
#                  Step 4  Stage Jingju role-type -- base corpus (Legend-
#                          decoded codes) + JaCRC (role_type_raw), both
#                          normalized to a common 5-class scheme
#                  Step 5  Stage Arab-Andalusian nawba -- tar.gz + the
#                          section-level join table, primary_nawba per
#                          recording
#                  Step 6  Convert each corpus's audio to 1 fixed crop/
#                          track, cache (crops + labels) to Google Drive, delete
#                          raw audio immediately after each corpus
#
# OUTPUT FILES : under PROJECT_DIR
#                  12_fidelity_cmd_carnatic.pt    {crops, labels, format_version}
#                  12_fidelity_hmd_hindustani.pt
#                  12_fidelity_jingju_roletype.pt
#                  12_fidelity_arab_andalusian_nawba.pt
#                  (each with a companion 12_fidelity_<name>_meta.json for
#                  the lightweight cache-validity check, same pattern as
#                  Program 10/11)
#
# Hyperparameters : SAMPLE_RATE=22050, SEGMENT_SECONDS=4 (must match E1
#                    exactly for embedding compatibility in Program 13),
#                    CROP_OFFSET_GUESSES_SECONDS=[15] (single crop),
#                    FIDELITY_FORMAT_VERSION=1.
# Dependencies : torch, torchaudio, pandas, tqdm,
#                 zipfile (stdlib), tarfile (stdlib), json (stdlib), re (stdlib)
#
# Change Log   :
#   v1.0  2026-09-23  Initial version
# Running time: 40:00
# !pip install torch torchaudio pandas tqdm
# =============================================================================

!pip install torch torchaudio pandas tqdm

import os
import re
import sys
import time
import json
import shutil
import zipfile
import tarfile
from datetime import datetime

import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from google.colab import drive
drive.mount('/content/drive')

RUN_START_TIME = time.time()
print(f"[{datetime.now().strftime('%H:%M:%S')}] Program 12_e4_fidelity_staging.py started.")

# --- Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/sounding-fair/"

SAMPLE_RATE = 22050
SEGMENT_SECONDS = 4
SEGMENT_LENGTH = SAMPLE_RATE * SEGMENT_SECONDS
CROP_OFFSET_GUESSES_SECONDS = [15]  # one fixed crop per track for fidelity eval
FIDELITY_FORMAT_VERSION = 1

MIN_EXPECTED = {
    "cmd_carnatic": 400, "hmd_hindustani": 250,
    "jingju_roletype": 250, "arab_andalusian_nawba": 140,
}

LOCAL_ROOT = "/content/local_data_e4stage"
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


def upload_meta(name_prefix, format_version, extra=None):
    meta = {"format_version": format_version}
    if extra:
        meta.update(extra)
    meta_local = os.path.join(LOCAL_ROOT, f"{name_prefix}_meta.json")
    with open(meta_local, "w") as f:
        json.dump(meta, f)
    copy_to_drive(meta_local, f"{name_prefix}_meta.json")


def remote_cache_is_valid(name_prefix, min_rows=1):
    meta_local = os.path.join(LOCAL_ROOT, f"{name_prefix}_meta_check.json")
    if os.path.exists(meta_local):
        os.remove(meta_local)
    found = copy_from_drive(f"{name_prefix}_meta.json", meta_local, skip_if_exists=False, verbose=False)
    if not found:
        return False
    with open(meta_local) as f:
        meta = json.load(f)
    if meta.get("format_version") != FIDELITY_FORMAT_VERSION:
        return False
    return meta.get("n_rows", 0) >= min_rows
# --- End Google Drive Helper Functions ---


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


def path_to_crop(path):
    """Same bounded-load approach as Program 08/11: torchaudio.info() is
    unavailable in this build, so try a fixed offset guess with a bounded
    num_frames; fall back to offset 0. Memory never scales with the
    file's true length."""
    guess_sr = 44100
    orig_segment_guess = guess_sr * SEGMENT_SECONDS
    for offset_s in CROP_OFFSET_GUESSES_SECONDS + [0]:
        offset_frames = guess_sr * offset_s
        for frame_offset in (offset_frames, 0):
            try:
                wf, sr = torchaudio.load(path, frame_offset=frame_offset, num_frames=orig_segment_guess)
            except Exception:
                continue
            if wf.shape[1] > 0:
                if wf.shape[0] > 1:
                    wf = wf.mean(dim=0, keepdim=True)
                if sr != SAMPLE_RATE:
                    wf = torchaudio.functional.resample(wf, sr, SAMPLE_RATE)
                if wf.shape[1] < SEGMENT_LENGTH:
                    wf = F.pad(wf, (0, SEGMENT_LENGTH - wf.shape[1]))
                elif wf.shape[1] > SEGMENT_LENGTH:
                    wf = wf[:, :SEGMENT_LENGTH]
                wf = torch.clamp(wf, -1.0, 1.0)
                return (wf * 32767).to(torch.int16).squeeze(0)
    return None


def cache_fidelity_corpus(name_prefix, path_label_pairs):
    """Given a list of (audio_path, label) pairs, convert to 1 fixed
    crop/track, cache to Google Drive with format_version tagging (skips entirely
    if a valid cache is already there)."""
    crops_local = os.path.join(LOCAL_ROOT, f"{name_prefix}.pt")
    crops_remote = f"{name_prefix}.pt"

    if remote_cache_is_valid(name_prefix, min_rows=MIN_EXPECTED.get(name_prefix, 1)):
        if copy_from_drive(crops_remote, crops_local, skip_if_exists=True):
            data = torch.load(crops_local, map_location="cpu", weights_only=False)
            print(f"[{name_prefix}] Reused cached crops from Google Drive: {data['crops'].shape[0]} tracks.")
            return data["crops"].shape[0]

    print(f"[{name_prefix}] Not cached (or stale) -- converting {len(path_label_pairs)} tracks...")
    crops_tensor = torch.zeros(len(path_label_pairs), SEGMENT_LENGTH, dtype=torch.int16)
    labels = []
    n_failed = 0
    for i, (path, label) in enumerate(tqdm(path_label_pairs, desc=f"[CROP] {name_prefix}")):
        c = path_to_crop(path)
        if c is not None:
            crops_tensor[i] = c
        else:
            n_failed += 1
        labels.append(label)
    print(f"[{name_prefix}] {len(path_label_pairs) - n_failed}/{len(path_label_pairs)} converted "
          f"({n_failed} left as silent placeholder rows).")

    torch.save({"crops": crops_tensor, "labels": labels, "format_version": FIDELITY_FORMAT_VERSION},
               crops_local)
    copy_to_drive(crops_local, crops_remote)
    upload_meta(name_prefix, FIDELITY_FORMAT_VERSION, {"n_rows": int(crops_tensor.shape[0])})
    os.remove(crops_local)
    return crops_tensor.shape[0]


def print_disk_usage(path="/content"):
    total, used, free = shutil.disk_usage(path)
    print(f"[DISK] {path}: {used / 1e9:.1f} GB used, {free / 1e9:.1f} GB free, "
          f"{total / 1e9:.1f} GB total.")


# --- Step 2/3: CMD/HMD ---

_SHARED_RAGADATASET_ZIP = os.path.join(LOCAL_ROOT, "ragadataset.zip")


def stage_cmd_hmd(tradition_subdir, name_prefix):
    """tradition_subdir: 'Carnatic' or 'Hindustani' within the shared
    RagaDataset archive. Label = the raga-UUID folder name (opaque, no
    name mapping exists in this archive -- confirmed during cataloging).
    CMD and HMD share the same 16GB zip -- copied ONCE to a path outside
    either tradition's own (deleted) root, so the second call reuses it
    instead of re-copying from Drive."""
    root = os.path.join(LOCAL_ROOT, name_prefix)
    os.makedirs(root, exist_ok=True)
    copy_from_drive_with_progress(
        "/content/drive/MyDrive/datasets/RagaRecognition/"
        "Indian_Art_Music_Raga_Recognition_Dataset-audio.zip", _SHARED_RAGADATASET_ZIP)

    prefix = f"RagaDataset/{tradition_subdir}/audio/"
    extract_dir = os.path.join(root, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(_SHARED_RAGADATASET_ZIP, "r") as zf:
        wanted = [m for m in zf.namelist()
                  if m.startswith(prefix) and m.endswith(".mp3") and not is_junk(m)]
        for member in tqdm(wanted, desc=f"[EXTRACT] {name_prefix}"):
            target = os.path.join(extract_dir, member)
            if os.path.exists(target):
                continue
            zf.extract(member, extract_dir)

    pairs = []
    for member in wanted:
        rel = member[len(prefix):]
        raga_uuid = rel.split("/")[0]
        audio_path = os.path.join(extract_dir, member)
        if os.path.isfile(audio_path):
            pairs.append((audio_path, raga_uuid))
    print(f"[{name_prefix}] {len(pairs)} (audio, raga_uuid) pairs found.")

    n_cached = cache_fidelity_corpus(name_prefix, pairs)
    shutil.rmtree(root, ignore_errors=True)
    print_disk_usage()
    return n_cached


# --- Step 4: Jingju role-type ---

def build_legend_lookup(*dataframes):
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
                lookup[cell] = cell
    return lookup


def leading_code(file_name):
    stem = os.path.splitext(file_name)[0]
    return stem.split("-", 1)[0]


def normalize_jingju_roletype(raw_label):
    if raw_label is None:
        return None
    raw_label = str(raw_label).strip().lower()
    mapping = {
        "dan": "dan", "nandan": "dan", "huadan": "dan",
        "laosheng": "laosheng", "nvlaosheng": "laosheng",
        "jing": "jing", "nvjing": "jing",
        "xiaosheng": "xiaosheng",
        "chou": "chou",
    }
    return mapping.get(raw_label)


def extract_roletype_from_legend_desc(desc):
    if not desc:
        return None
    m = re.search(r"role-type:\s*(\w+)", desc)
    if m:
        return normalize_jingju_roletype(m.group(1))
    return None


def stage_jingju_roletype():
    name_prefix = "jingju_roletype"
    root = os.path.join(LOCAL_ROOT, name_prefix)
    os.makedirs(root, exist_ok=True)
    pairs = []

    # --- base corpus ---
    df_dan = pd.read_csv("/content/drive/MyDrive/datasets/Jingju/catalogue-dan.csv")
    df_laosheng = pd.read_csv("/content/drive/MyDrive/datasets/Jingju/catalogue-laosheng.csv")
    legend_lookup = build_legend_lookup(df_dan, df_laosheng)
    catalogue = pd.concat([df_dan, df_laosheng], ignore_index=True)

    base_zip = os.path.join(root, "wav_mono-1.zip")
    copy_from_drive_with_progress("/content/drive/MyDrive/datasets/Jingju/wav_mono-1.zip", base_zip)
    base_extract = os.path.join(root, "base_extracted")
    with zipfile.ZipFile(base_zip, "r") as zf:
        wanted = [m for m in zf.namelist() if not is_junk(m)]
        for member in tqdm(wanted, desc="[EXTRACT] jingju base"):
            target = os.path.join(base_extract, member)
            if os.path.exists(target):
                continue
            zf.extract(member, base_extract)

    # Build a filename->path index via a full walk, rather than assuming
    # a specific subfolder structure inside wav_mono-1.zip that hasn't
    # actually been confirmed (unlike CMD/HMD's and Arab-Andalusian's
    # paths, which came from already-verified cataloging).
    base_wav_index = {}
    for dirpath, _, filenames in os.walk(base_extract):
        for fn in filenames:
            if fn.endswith(".wav"):
                base_wav_index[fn] = os.path.join(dirpath, fn)
    print(f"[jingju base] {len(base_wav_index)} .wav files found under wav_mono-1.zip's extraction.")

    for _, r in catalogue.iterrows():
        file_name = str(r["File name"]).strip()
        code = leading_code(file_name)
        role_type = extract_roletype_from_legend_desc(legend_lookup.get(code))
        if role_type is None:
            continue
        found_path = base_wav_index.get(f"{file_name}.wav")
        if found_path is not None:
            pairs.append((found_path, role_type))
    print(f"[jingju base] {len(pairs)} (audio, role_type) pairs so far.")

    # --- JaCRC ---
    jacrc_zip = os.path.join(root, "JaCRC-recordings.zip")
    copy_from_drive_with_progress(
        "/content/drive/MyDrive/datasets/Jingju/JaCRC/JaCRC-recordings.zip", jacrc_zip)
    jacrc_extract = os.path.join(root, "jacrc_extracted")
    with zipfile.ZipFile(jacrc_zip, "r") as zf:
        wanted = [m for m in zf.namelist() if not is_junk(m)]
        for member in tqdm(wanted, desc="[EXTRACT] jacrc"):
            target = os.path.join(jacrc_extract, member)
            if os.path.exists(target):
                continue
            zf.extract(member, jacrc_extract)

    df_main = pd.read_csv(
        "/content/drive/MyDrive/datasets/Jingju/JaCRC/JaCRC-recordings-info.csv", sep=";")
    df_student = pd.read_csv(
        "/content/drive/MyDrive/datasets/Jingju/JaCRC/JaCRC-students-recordings-info.csv", sep=";")

    main_pool_dir = student_dir = None
    for dirpath, _, _ in os.walk(jacrc_extract):
        if os.path.basename(dirpath) == "JaCRC-recordings":
            main_pool_dir = dirpath
        if os.path.basename(dirpath) == "JaCRC-students-recordings":
            student_dir = dirpath

    n_before_jacrc = len(pairs)
    for _, r in df_main.iterrows():
        role_type = normalize_jingju_roletype(r.get("role type"))
        if role_type is None or main_pool_dir is None:
            continue
        audio_path = os.path.join(main_pool_dir, str(r["file name"]).strip())
        if os.path.isfile(audio_path):
            pairs.append((audio_path, role_type))
    for _, r in df_student.iterrows():
        role_type = normalize_jingju_roletype(r.get("role type"))
        if role_type is None or student_dir is None:
            continue
        audio_path = os.path.join(student_dir, str(r["file name"]).strip())
        if os.path.isfile(audio_path):
            pairs.append((audio_path, role_type))
    print(f"[jacrc] {len(pairs) - n_before_jacrc} (audio, role_type) pairs added.")
    print(f"[{name_prefix}] {len(pairs)} total (audio, role_type) pairs.")

    n_cached = cache_fidelity_corpus(name_prefix, pairs)
    shutil.rmtree(root, ignore_errors=True)
    print_disk_usage()
    return n_cached


# --- Step 5: Arab-Andalusian nawba ---

def load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def stage_arab_andalusian_nawba():
    name_prefix = "arab_andalusian_nawba"
    root = os.path.join(LOCAL_ROOT, name_prefix)
    os.makedirs(root, exist_ok=True)
    targz = os.path.join(root, "ArabAndalusianDataset.tar.gz")
    copy_from_drive_with_progress(
        "/content/drive/MyDrive/datasets/Arab_Andalusian/ArabAndalusianDataset.tar.gz", targz)
    extract_dir = os.path.join(root, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(targz, "r:gz") as tf:
        tf.extractall(extract_dir, filter="data")

    archive_root = os.path.join(extract_dir, "ArabAndalusianDataset")
    if not os.path.isdir(archive_root):
        archive_root = extract_dir
    documents_root = os.path.join(archive_root, "documents")

    nawba_lookup = {e["id"]: e for e in load_json_file(
        os.path.join(archive_root, "andalusian_nawba.json"))["results"]}
    description = load_json_file(os.path.join(archive_root, "andalusian_description.json"))

    pairs = []
    for entry in description:
        mbid = entry.get("mbid")
        sections = entry.get("sections", [])
        if not sections:
            continue
        primary_nawba_id = sections[0].get("nawba", {}).get("id")
        nawba_name = nawba_lookup.get(primary_nawba_id, {}).get("transliterated_name")
        if nawba_name is None:
            continue
        audio_path = os.path.join(documents_root, mbid, f"{mbid}.mp3")
        if os.path.isfile(audio_path):
            pairs.append((audio_path, nawba_name))
    print(f"[{name_prefix}] {len(pairs)} (audio, nawba) pairs found.")

    n_cached = cache_fidelity_corpus(name_prefix, pairs)
    shutil.rmtree(root, ignore_errors=True)
    print_disk_usage()
    return n_cached


def main():
    ensure_project_dir()
    print_disk_usage()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 2: staging CMD (Carnatic raga)...")
    n_cmd = stage_cmd_hmd("Carnatic", "cmd_carnatic")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 3: staging HMD (Hindustani raga)...")
    n_hmd = stage_cmd_hmd("Hindustani", "hmd_hindustani")
    if os.path.exists(_SHARED_RAGADATASET_ZIP):
        os.remove(_SHARED_RAGADATASET_ZIP)  # shared by both CMD and HMD -- safe to delete now
    print_disk_usage()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 4: staging Jingju role-type...")
    n_jingju = stage_jingju_roletype()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 5: staging Arab-Andalusian nawba...")
    n_arab = stage_arab_andalusian_nawba()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] All 4 fidelity corpora staged: "
          f"CMD={n_cmd}, HMD={n_hmd}, Jingju={n_jingju}, Arab-Andalusian={n_arab}.")

    summary = {
        "program": "12_e4_fidelity_staging.py",
        "n_cmd_carnatic": n_cmd, "n_hmd_hindustani": n_hmd,
        "n_jingju_roletype": n_jingju, "n_arab_andalusian_nawba": n_arab,
        "elapsed_seconds": round(time.time() - RUN_START_TIME, 1),
    }
    local_summary_path = os.path.join(LOCAL_ROOT, "12_summary.json")
    with open(local_summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    copy_to_drive(local_summary_path, "12_summary.json")

    return summary


if __name__ == "__main__":
    summary = main()
    elapsed = time.time() - RUN_START_TIME
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Program 12_e4_fidelity_staging.py finished.")
    print(f"[TOTAL RUNNING TIME] {time.strftime('%H:%M:%S', time.gmtime(elapsed))} ({elapsed:.1f}s)")