'''
J. Elizabeth Liebl; jliebl@gmu.edu
03/09/2026
ASR for ICLDC
Gen Transcript-Audio file dataset
'''

import csv
import os
import re
from pathlib import Path
import pandas as pd

# Precompile regex once
ITEM_SPLIT_RE = re.compile(r"\s+(?=item\s\[\d\]:)")
INTERVAL_SPLIT_RE = re.compile(r"\s+(?=intervals\s\[\d+\]:)")
PHONE_RE = re.compile(r"(?<=\s{12}text\s=\s['\"]).*?(?=['\"])")

BASE_DIR = Path("/scratch/jliebl/ASR_ComputEL")
ID_FILE = BASE_DIR / "ASR_ComputEL/mdc_ids3.tsv"
DATA_DIR = BASE_DIR / "CV_VC_Data"
OUT_DIR = DATA_DIR / "Dataframes" / "full"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def path_audiofile(tg_file: Path) -> str:
    """
    Convert a TextGrid path like:
    /.../lang_vc/foo.TextGrid
    to:
    /.../lang_cv/clips/foo.mp3
    """
    s = str(tg_file)
    s = s.replace("_vc/", "_cv/clips/")
    s = s.replace(".TextGrid", ".mp3")
    return s


def parse_text_grid(text_grid_file: Path) -> str:
    """
    Extract phone-tier transcript from a TextGrid file.
    Returns a concatenated string of phone labels.
    """
    with open(text_grid_file, "r", encoding="utf-8", errors="ignore") as f:
        tg = f.read()

    # Normalize quotes only once
    tg = tg.replace('"', "'")

    sections = ITEM_SPLIT_RE.split(tg)

    # Guard against malformed files
    if len(sections) < 3:
        return ""

    intervals = INTERVAL_SPLIT_RE.split(sections[2])

    phones = []
    for interval in intervals[1:]:
        match = PHONE_RE.search(interval)
        if match:
            phones.append(match.group())

    return "".join(phones)


id_df = pd.read_csv(ID_FILE, sep="\t")
langs = id_df["CV_ID"].dropna().unique()

for l in langs:
    tg_folder = DATA_DIR / f"{l}_vc"
    out_file = OUT_DIR / f"{l}_data.tsv"

    print(f"Processing {l}...")

    if not tg_folder.exists():
        print(f"Skipping {l}: folder not found -> {tg_folder}")
        continue

    if out_file.exists():
        print(f"Skipping {l}: output already exists")
        continue
    
    tg_iter = tg_folder.glob("*.TextGrid")

    count = 0
    with open(out_file, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout, delimiter="\t")
        writer.writerow(["Audio", "Transcript"])

        for tg_path in tg_iter:
            audio_path = path_audiofile(tg_path)
            transcript = parse_text_grid(tg_path)
            writer.writerow([audio_path, transcript])

            count += 1
            if count % 10000 == 0:
                print(f"{l}: processed {count} TextGrids")

    print(f"Finished {l}: wrote {count} rows to {out_file}")