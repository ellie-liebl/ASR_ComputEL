'''
J. Elizabeth Liebl; jliebl@gmu.edu
03/08/2026
ASR for ComputEL
Download VC Data
'''
import pandas as pd
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

repo_id = "pacscilab/VoxCommunis"
repo_type = "dataset"

tsv_path = "/scratch/jliebl/ASR_ComputEL/ASR_ComputEL/mdc_ids3.tsv"
out_dir = Path("/scratch/jliebl/ASR_ComputEL/CV_VC_Data")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(tsv_path, sep="\t")
df["CV_ID"] = df["CV_ID"].fillna("").astype(str).str.strip()
df["VC_ID"] = df["VC_ID"].fillna("").astype(str).str.strip()

api = HfApi()
repo_files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)

# Keep only textgrid zip archives
textgrid_files = [
    f for f in repo_files
    if f.startswith("textgrids/") and f.endswith(".zip")
]

for _, row in df.iterrows():
    cv_id = row["CV_ID"]
    vc_id = row["VC_ID"]

    if not vc_id:
        print(f"Skipping missing VC_ID for CV_ID={cv_id}", flush=True)
        continue

    # Match e.g. textgrids/ab_xpf_textgrids17_acoustic16.zip
    matches = [
        f for f in textgrid_files
        if Path(f).name.startswith(f"{vc_id}_textgrids")
    ]

    if len(matches) == 0:
        print(f"No match found for {cv_id} | {vc_id}", flush=True)
        continue

    if len(matches) > 1:
        print(f"Multiple matches for {cv_id} | {vc_id}: {matches}", flush=True)
        # pick the first one, or handle manually
        target = matches[0]
    else:
        target = matches[0]

    local_file = out_dir / Path(target).name
    if local_file.exists():
        print(f"Already downloaded, skipping: {local_file.name}", flush=True)
        continue

    print(f"Downloading {target}", flush=True)
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=target,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
    )
    print(f"Saved to {path}", flush=True)