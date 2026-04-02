'''
J. Elizabeth Liebl; jliebl@gmu.edu
03/08/2026
ASR for ComputEL
Unzip data
'''
import tarfile
import zipfile
from pathlib import Path

base_dir = Path("/scratch/jliebl/ASR_ComputEL/CV_VC_Data")

for item in base_dir.iterdir():

    # skip hidden directories like .cache
    if item.name.startswith("."):
        continue

    if item.suffixes == [".tar", ".gz"]:
        # Example: ab_cv.tar.gz -> ab_cv
        out_dir = base_dir / item.name.replace(".tar.gz", "")
        out_dir.mkdir(exist_ok=True)

        print(f"Extracting {item.name} -> {out_dir}", flush=True)

        with tarfile.open(item, "r:gz") as tar:
            tar.extractall(out_dir)

    elif item.suffix == ".zip":
        # Example: ab_vc.zip -> ab_vc
        out_dir = base_dir / item.stem
        out_dir.mkdir(exist_ok=True)

        print(f"Extracting {item.name} -> {out_dir}", flush=True)

        with zipfile.ZipFile(item, "r") as zip_ref:
            zip_ref.extractall(out_dir)

print("Done.")