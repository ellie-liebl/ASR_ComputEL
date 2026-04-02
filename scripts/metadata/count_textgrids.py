'''
J. Elizabeth Liebl; jliebl@gmu.edu
03/03/2026
ASR for ComputEL
GPT generated textgrid counting
'''
import re
import struct
import requests
import pandas as pd
from huggingface_hub import list_repo_files, hf_hub_url

REPO_ID = "pacscilab/VoxCommunis"
REPO_TYPE = "dataset"
FOLDER = "textgrids/"
TAIL_BYTES = 1024 * 1024  # 1MB tail fetch (usually enough to include EOCD)

EOCD_SIG = b"PK\x05\x06"
CD_SIG   = b"PK\x01\x02"


def _find_eocd(tail: bytes) -> int:
    idx = tail.rfind(EOCD_SIG)
    if idx == -1:
        raise ValueError("EOCD not found in tail chunk (increase TAIL_BYTES).")
    return idx


def _parse_eocd(eocd: bytes):
    # EOCD layout (little-endian):
    # sig(4), disk(2), cd_disk(2), disk_entries(2), total_entries(2),
    # cd_size(4), cd_offset(4), comment_len(2)
    if eocd[:4] != EOCD_SIG:
        raise ValueError("Not an EOCD record.")
    fields = struct.unpack_from("<4sHHHHIIH", eocd, 0)
    _, disk, cd_disk, disk_entries, total_entries, cd_size, cd_offset, comment_len = fields
    return {
        "disk_entries": disk_entries,
        "total_entries": total_entries,
        "cd_size": cd_size,
        "cd_offset": cd_offset,
        "comment_len": comment_len,
    }


def _count_textgrids_in_cd(cd: bytes) -> int:
    i = 0
    count = 0
    n = len(cd)

    while i + 46 <= n:
        if cd[i:i+4] != CD_SIG:
            break

        fn_len, extra_len, com_len = struct.unpack_from("<HHH", cd, i + 28)
        fn_start = i + 46
        fn_end = fn_start + fn_len

        filename = cd[fn_start:fn_end]
        if filename.endswith(b".TextGrid"):
            count += 1

        i = fn_end + extra_len + com_len

    return count


def _head_content_length(url: str) -> int:
    r = requests.head(url, allow_redirects=True, timeout=60)
    r.raise_for_status()
    cl = r.headers.get("Content-Length")
    if cl is None:
        raise ValueError("No Content-Length header; cannot range-fetch reliably.")
    return int(cl)


def _range_get(url: str, start: int = None, end: int = None, suffix: int = None) -> bytes:
    headers = {}
    if suffix is not None:
        headers["Range"] = f"bytes=-{suffix}"
    elif start is not None and end is not None:
        headers["Range"] = f"bytes={start}-{end}"
    else:
        raise ValueError("Provide either suffix or start+end.")

    r = requests.get(url, headers=headers, allow_redirects=True, timeout=120)
    r.raise_for_status()
    return r.content


def _extract_pipeline_tag(zip_filename: str) -> str:
    # e.g. am_epi_textgrids20_acoustic20.zip -> am_epi
    return re.split(r"_textgrids", zip_filename, maxsplit=1)[0]


def voxcommunis_textgrid_counts_df(
    repo_id: str = REPO_ID,
    folder: str = FOLDER,
    tail_bytes: int = TAIL_BYTES,
) -> pd.DataFrame:
    files = list_repo_files(repo_id, repo_type=REPO_TYPE)
    zips = sorted([f for f in files if f.startswith(folder) and f.endswith(".zip")])

    rows = []
    for path in zips:
        zip_name = path.split("/")[-1]
        pipeline_tag = _extract_pipeline_tag(zip_name)

        url = hf_hub_url(repo_id, path, repo_type=REPO_TYPE)
        size = _head_content_length(url)

        # tail fetch -> EOCD
        tail = _range_get(url, suffix=min(tail_bytes, size))
        eocd_pos = _find_eocd(tail)
        eocd = tail[eocd_pos:eocd_pos+22]
        e = _parse_eocd(eocd)

        # central directory fetch
        cd = _range_get(url, start=e["cd_offset"], end=e["cd_offset"] + e["cd_size"] - 1)
        tg_count = _count_textgrids_in_cd(cd)

        rows.append(
            {
                "pipeline_tag": pipeline_tag,
                "zip_path": path,
                "zip_filename": zip_name,
                "zip_size_bytes": size,
                "zip_total_entries": e["total_entries"],
                "textgrid_count": tg_count,
            }
        )

    df = pd.DataFrame(rows).sort_values(["pipeline_tag"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = voxcommunis_textgrid_counts_df()
    print(df.head())
    # Optional: save
    df.to_csv("ASR_ICLDC/rel_langs_tg_counts.tsv", sep="\t", index=False)