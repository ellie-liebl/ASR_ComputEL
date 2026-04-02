'''
J. Elizabeth Liebl; jliebl@gmu.edu
03/07/2026
ASR for ComputEL
Download CV Data
'''

from datacollective.datasets import save_dataset_to_disk
from datacollective.errors import DownloadError
import pandas as pd
from tqdm import tqdm
import time
import requests

df = pd.read_csv("/scratch/jliebl/ASR_ComputEL/ASR_ComputEL/mdc_ids3.tsv", sep="\t")

data_ids = df["Dataset_ID"].astype(str).str.strip().tolist()
names = df["CV_ID"].astype(str).tolist()

for d, n in tqdm(list(zip(data_ids, names)), total=len(data_ids)):
    print(f"Starting: {n} | {d}", flush=True)

    success = False

    for attempt in range(10):
        try:
            save_dataset_to_disk(d)
            print(f"Finished: {n} | {d}", flush=True)
            success = True
            break

        except requests.exceptions.HTTPError as e:
            resp = e.response

            if resp is not None and resp.status_code == 429:
                limit = resp.headers.get("X-RateLimit-Limit", "MISSING")
                remaining = resp.headers.get("X-RateLimit-Remaining", "MISSING")
                retry_after = resp.headers.get("Retry-After", "MISSING")

                print(f"RATE LIMIT HIT for {n} ({d})", flush=True)
                print(f"  X-RateLimit-Limit: {limit}", flush=True)
                print(f"  X-RateLimit-Remaining: {remaining}", flush=True)
                print(f"  Retry-After: {retry_after}", flush=True)

                try:
                    wait = int(retry_after)
                except (TypeError, ValueError):
                    wait = 600  # fallback: 10 minutes

                print(f"Waiting {wait} seconds before retry...", flush=True)
                time.sleep(wait)
                continue

            status = resp.status_code if resp is not None else "unknown"
            print(f"HTTPError for {n} ({d}): status={status}", flush=True)
            if resp is not None:
                print(resp.text[:1000], flush=True)
            break

        except RuntimeError as e:
            # Fallback in case datacollective converts 429 to RuntimeError
            if "Rate limit exceeded" in str(e):
                print(f"RATE LIMIT HIT for {n} ({d})", flush=True)
                print("  X-RateLimit-Limit: unavailable (wrapped RuntimeError)", flush=True)
                print("  X-RateLimit-Remaining: unavailable (wrapped RuntimeError)", flush=True)
                print("  Retry-After: unavailable (wrapped RuntimeError)", flush=True)

                wait = 600
                print(f"Waiting {wait} seconds before retry...", flush=True)
                time.sleep(wait)
                continue
            else:
                print(f"RuntimeError for {n} ({d}): {e}", flush=True)
                break

        except DownloadError as e:
            wait = 60 * (attempt + 1)
            print(f"DownloadError for {n} ({d}): {e}", flush=True)
            print(f"Waiting {wait} seconds, then retrying resume...", flush=True)
            time.sleep(wait)

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ) as e:
            wait = 60 * (attempt + 1)
            print(f"Network error for {n} ({d}): {type(e).__name__}: {e}", flush=True)
            print(f"Waiting {wait} seconds, then retrying...", flush=True)
            time.sleep(wait)

        except Exception as e:
            print(f"Unexpected error for {n} ({d}): {type(e).__name__}: {e}", flush=True)
            break

    if not success:
        print(f"Skipping after retries: {n} | {d}", flush=True)

    time.sleep(1.2)