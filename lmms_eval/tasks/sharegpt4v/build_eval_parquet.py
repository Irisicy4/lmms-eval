"""Reproducibility script for the ShareGPT4V-COCO eval parquet.

Usage:
    HF_TOKEN=<token> python build_eval_parquet.py

Steps:
1. Load the official ShareGPT4V instruct caption JSON (100K rows) from
   ``Lin-Chen/ShareGPT4V``.
2. Keep entries whose ``image`` path starts with ``coco/`` (all happen to be
   COCO train2017).
3. For ``N_SAMPLES`` entries, fetch the matching COCO 2017 image straight
   from the public ``images.cocodataset.org`` mirror.
4. Save a single parquet ``data/test-00000-of-00001.parquet`` with columns
   ``image`` (PIL.Image), ``image_id`` (str) and ``caption`` (str -- the
   ShareGPT4V GT).
5. Push it to the HF Hub dataset repo ``Icey444/ShareGPT4V-COCO-eval``.
"""

import io
import json
import os

import requests
from datasets import Dataset, Features, Image, Value
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image as PILImage

HF_TOKEN = os.environ["HF_TOKEN"]
REPO_ID = "Icey444/ShareGPT4V-COCO-eval"
N_SAMPLES = 500
OUT_DIR = "/tmp/sharegpt4v_build"
PARQUET_PATH = os.path.join(OUT_DIR, "test-00000-of-00001.parquet")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    json_path = hf_hub_download(
        repo_id="Lin-Chen/ShareGPT4V",
        filename="sharegpt4v_instruct_gpt4-vision_cap100k.json",
        repo_type="dataset",
        token=HF_TOKEN,
    )
    print(f"loaded ShareGPT4V JSON: {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    coco_entries = [x for x in data if x.get("image", "").startswith("coco/")]
    print(f"total rows: {len(data)}; coco subset: {len(coco_entries)}")

    rows = []
    seen_ids = set()
    session = requests.Session()
    for entry in coco_entries:
        if len(rows) >= N_SAMPLES:
            break
        rel_path = entry["image"]  # e.g. coco/train2017/000000000009.jpg
        split = rel_path.split("/")[1]
        filename = rel_path.split("/")[-1]
        image_id = filename.split(".")[0]
        if image_id in seen_ids:
            continue
        url = f"http://images.cocodataset.org/{split}/{filename}"
        try:
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
            img = PILImage.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            print(f"  [skip] {url}: {e}")
            continue

        caption = entry["conversations"][1]["value"].strip()
        rows.append({"image": img, "image_id": image_id, "caption": caption})
        seen_ids.add(image_id)
        if len(rows) % 25 == 0:
            print(f"  fetched {len(rows)} / target {N_SAMPLES}")
    print(f"built {len(rows)} rows")

    features = Features(
        {
            "image": Image(),
            "image_id": Value("string"),
            "caption": Value("string"),
        }
    )
    ds = Dataset.from_list(rows, features=features)
    ds.to_parquet(PARQUET_PATH)
    print(f"wrote parquet: {PARQUET_PATH} ({os.path.getsize(PARQUET_PATH) / 1e6:.1f} MB)")

    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True, private=False)
    api.upload_file(
        path_or_fileobj=PARQUET_PATH,
        path_in_repo="data/test-00000-of-00001.parquet",
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print(f"uploaded to https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
