"""Utility functions for the Localized Narratives test-split caption task.

The dataset (``coastalcph/LocalizedNarratives``) only ships the textual
narrative annotations; images must be fetched on the fly. The test split is
entirely composed of Open Images (``dataset_id == "open_image"``), so we
resolve each ``image_id`` to its public Open Images URL and download the JPEG.

Captioning metrics (BLEU-1..4 / METEOR / ROUGE-L / CIDEr) are computed with
the standard ``pycocoevalcap`` machinery, reusing the COCO-caption
aggregator implementation. Each Localized Narrative is treated as a single
reference per image, which matches how the LN benchmark is typically scored.
"""

from io import BytesIO

import requests
from PIL import Image

# Re-export the COCO caption aggregator helpers so the YAML can wire them in
# directly via ``utils.coco_bleu1`` etc. Keeping a single source of truth for
# the metric implementation avoids drift between the two tasks.
from lmms_eval.tasks.coco_cap.utils import (
    coco_bleu1,
    coco_bleu2,
    coco_bleu3,
    coco_bleu4,
    coco_cider,
    coco_meteor,
    coco_rougel,
)

__all__ = [
    "ln_doc_to_visual",
    "ln_doc_to_text",
    "ln_process_result",
    "coco_bleu1",
    "coco_bleu2",
    "coco_bleu3",
    "coco_bleu4",
    "coco_meteor",
    "coco_rougel",
    "coco_cider",
]

OPEN_IMAGES_URL_TEMPLATE = "https://s3.amazonaws.com/open-images-dataset/test/{image_id}.jpg"


def _open_images_url(image_id: str) -> str:
    return OPEN_IMAGES_URL_TEMPLATE.format(image_id=image_id)


def _image_id_to_int(image_id: str) -> int:
    """Map a hex Open Images image_id to a stable integer for COCO-API indexing."""
    return int(image_id, 16)


def ln_doc_to_visual(doc):
    """Fetch the Open Images JPEG for this narrative and return a PIL image list."""
    url = _open_images_url(doc["image_id"])
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    return [image.convert("RGB")]


def ln_doc_to_text(doc):
    return "Describe the image briefly."


def ln_process_result(doc, result):
    """Format the model prediction into the dict expected by the COCO aggregator.

    Each Localized Narrative provides a single ground-truth caption per
    image, so the ``answer`` field is a list of one reference.
    """
    pred = result[0] if len(result) > 0 else ""
    image_id_str = doc["image_id"]
    image_id_int = _image_id_to_int(image_id_str)

    data_dict = {
        "answer": [doc["caption"]],
        "pred": pred,
        "image_id": image_id_int,
        "id": image_id_str,
    }
    metrics = ["Bleu_4", "Bleu_3", "Bleu_2", "Bleu_1", "METEOR", "ROUGE_L", "CIDEr"]
    return {f"coco_{metric}": data_dict for metric in metrics}
