import json
import os

from loguru import logger as eval_logger
from pycocoevalcap.eval import Bleu, Cider, COCOEvalCap, Meteor, Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

dir_name = os.path.dirname(os.path.abspath(__file__))

DOCCI_METRICS = ["Bleu_4", "Bleu_3", "Bleu_2", "Bleu_1", "METEOR", "ROUGE_L", "CIDEr"]


def docci_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def docci_doc_to_text(doc):
    # The dataset ships a per-row question (e.g. "Describe this image") but we
    # keep a short, fixed prompt to stay consistent with the COCO-Karpathy style
    # caption task and avoid drift across rows.
    return "Describe the image briefly."


def docci_process_result(doc, result):
    """
    Args:
        doc: an instance of the eval dataset
        result: [pred]
    Returns:
        a dictionary keyed by metric name, value is the data payload used by
        the aggregator. DOCCI has a single ground-truth description per image,
        so we wrap it in a one-element list to match the COCOEvalCap interface.
    """
    pred = result[0] if len(result) > 0 else ""
    # DOCCI rows do not ship a numeric image id; use the row index that
    # `datasets` exposes if available, otherwise fall back to a hash of the
    # ground truth to give COCOEvalCap a unique integer per example.
    image_id = doc.get("id", None)
    if image_id is None:
        image_id = abs(hash(doc["answer"])) % (10**12)
    image_id = int(image_id) if not isinstance(image_id, int) else image_id

    data_dict = {
        "answer": [doc["answer"]],
        "pred": pred,
        "image_id": image_id,
        "id": image_id,
    }

    return {f"docci_{metric}": data_dict for metric in DOCCI_METRICS}


def docci_aggregation_result(results, metric, args):
    scorers = [
        (Bleu(4), "Bleu_1"),
        (Bleu(4), "Bleu_2"),
        (Bleu(4), "Bleu_3"),
        (Bleu(4), "Bleu_4"),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]
    scorers_dict = {s[1]: s for s in scorers}

    stored_results = []
    dataset = {"annotations": [], "images": []}
    idx = 0
    for result in results:
        stored_results.append({"image_id": int(result["image_id"]), "caption": result["pred"]})
        for a in result["answer"]:
            dataset["annotations"].append({"image_id": int(result["image_id"]), "caption": a, "id": idx})
            idx += 1
        dataset["images"].append({"id": result["image_id"]})

    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()

    coco_result = coco.loadRes(stored_results)
    coco_eval = COCOEvalCap(coco, coco_result)

    imgIds = coco_eval.params["image_id"]
    gts = {}
    res = {}
    for imgId in imgIds:
        gts[imgId] = coco_eval.coco.imgToAnns[imgId]
        res[imgId] = coco_eval.cocoRes.imgToAnns[imgId]

    eval_logger.info("tokenization...")
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    eval_logger.info(f"Computing {metric} scores...")

    score, scores = scorers_dict[metric][0].compute_score(gts, res)
    if type(score) == list:
        n = int(metric.split("_")[-1])
        score = score[n - 1]

    path = generate_submission_file("docci_test_captions_alg_results.json", args)
    if not os.path.exists(path):
        eval_logger.info("Storing prediction that can be submitted ...")
        with open(path, "w") as f:
            json.dump(stored_results, f, indent=4)

    return score


def docci_bleu4(results, args):
    return docci_aggregation_result(results, "Bleu_4", args)


def docci_bleu3(results, args):
    return docci_aggregation_result(results, "Bleu_3", args)


def docci_bleu2(results, args):
    return docci_aggregation_result(results, "Bleu_2", args)


def docci_bleu1(results, args):
    return docci_aggregation_result(results, "Bleu_1", args)


def docci_meteor(results, args):
    return docci_aggregation_result(results, "METEOR", args)


def docci_rougel(results, args):
    return docci_aggregation_result(results, "ROUGE_L", args)


def docci_cider(results, args):
    return docci_aggregation_result(results, "CIDEr", args)
