import json
import os

from loguru import logger as eval_logger
from pycocoevalcap.eval import Bleu, Cider, COCOEvalCap, Meteor, Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

dir_name = os.path.dirname(os.path.abspath(__file__))

SHAREGPT4V_METRICS = ["Bleu_4", "Bleu_3", "Bleu_2", "Bleu_1", "METEOR", "ROUGE_L", "CIDEr"]


def sharegpt4v_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


_SHAREGPT4V_DEFAULT_PROMPT = "Analyze the image in a comprehensive and detailed manner."


def sharegpt4v_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    # Default prompt is the Share-Captioner training prompt from the
    # ShareGPT4V paper (Chen et al., 2023; see ECCV 2024 supplement Appendix E
    # "Details about Share-Captioner"). This is the prompt that reproduces the
    # ShareGPT4V caption distribution that our references come from.
    if lmms_eval_specific_kwargs is None:
        return _SHAREGPT4V_DEFAULT_PROMPT
    return lmms_eval_specific_kwargs.get("prompt", _SHAREGPT4V_DEFAULT_PROMPT)


def sharegpt4v_process_result(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        result: [pred]
    Returns:
        a dict mapping each metric name to the per-sample payload needed
        by the corresponding aggregator.
    """
    pred = result[0] if len(result) > 0 else ""
    # COCO image ids are zero-padded strings in the JSON (e.g. "000000000009").
    # Strip the padding so we get a plain int id that pycocoevalcap is happy with.
    image_id = int(doc["image_id"])
    # Single ShareGPT4V reference, kept as a list for API parity with COCO captions.
    data_dict = {
        "answer": [doc["caption"]],
        "pred": pred,
        "image_id": image_id,
    }
    return {f"sharegpt4v_{metric}": data_dict for metric in SHAREGPT4V_METRICS}


def sharegpt4v_aggregation_result(results, metric, args=None):
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
        dataset["images"].append({"id": int(result["image_id"])})

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
    if isinstance(score, list):
        n = int(metric.split("_")[-1])
        score = score[n - 1]

    path = generate_submission_file("sharegpt4v_coco_eval_alg_results.json", args)
    if not os.path.exists(path):
        eval_logger.info("Storing prediction that can be submitted to the server ...")
        with open(path, "w") as f:
            json.dump(stored_results, f, indent=4)

    return score


def sharegpt4v_bleu4(results, args=None):
    return sharegpt4v_aggregation_result(results, "Bleu_4", args)


def sharegpt4v_bleu3(results, args=None):
    return sharegpt4v_aggregation_result(results, "Bleu_3", args)


def sharegpt4v_bleu2(results, args=None):
    return sharegpt4v_aggregation_result(results, "Bleu_2", args)


def sharegpt4v_bleu1(results, args=None):
    return sharegpt4v_aggregation_result(results, "Bleu_1", args)


def sharegpt4v_meteor(results, args=None):
    return sharegpt4v_aggregation_result(results, "METEOR", args)


def sharegpt4v_rougel(results, args=None):
    return sharegpt4v_aggregation_result(results, "ROUGE_L", args)


def sharegpt4v_cider(results, args=None):
    return sharegpt4v_aggregation_result(results, "CIDEr", args)
