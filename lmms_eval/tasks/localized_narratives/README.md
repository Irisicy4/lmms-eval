# Localized Narratives (test split)

This task evaluates open-ended image captioning on the **test split** of the
[Localized Narratives](https://google.github.io/localized-narratives/) benchmark.
Each Localized Narrative is a fluent, free-form description of an image that
was spoken aloud while the annotator hovered the mouse over the regions being
described, producing a synchronized voice + trace + transcript annotation.

We score generations against the transcript (one reference caption per
image) with the standard COCO caption metrics: BLEU-1..4, METEOR, ROUGE-L,
and CIDEr.

## Data source

We pull the annotations from
[`coastalcph/LocalizedNarratives`](https://huggingface.co/datasets/coastalcph/LocalizedNarratives),
a parquet-only mirror of the official release (the upstream
`google/localized-narratives` loading script is no longer compatible with
modern `datasets`). The test split contains 126,020 narratives, all sourced
from Open Images (`dataset_id == "open_image"`), so images are fetched on
demand from the public Open Images CDN at
`https://s3.amazonaws.com/open-images-dataset/test/<image_id>.jpg`.

> Note on the "COCO subset": Localized Narratives spans four image sources
> (COCO, Open Images, Flickr30k, ADE20K) across its full release, but the
> COCO portion lives in the training splits only. The
> `coastalcph/LocalizedNarratives` parquet test split is therefore evaluated
> in its entirety (Open Images images). If a future mirror exposes a COCO
> test subset, a `process_docs` filter on `dataset_id == "mscoco"` can be
> added in the YAML.

## Task definition

- **Output type:** `generate_until`
- **Prompt:** `"Describe the image briefly."`
- **References:** single ground-truth caption per image (the LN transcript)
- **Metrics:** `coco_Bleu_{1..4}`, `coco_METEOR`, `coco_ROUGE_L`, `coco_CIDEr`
  (computed via `pycocoevalcap`, shared with `coco_cap`)

## Files

- `localized_narratives_test.yaml` — task config
- `utils.py` — `ln_doc_to_visual`, `ln_doc_to_text`, `ln_process_result`,
  plus re-exports of the `coco_cap` aggregator helpers

## Running

```bash
python3 -m lmms_eval \
    --model <model_name> \
    --tasks localized_narratives_test \
    --batch_size 1
```

Image downloads are issued on the fly from the Open Images CDN; make sure
the eval host has outbound HTTPS.

## Citation

```bibtex
@inproceedings{PontTuset_eccv2020,
  author    = {Jordi Pont-Tuset and Jasper Uijlings and Soravit Changpinyo
               and Radu Soricut and Vittorio Ferrari},
  title     = {Connecting Vision and Language with Localized Narratives},
  booktitle = {ECCV},
  year      = {2020}
}
```
