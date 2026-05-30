# DOCCI (Descriptions of Connected and Contrasting Images)

DOCCI is a dataset of 15k images paired with long, detailed, human-written
English descriptions designed to capture key visual challenges such as spatial
relations, counting, text rendering, world knowledge and fine-grained
properties.

This task wires up the public **test** split (~5,000 image / description
pairs) as an image-captioning benchmark inside `lmms-eval`.

## Source

* Official dataset: [`google/docci`](https://huggingface.co/datasets/google/docci)
  (loading script form — currently incompatible with newer `datasets` releases).
* Parquet mirror used here: [`not-lain/docci`](https://huggingface.co/datasets/not-lain/docci).

Each row of the parquet mirror exposes:

| column   | type   | notes                                                       |
| -------- | ------ | ----------------------------------------------------------- |
| `image`  | PIL    | RGB image.                                                  |
| `question` | str  | Fixed prompt shipped with the dataset (`"Describe this image"`). |
| `answer` | str    | The long-form human description (single ground truth).      |

## Task

* `docci_test` — generate a caption for each test image and score it against
  the single reference description using BLEU-1/4, METEOR, ROUGE-L and CIDEr
  (computed via `pycocoevalcap`, exactly like `coco_karpathy_test`).

The prompt sent to the model is `"Describe the image briefly."` to keep
parity with `coco_karpathy_test` and other captioning tasks in the suite.

## Usage

```bash
python3 -m lmms_eval \
    --model <your_model> \
    --tasks docci_test \
    --batch_size 1 \
    --limit 8
```

## Citation

```bibtex
@inproceedings{OnoeDocci2024,
  author    = {Yasumasa Onoe and Sunayana Rane and Zachary Berger and
               Yonatan Bitton and Jaemin Cho and Roopal Garg and
               Alexander Ku and Zarana Parekh and Jordi Pont-Tuset and
               Garrett Tanzer and Su Wang and Jason Baldridge},
  title     = {{DOCCI}: Descriptions of Connected and Contrasting Images},
  booktitle = {ECCV},
  year      = {2024}
}
```
