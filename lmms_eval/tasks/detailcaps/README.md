# DetailCaps-4870

**DetailCaps-4870** (a.k.a. the Detailed Caption Dataset / DCD) is a
detail-image-captioning benchmark introduced in *"Benchmarking and
Improving Detail Image Caption"* (EMNLP 2024, arXiv:2405.19092).

The benchmark contains 4,870 images sourced from COCO, SAM, NoCaps and other
public collections. Each image is paired with three detailed reference
captions produced by strong proprietary VLMs (GPT-4O, GPT-4V, Gemini 1.5
Pro). The captions are notably longer and more attribute-rich than the
one-sentence captions used by COCO Captions, making DetailCaps-4870 a useful
stress test for the long-form descriptive capability of modern LMMs.

- HuggingFace dataset: <https://huggingface.co/datasets/foundation-multimodal-models/DetailCaps-4870>
- Paper: <https://arxiv.org/abs/2405.19092>

## Tasks

This directory exposes two related tasks. They share the same underlying
dataset but differ in their metric stack and prompt.

| Task | Prompt | Metrics |
| ---- | ------ | ------- |
| `detailcaps` | `Describe this image in detail.` | CAPTURE + BLEU-1/2/3/4 + METEOR + ROUGE-L + CIDEr |
| `detailcaps_4870` | `Describe the image in detail.` | BLEU-1, BLEU-4, METEOR, ROUGE-L, CIDEr |

`detailcaps_4870` is a lightweight variant that uses the standard COCO
caption metric stack (no `capture-metric` dependency) and treats all three
GT captions as multi-references for BLEU / METEOR / ROUGE-L / CIDEr.

## Running

```bash
python3 -m lmms_eval \
  --model <your_model> \
  --tasks detailcaps_4870 \
  --batch_size 1 \
  --output_path ./logs/
```

## Citation

```bibtex
@inproceedings{dong2024benchmarking,
  title     = {Benchmarking and Improving Detail Image Caption},
  author    = {Dong, Hongyuan and Wu, Jiawen and Li, Bohong and Zhong, Jixing
               and Cui, Yang and Wang, Yiding and Wei, Jichao and Wang, Yifeng
               and Wang, Yibo and others},
  booktitle = {Proceedings of the 2024 Conference on Empirical Methods in
               Natural Language Processing (EMNLP)},
  year      = {2024},
  eprint    = {2405.19092},
  archivePrefix = {arXiv}
}
```
