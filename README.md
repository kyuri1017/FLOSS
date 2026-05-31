# FLOSS Smoke-Test Reproduction Assignment

This repository contains an assignment-oriented smoke-test reproduction for the paper:

**FLOSS: Free Lunch in Open-vocabulary Semantic Segmentation**

The experiment was inspected, prepared, and documented using **Codex as the only AI coding tool**.

## Scope

This is **not** a full paper benchmark reproduction. It is a minimal smoke test designed to verify the FLOSS idea on a tiny Cityscapes subset using precomputed CLIP-DINOiser features.

The original notebook was preserved:

```text
notebooks/FLOSS_Reproducibility.ipynb
```

No edits were made to that notebook.

## Workflow Types

Full dataset evaluation:

- Uses repository evaluation entry points such as `tools/test.py`.
- Requires raw datasets under `data/`, CLIP-DINOiser checkpoint files, and larger compute.
- Was inspected but not used for this assignment smoke test.

Dry-run:

```bash
python scripts/run_floss_smoke_test.py --dry-run
```

- Lists the exact Hugging Face files that would be downloaded.
- Prints planned output paths.
- Does not download feature files.
- Does not run FLOSS computation.
- Does not write metrics.

Smoke-test execution:

```bash
python scripts/run_floss_smoke_test.py --device cpu
```

- Downloads or reuses only 16 selected Hugging Face files.
- Uses 5 train feature files and 5 validation feature files.
- Runs CPU-compatible baseline and FLOSS expert-fusion evaluation.
- Writes outputs under `results/floss_smoke_test/`.

## Confirmed Download Manifest

The smoke test downloaded/reused only these 16 selected Hugging Face files from `yasserben/floss-features`:

```text
cityscapes_text_features.pt
vision_features_train/aachen_000000_000019_leftImg8bit.pt
vision_features_train/aachen_000001_000019_leftImg8bit.pt
vision_features_train/aachen_000002_000019_leftImg8bit.pt
vision_features_train/aachen_000003_000019_leftImg8bit.pt
vision_features_train/aachen_000004_000019_leftImg8bit.pt
vision_features_val/frankfurt_000000_000294_leftImg8bit.pt
vision_features_val/frankfurt_000000_000576_leftImg8bit.pt
vision_features_val/frankfurt_000000_001016_leftImg8bit.pt
vision_features_val/frankfurt_000000_001236_leftImg8bit.pt
vision_features_val/frankfurt_000000_001751_leftImg8bit.pt
gt_val/frankfurt_000000_000294_gtFine_labelTrainIds.png
gt_val/frankfurt_000000_000576_gtFine_labelTrainIds.png
gt_val/frankfurt_000000_001016_gtFine_labelTrainIds.png
gt_val/frankfurt_000000_001236_gtFine_labelTrainIds.png
gt_val/frankfurt_000000_001751_gtFine_labelTrainIds.png
```

The exact manifest is saved at:

```text
results/floss_smoke_test/download_manifest.json
```

## Results

Confirmed smoke-test results:

| Method | mIoU |
|---|---:|
| Baseline averaged-template | 20.3923 |
| FLOSS expert-fusion | 23.3612 |
| Delta | +2.9689 |

These values come from:

```text
results/floss_smoke_test/metrics.json
```

## Generated Files

```text
results/floss_smoke_test/download_manifest.json
results/floss_smoke_test/execution.log
results/floss_smoke_test/metrics.json
results/floss_smoke_test/per_class_iou.csv
results/floss_smoke_test/rankings_smoke.json
results/floss_smoke_test/scatter_mosaic.png
```

## Dependency Notes

The environment initially lacked several dependencies. Only missing dependencies reported during execution were installed as needed:

- `huggingface_hub` missing: installed for dry-run file listing.
- `numpy` missing: installed for runtime array handling.
- `torch` missing: installed CPU PyTorch only.
- `Pillow` missing: installed for ground-truth PNG loading.
- `matplotlib` missing: installed for `scatter_mosaic.png`.

The full `requirements.txt` was not installed.

## Submission Documents

```text
README.md
PROMPTS.md
REPORT.md
EMAIL_DRAFT.txt
scripts/README_smoke_test.md
scripts/run_floss_smoke_test.py
```

