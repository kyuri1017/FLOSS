# FLOSS Smoke Test Workflow

This assignment workflow uses a standalone script instead of copying
`notebooks/FLOSS_Reproducibility.ipynb`.

Reason: the original notebook uses `snapshot_download` with broad patterns that
can fetch all train and validation feature files. The script in this directory
uses `list_repo_files` plus `hf_hub_download` to select and download only the
small 5-image subset required for a CPU-oriented smoke test.

## Created Files

- `scripts/run_floss_smoke_test.py`: selective-download FLOSS smoke-test script.
- `scripts/README_smoke_test.md`: this runbook.
- `results/floss_smoke_test/.gitkeep`: placeholder output directory.

The original notebook is intentionally unchanged:

- `notebooks/FLOSS_Reproducibility.ipynb`

## Smoke-Test Settings

The script is configured for:

- `DATASET = "cityscapes"`
- `NUM_IMAGES = 5`
- `NUM_VAL_IMAGES = 5`
- CPU fallback when CUDA is unavailable
- output directory: `results/floss_smoke_test/`

The ranking resolution defaults to `(512, 256)` to reduce CPU and memory cost.
This is intentionally smaller than the notebook default and should be reported
as a smoke-test approximation, not a full reproduction of the paper metrics.

## Files Downloaded After Approval

The script first inspects the Hugging Face dataset file list from:

```text
yasserben/floss-features
```

It then downloads only:

- `cityscapes_text_features.pt`
- first 5 sorted files from `vision_features_train/*.pt`
- first 5 sorted files from `vision_features_val/*.pt`
- 5 matching files from `gt_val/*.png`

Ground-truth matching follows the notebook convention:

```text
{city}_{seq}_{frame}_leftImg8bit.pt
-> {city}_{seq}_{frame}_gtFine_labelTrainIds.png
```

The script verifies each expected `gt_val` file exists in the Hugging Face file
list before downloading. If the naming convention does not match, it fails
instead of guessing.

## Command To Run After Approval

From the repository root:

Dry-run manifest check. This lists the exact selected Hugging Face files and
planned output paths without downloading feature files or running FLOSS
computation:

```bash
python scripts/run_floss_smoke_test.py --dry-run
```

Real smoke-test run:

```bash
python scripts/run_floss_smoke_test.py
```

Optional explicit CPU run:

```bash
python scripts/run_floss_smoke_test.py --device cpu
```

Optional lower-cost run if needed:

```bash
python scripts/run_floss_smoke_test.py --device cpu --ranking-resolution 256 128
```

## Expected Outputs

All generated outputs are written under:

```text
results/floss_smoke_test/
```

Expected files:

- `execution.log`: command log and printed summary.
- `download_manifest.json`: exact Hugging Face files selected for download.
- `rankings_smoke.json`: entropy ranking computed from 5 train feature files.
- `metrics.json`: baseline mIoU, FLOSS mIoU, deltas, and per-class IoU values.
- `per_class_iou.csv`: per-class baseline/FLOSS IoU table.
- `scatter_mosaic.png`: compact ranking plot for selected classes.
- `hf_cache/`: Hugging Face cache for the selected files only.

## Runtime Risk

Low to medium on CPU. The script only uses 5 training and 5 validation feature
files, but still evaluates all 80 templates for 19 classes. The default ranking
resolution is reduced for CPU safety.

## Storage Risk

Low to medium relative to the original notebook. The script does not call
`snapshot_download` with broad `vision_features_*/*.pt` patterns. It downloads
16 files total: 1 text-feature file, 5 train feature files, 5 validation feature
files, and 5 ground-truth PNGs.

Exact size is not known until Hugging Face metadata/download is available, but
the selected-file strategy avoids pulling the full feature dataset.

## Assignment Use

This smoke test supports the report by producing:

- a reproducible minimal baseline averaged-template mIoU;
- a minimal FLOSS expert-fusion mIoU;
- a per-class IoU comparison table;
- a plot showing entropy-ranked templates;
- an execution log and manifest documenting exactly what was run and downloaded.

Because the run uses only 5 train and 5 validation images with reduced ranking
resolution, results should be described as a sanity-check experiment, not as
paper-level reproduction.
