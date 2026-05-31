# PROMPTS

This file records the major Codex prompts and actions used for the assignment. Codex was the only AI coding tool used.

No secrets, API keys, tokens, or credentials are included.

## Prompt Log

The exact full conversation is not reproduced here. Items marked `reconstructed` summarize the user's prompts from the confirmed interaction history.

1. `reconstructed`: Inspect the FLOSS repository before modifying or running anything. Read README, requirements, setup files, scripts, configs, and demo/evaluation instructions. Identify baseline, FLOSS-applied workflow, dataset support, expected inputs/outputs, and recommend the smallest experiment.

   Action: Codex inspected local files only. It identified `tools/test.py`, `tools/eval_naclip.py`, `configs/`, `rankings/`, and `notebooks/FLOSS_Reproducibility.ipynb`. No installation or evaluation was run.

2. `reconstructed`: Inspect the notebook workflow in detail before running anything. Focus on notebook downloads, inputs, outputs, `NUM_VAL_IMAGES`, CPU feasibility, repository modifications, and execution plan.

   Action: Codex inspected `notebooks/FLOSS_Reproducibility.ipynb`, `requirements.txt`, `tools/test.py`, `configs/clipdinoiser.py`, and `rankings/clipdinoiser/`. It found broad `snapshot_download` usage in the notebook and recommended a safer selective-download smoke test.

3. `reconstructed`: Prepare a minimal CPU-safe, storage-safe FLOSS smoke-test workflow without modifying the original notebook. Use `NUM_IMAGES = 5`, `NUM_VAL_IMAGES = 5`, avoid broad Hugging Face downloads, and write outputs under `results/floss_smoke_test/`.

   Action: Codex created `scripts/run_floss_smoke_test.py`, `scripts/README_smoke_test.md`, and `results/floss_smoke_test/.gitkeep`.

4. `reconstructed`: Inspect the new smoke-test files and add a `--dry-run` option if absent.

   Action: Codex added `--dry-run` to the script. The dry-run lists selected Hugging Face files and output paths without downloading feature files, running FLOSS computation, or writing metrics.

5. `reconstructed`: Run only the dry-run command.

   Action: The first dry-run failed because `huggingface_hub` was missing. After approval, Codex installed only `huggingface_hub` and ran the dry-run. The dry-run selected exactly 16 files.

6. `reconstructed`: Run the real smoke test once on CPU, downloading only the 16 confirmed files.

   Action: Codex ran `python scripts/run_floss_smoke_test.py --device cpu`. Missing dependencies were installed only as reported: `numpy`, CPU `torch`, `Pillow`, and `matplotlib`. The smoke test completed and generated outputs under `results/floss_smoke_test/`.

7. `reconstructed`: Create final submission documentation: `README.md`, `PROMPTS.md`, `REPORT.md`, and `EMAIL_DRAFT.txt`.

   Action: Codex generated the final assignment documents using only confirmed execution history and generated files.

## Commands Confirmed

Dry-run:

```bash
python scripts/run_floss_smoke_test.py --dry-run
```

Real smoke test:

```bash
python scripts/run_floss_smoke_test.py --device cpu
```

## Confirmed Result Summary

| Method | mIoU |
|---|---:|
| Baseline averaged-template | 20.3923 |
| FLOSS expert-fusion | 23.3612 |
| Delta | +2.9689 |

