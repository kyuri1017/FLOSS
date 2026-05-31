#!/usr/bin/env python3
"""CPU-safe FLOSS smoke test using a tiny Cityscapes feature subset.

This script intentionally downloads only the files needed for a 5-train-image
ranking pass and a 5-val-image baseline/FLOSS comparison. It does not use the
full MMSegmentation evaluation entry point or require raw Cityscapes files.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DATASET = "cityscapes"
HF_REPO = "yasserben/floss-features"
NUM_IMAGES = 5
NUM_VAL_IMAGES = 5
NUM_CLASSES = 19
IGNORE_INDEX = 255
DEFAULT_OUTPUT_DIR = Path("results/floss_smoke_test")
DEFAULT_CACHE_DIR = DEFAULT_OUTPUT_DIR / "hf_cache"
np = None
torch = None
F = None
Image = None

CLASS_NAMES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

IMAGENET_TEMPLATES = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FLOSS 5-image smoke test.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--num-images", type=int, default=NUM_IMAGES)
    parser.add_argument("--num-val-images", type=int, default=NUM_VAL_IMAGES)
    parser.add_argument(
        "--ranking-resolution",
        type=int,
        nargs=2,
        default=(512, 256),
        metavar=("HEIGHT", "WIDTH"),
        help="Small CPU-safe resolution used only while ranking templates.",
    )
    parser.add_argument("--repo-id", default=HF_REPO)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "List the exact Hugging Face files and output paths without "
            "downloading files or running FLOSS computation."
        ),
    )
    return parser.parse_args()


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "execution.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logging.info("Writing log to %s", log_path)


def get_device(choice: str) -> torch.device:
    require_runtime_dependencies()
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
        return torch.device("cuda")
    if choice == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def require_runtime_dependencies() -> None:
    global F, Image, np, torch
    if torch is not None:
        return
    try:
        import numpy as np_module
        import torch as torch_module
        import torch.nn.functional as f_module
        from PIL import Image as image_module
    except ImportError as exc:
        raise RuntimeError(
            "Missing runtime dependency. Install dependencies only after approval."
        ) from exc
    np = np_module
    torch = torch_module
    F = f_module
    Image = image_module


def require_huggingface_hub():
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: huggingface_hub. Install dependencies only after approval."
        ) from exc
    return hf_hub_download, list_repo_files


def select_remote_files(
    repo_id: str,
    num_train: int,
    num_val: int,
) -> Tuple[List[str], List[str], List[str], str]:
    _, list_repo_files = require_huggingface_hub()
    all_files = set(list_repo_files(repo_id=repo_id, repo_type="dataset"))

    text_file = "cityscapes_text_features.pt"
    if text_file not in all_files:
        raise FileNotFoundError(f"Required Hugging Face file is missing: {text_file}")

    train_files = sorted(
        f
        for f in all_files
        if f.startswith("vision_features_train/") and f.endswith(".pt")
    )[:num_train]
    val_files = sorted(
        f
        for f in all_files
        if f.startswith("vision_features_val/") and f.endswith(".pt")
    )[:num_val]

    if len(train_files) < num_train:
        raise RuntimeError(f"Only found {len(train_files)} train feature files.")
    if len(val_files) < num_val:
        raise RuntimeError(f"Only found {len(val_files)} validation feature files.")

    gt_files = []
    for val_file in val_files:
        stem = Path(val_file).stem
        gt_name = stem.replace("_leftImg8bit", "_gtFine_labelTrainIds") + ".png"
        gt_file = f"gt_val/{gt_name}"
        if gt_file not in all_files:
            raise FileNotFoundError(
                "Could not find matching ground-truth file for "
                f"{val_file}. Expected {gt_file}."
            )
        gt_files.append(gt_file)

    return train_files, val_files, gt_files, text_file


def download_selected_files(
    repo_id: str,
    files: Iterable[str],
    cache_dir: Path,
) -> Dict[str, Path]:
    hf_hub_download, _ = require_huggingface_hub()
    local_paths = {}
    for remote_path in files:
        logging.info("Downloading or reusing %s", remote_path)
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=remote_path,
            cache_dir=str(cache_dir),
        )
        local_paths[remote_path] = Path(local_path)
    return local_paths


def load_features(paths: Sequence[Path]) -> List[torch.Tensor]:
    return [torch.load(path, weights_only=False) for path in paths]


def compute_entropy(probs: torch.Tensor, dim: int = 1, eps: float = 1e-10) -> torch.Tensor:
    return -(probs * torch.log(probs + eps)).sum(dim=dim)


class TemplateRanker:
    def __init__(
        self,
        text_features: Dict[str, torch.Tensor],
        device: torch.device,
        ranking_resolution: Tuple[int, int],
        temperature: float = 0.01,
    ) -> None:
        self.text_features = text_features
        self.device = device
        self.ranking_resolution = ranking_resolution
        self.temperature = temperature
        self.metric_accumulator = defaultdict(lambda: defaultdict(lambda: {"sum": 0.0, "count": 0}))
        self.total_pixels = 0

    def compute_segmentation_probs(self, image_features: torch.Tensor) -> torch.Tensor:
        _, _, _, _ = image_features.shape
        per_template = self.text_features["per_template"].to(image_features.device)
        templates, classes, dim = per_template.shape
        image_features = image_features / (image_features.norm(dim=1, keepdim=True) + 1e-6)
        weights = per_template.reshape(templates * classes, dim)
        logits = F.conv2d(image_features, weights[:, :, None, None])
        logits = logits.reshape(image_features.shape[0], templates, classes, *image_features.shape[-2:])
        probs = F.softmax(logits / self.temperature, dim=2)
        return probs.permute(1, 0, 2, 3, 4)

    def update(self, image_features: torch.Tensor) -> None:
        image_features = image_features.to(self.device)
        probs = self.compute_segmentation_probs(image_features)
        height, width = self.ranking_resolution
        for template_id in range(len(IMAGENET_TEMPLATES)):
            template_probs = F.interpolate(
                probs[template_id],
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            entropy = compute_entropy(template_probs, dim=1)
            predictions = template_probs.argmax(dim=1)
            for class_id in range(NUM_CLASSES):
                mask = predictions == class_id
                if mask.any():
                    values = entropy[mask]
                    self.metric_accumulator[template_id][class_id]["sum"] += values.sum().item()
                    self.metric_accumulator[template_id][class_id]["count"] += int(mask.sum().item())
        self.total_pixels += image_features.shape[0] * height * width

    def rankings(self) -> Dict[str, Dict[str, Dict[str, List[dict]]]]:
        output = {"model": "clipdinoiser", "dataset": DATASET, "split": "train-smoke", "classes": {}}
        for class_id, class_name in enumerate(CLASS_NAMES):
            rows = []
            for template_id in range(len(IMAGENET_TEMPLATES)):
                data = self.metric_accumulator[template_id][class_id]
                if data["count"] > 0:
                    entropy = data["sum"] / data["count"]
                    pixel_percentage = data["count"] / self.total_pixels * 100
                else:
                    entropy = math.inf
                    pixel_percentage = 0.0
                rows.append(
                    {
                        "template_id": template_id,
                        "entropy": entropy,
                        "pixel_percentage": pixel_percentage,
                    }
                )
            rows.sort(key=lambda item: item["entropy"])
            for rank, item in enumerate(rows, start=1):
                item["rank"] = rank
            output["classes"][class_name] = {"entropy_ranking": rows}
        return output


def get_expert_template_ids(rankings: dict, top_k: int = 4) -> List[List[int]]:
    expert_ids = []
    for class_name in CLASS_NAMES:
        ranking = rankings["classes"][class_name]["entropy_ranking"]
        top_ids = [row["template_id"] for row in ranking if row.get("pixel_percentage", 0.0) > 0.0][:top_k]
        expert_ids.append(top_ids or [0])
    return expert_ids


def build_inference_bundle(
    text_features: Dict[str, torch.Tensor],
    expert_template_ids: List[List[int]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    per_template = text_features["per_template"].to(device)
    averaged = text_features["averaged"].to(device)
    expert_classifiers = torch.stack(
        [per_template[expert_template_ids[class_id], :, :].mean(dim=0) for class_id in range(NUM_CLASSES)],
        dim=0,
    )
    return {"averaged": averaged, "expert_classifiers": expert_classifiers}


def compute_predictions_default(
    vision_feat: torch.Tensor,
    inference_bundle: Dict[str, torch.Tensor],
    temperature: float = 0.01,
) -> torch.Tensor:
    averaged = inference_bundle["averaged"]
    vision_feat = vision_feat / (vision_feat.norm(dim=1, keepdim=True) + 1e-6)
    logits = F.conv2d(vision_feat, averaged[:, :, None, None])
    probs = F.softmax(logits / temperature, dim=1)
    return probs.argmax(dim=1)


def compute_predictions_expert_fusion(
    vision_feat: torch.Tensor,
    inference_bundle: Dict[str, torch.Tensor],
    temperature: float = 0.01,
) -> torch.Tensor:
    batch, _, height, width = vision_feat.shape
    vision_feat = vision_feat / (vision_feat.norm(dim=1, keepdim=True) + 1e-6)
    expert_classifiers = inference_bundle["expert_classifiers"]
    averaged = inference_bundle["averaged"]
    classes, _, dim = expert_classifiers.shape

    expert_weights = expert_classifiers.reshape(classes * classes, dim)
    expert_logits = F.conv2d(vision_feat, expert_weights[:, :, None, None])
    expert_logits = expert_logits.reshape(batch, classes, classes, height, width).permute(1, 0, 2, 3, 4)
    expert_probs = F.softmax(expert_logits / temperature, dim=2)

    default_logits = F.conv2d(vision_feat, averaged[:, :, None, None])
    default_probs = F.softmax(default_logits / temperature, dim=1)

    expert_preds = expert_probs.argmax(dim=2)
    expert_indices = torch.arange(classes, device=vision_feat.device).view(classes, 1, 1, 1)
    expert_masks = expert_preds == expert_indices
    gather_idx = torch.arange(classes, device=vision_feat.device).view(classes, 1, 1, 1, 1)
    gather_idx = gather_idx.expand(classes, batch, 1, height, width)
    expert_self_conf = expert_probs.gather(2, gather_idx).squeeze(2)
    expert_self_conf = expert_self_conf.masked_fill(~expert_masks, float("-inf"))
    best_expert = expert_self_conf.argmax(dim=0)
    no_expert_match = ~expert_masks.any(dim=0)

    expert_probs_perm = expert_probs.permute(1, 3, 4, 0, 2)
    best_expert_idx = best_expert.unsqueeze(-1).unsqueeze(-1).expand(batch, height, width, 1, classes)
    selected = torch.gather(expert_probs_perm, 3, best_expert_idx).squeeze(3).permute(0, 3, 1, 2)
    fused_probs = torch.where(no_expert_match.unsqueeze(1), default_probs, selected)
    return fused_probs.argmax(dim=1)


def compute_miou(predictions: Sequence[torch.Tensor], gt_maps: Sequence[np.ndarray]) -> Tuple[dict, float]:
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.float64)
    for pred, gt in zip(predictions, gt_maps):
        pred = pred.cpu()
        if tuple(pred.shape) != tuple(gt.shape):
            pred = F.interpolate(
                pred.unsqueeze(0).unsqueeze(0).float(),
                size=gt.shape,
                mode="nearest",
            ).squeeze().long()
        gt_tensor = torch.as_tensor(gt, dtype=torch.long)
        valid = gt_tensor != IGNORE_INDEX
        if valid.any():
            pred_valid = pred[valid]
            gt_valid = gt_tensor[valid]
            hist = torch.bincount(
                gt_valid * NUM_CLASSES + pred_valid,
                minlength=NUM_CLASSES * NUM_CLASSES,
            ).reshape(NUM_CLASSES, NUM_CLASSES)
            confusion += hist.to(torch.float64)

    intersection = torch.diag(confusion)
    union = confusion.sum(dim=1) + confusion.sum(dim=0) - intersection
    iou = torch.full((NUM_CLASSES,), float("nan"), dtype=torch.float64)
    valid = union > 0
    iou[valid] = intersection[valid] / union[valid] * 100.0
    per_class = {name: float(iou[idx].item()) for idx, name in enumerate(CLASS_NAMES)}
    finite = iou[torch.isfinite(iou)]
    miou = float(finite.mean().item()) if finite.numel() else 0.0
    return per_class, miou


def save_scatter(rankings: dict, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    class_names = CLASS_NAMES[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for ax, class_name in zip(axes, class_names):
        ranking = rankings["classes"][class_name]["entropy_ranking"][:20]
        x = [row["rank"] for row in ranking]
        y = [row["entropy"] for row in ranking]
        labels = [row["template_id"] for row in ranking]
        ax.scatter(x, y, s=35)
        for rank, entropy, template_id in zip(x, y, labels):
            ax.annotate(str(template_id), (rank, entropy), fontsize=7)
        ax.set_title(class_name)
        ax.set_xlabel("rank")
        ax.set_ylabel("entropy")
    fig.suptitle("FLOSS smoke test: top-20 entropy-ranked templates")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_per_class_csv(path: Path, baseline: dict, floss: dict) -> None:
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["class", "baseline_iou", "floss_iou", "delta"])
        for class_name in CLASS_NAMES:
            base = baseline[class_name]
            fused = floss[class_name]
            writer.writerow([class_name, f"{base:.4f}", f"{fused:.4f}", f"{fused - base:.4f}"])


def print_dry_run_summary(args: argparse.Namespace) -> None:
    train_files, val_files, gt_files, text_file = select_remote_files(
        args.repo_id, args.num_images, args.num_val_images
    )
    all_remote_files = [text_file] + train_files + val_files + gt_files
    output_paths = [
        args.output_dir / "execution.log",
        args.output_dir / "download_manifest.json",
        args.output_dir / "rankings_smoke.json",
        args.output_dir / "metrics.json",
        args.output_dir / "per_class_iou.csv",
        args.output_dir / "scatter_mosaic.png",
        args.cache_dir,
    ]

    print("FLOSS smoke-test dry run")
    print(f"Repo: {args.repo_id}")
    print(f"Dataset: {DATASET}")
    print(f"Train images: {args.num_images}")
    print(f"Validation images: {args.num_val_images}")
    print(f"Files that would be downloaded: {len(all_remote_files)}")
    for remote_path in all_remote_files:
        print(f"  {remote_path}")
    print("Output paths that would be used:")
    for output_path in output_paths:
        print(f"  {output_path}")
    print("Dry run complete: no files were downloaded and no metrics were written.")


def main() -> None:
    args = parse_args()

    if args.dry_run:
        print_dry_run_summary(args)
        return

    require_runtime_dependencies()
    setup_logging(args.output_dir)
    device = get_device(args.device)
    logging.info("Dataset: %s", DATASET)
    logging.info("Train images: %s", args.num_images)
    logging.info("Validation images: %s", args.num_val_images)
    logging.info("Device: %s", device)
    logging.info("Ranking resolution: %s", tuple(args.ranking_resolution))

    train_files, val_files, gt_files, text_file = select_remote_files(
        args.repo_id, args.num_images, args.num_val_images
    )
    all_remote_files = [text_file] + train_files + val_files + gt_files
    manifest = {
        "repo_id": args.repo_id,
        "dataset": DATASET,
        "num_images": args.num_images,
        "num_val_images": args.num_val_images,
        "files": all_remote_files,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2))
    logging.info("Download manifest written to %s", args.output_dir / "download_manifest.json")

    local_paths = download_selected_files(args.repo_id, all_remote_files, args.cache_dir)
    text_features = torch.load(local_paths[text_file], weights_only=False)
    train_features = load_features([local_paths[path] for path in train_files])
    val_features = load_features([local_paths[path] for path in val_files])
    gt_maps = [np.array(Image.open(local_paths[path])) for path in gt_files]

    ranker = TemplateRanker(text_features, device, tuple(args.ranking_resolution))
    with torch.inference_mode():
        for idx, feature in enumerate(train_features, start=1):
            logging.info("Ranking train feature %s/%s", idx, len(train_features))
            ranker.update(feature)
    rankings = ranker.rankings()
    rankings_path = args.output_dir / "rankings_smoke.json"
    rankings_path.write_text(json.dumps(rankings, indent=2))
    logging.info("Smoke rankings written to %s", rankings_path)

    expert_ids = get_expert_template_ids(rankings)
    inference_bundle = build_inference_bundle(text_features, expert_ids, device)

    baseline_predictions = []
    floss_predictions = []
    with torch.inference_mode():
        for idx, feature in enumerate(val_features, start=1):
            logging.info("Evaluating validation feature %s/%s", idx, len(val_features))
            feature = feature.to(device)
            baseline_predictions.append(compute_predictions_default(feature, inference_bundle).squeeze(0))
            floss_predictions.append(compute_predictions_expert_fusion(feature, inference_bundle).squeeze(0))

    baseline_iou, baseline_miou = compute_miou(baseline_predictions, gt_maps)
    floss_iou, floss_miou = compute_miou(floss_predictions, gt_maps)
    metrics = {
        "dataset": DATASET,
        "num_images": args.num_images,
        "num_val_images": args.num_val_images,
        "device": str(device),
        "baseline_miou": baseline_miou,
        "floss_miou": floss_miou,
        "delta_miou": floss_miou - baseline_miou,
        "baseline_per_class_iou": baseline_iou,
        "floss_per_class_iou": floss_iou,
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    write_per_class_csv(args.output_dir / "per_class_iou.csv", baseline_iou, floss_iou)
    save_scatter(rankings, args.output_dir / "scatter_mosaic.png")

    logging.info("Baseline mIoU: %.4f", baseline_miou)
    logging.info("FLOSS mIoU: %.4f", floss_miou)
    logging.info("Delta mIoU: %.4f", floss_miou - baseline_miou)
    logging.info("Metrics written to %s", metrics_path)


if __name__ == "__main__":
    main()
