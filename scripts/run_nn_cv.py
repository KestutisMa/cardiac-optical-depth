#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run k-fold cross-validation for the depth-regression NN.")
    p.add_argument("--study-id", required=True)
    p.add_argument("--dataset-name", default=None)
    p.add_argument("--outputs-root", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--k", type=int, default=5)
    p.add_argument(
        "--cv-policy",
        choices=("depth_blocks", "interleaved", "grouped_depth_interleaved"),
        default="depth_blocks",
    )
    p.add_argument("--include-control", action="store_true")
    p.add_argument("--run-id", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-folds", type=int, default=None)
    return p.parse_args()


def _metric_arrays(gt: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    ss_res = float(np.sum((gt - pred) ** 2))
    ss_tot = float(np.sum((gt - float(np.mean(gt))) ** 2))
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))
    if gt.size > 1 and float(np.std(gt)) > 0 and float(np.std(pred)) > 0:
        r = float(np.corrcoef(gt, pred)[0, 1])
    else:
        r = float("nan")
    return {"n": int(gt.size), "rmse_mm": rmse, "r": r, "r2": r2}


def _load_prediction_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    args = _parse_args()
    root = _repo_root()
    sys.path.insert(0, str(root / "src"))
    from lmt_optical.layout import StudyLayout, default_outputs_root  # noqa: E402

    outputs_root = Path(args.outputs_root) if args.outputs_root else default_outputs_root()
    study = StudyLayout(outputs_root=outputs_root, study_id=str(args.study_id))
    study.ensure()
    dataset_name = args.dataset_name
    if dataset_name is None:
        latest = study.nn_dataset_dir / "LATEST"
        dataset_name = latest.read_text().strip() if latest.is_file() else "emission_top_pixel_1d"
    data_dir = study.nn_dataset_dir / str(dataset_name)
    if not (data_dir / "signals.npy").is_file():
        raise FileNotFoundError(data_dir / "signals.npy")

    config_path = Path(args.config) if args.config else root / "nn" / "exp_dev.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    k = int(args.k)
    if k < 3:
        raise ValueError("--k must be >= 3")

    depths = np.load(data_dir / "targets_depth.npy").astype(int)
    n = int(depths.size)
    thickness_path = data_dir / "targets_thickness.npy"
    thickness = np.load(thickness_path).astype(int) if thickness_path.is_file() else np.full(n, -1, dtype=int)
    gmult_path = data_dir / "targets_conductivity_multiplier.npy"
    gmult = np.load(gmult_path).astype(float) if gmult_path.is_file() else np.full(n, np.nan)
    case_ids_path = data_dir / "case_ids.json"
    case_ids = json.loads(case_ids_path.read_text()) if case_ids_path.is_file() else [str(i) for i in range(n)]
    if not (len(case_ids) == n == thickness.size == gmult.size):
        raise ValueError("Dataset labels and case IDs have inconsistent lengths")

    if not args.include_control and thickness_path.is_file():
        selected_idx = np.flatnonzero(thickness > 0)
    else:
        selected_idx = np.arange(n, dtype=int)
    if selected_idx.size < k:
        raise ValueError(f"Dataset too small for k={k}: selected N={selected_idx.size}")

    order = np.argsort(depths[selected_idx], kind="stable")
    sorted_idx = selected_idx[order]
    if args.cv_policy == "depth_blocks":
        blocks = [np.asarray(x, dtype=int) for x in np.array_split(sorted_idx, k)]
    elif args.cv_policy == "interleaved":
        blocks = [np.asarray(sorted_idx[i::k], dtype=int) for i in range(k)]
    else:
        unique_depths = np.unique(depths[selected_idx])
        depth_folds = [unique_depths[i::k] for i in range(k)]
        blocks = [selected_idx[np.isin(depths[selected_idx], fold_depths)] for fold_depths in depth_folds]
    if any(block.size == 0 for block in blocks):
        raise RuntimeError(f"Empty CV block: {[int(block.size) for block in blocks]}")
    if args.cv_policy == "grouped_depth_interleaved":
        memberships: dict[int, int] = {}
        for fold, block in enumerate(blocks):
            for depth in np.unique(depths[block]):
                if int(depth) in memberships:
                    raise RuntimeError(f"Depth {depth} appears in multiple test blocks")
                memberships[int(depth)] = fold

    meta = json.loads((data_dir / "meta.json").read_text()) if (data_dir / "meta.json").is_file() else {}
    voxel_mm = float((meta.get("voxel", {}) or {}).get("length_mm", 0.05))
    depth_min = int(np.min(depths[selected_idx]))
    depth_max = int(np.max(depths[selected_idx]))
    run_id = args.run_id or datetime.now(timezone.utc).strftime("cv_%Y%m%dT%H%M%SZ")
    cv_dir = study.nn_run_dir / str(dataset_name) / str(run_id)
    cv_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    (cv_dir / "cv_meta.json").write_text(
        json.dumps(
            {
                "study_id": str(args.study_id),
                "dataset_name": str(dataset_name),
                "data_dir": str(data_dir.resolve()),
                "config": str(config_path.resolve()),
                "k": k,
                "cv_policy": args.cv_policy,
                "created_at": created_at,
            },
            indent=2,
        )
    )

    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = os.pathsep.join([str(root / "nn"), env_base.get("PYTHONPATH", "")]).strip(os.pathsep)
    env_base["LMT_NN_DATA_DIR"] = str(data_dir)
    env_base["LMT_NN_DEPTH_MIN_IDX"] = str(depth_min)
    env_base["LMT_NN_DEPTH_MAX_IDX"] = str(depth_max)
    env_base["LMT_NN_VOXEL_MM"] = str(voxel_mm)

    max_folds = k if args.max_folds is None else int(args.max_folds)
    if not 1 <= max_folds <= k:
        raise ValueError(f"--max-folds must be between 1 and {k}")
    fold_summaries: list[dict[str, object]] = []
    pooled_rows: list[dict[str, object]] = []
    for fold in range(max_folds):
        test_idx = blocks[fold].astype(int).tolist()
        val_idx = blocks[(fold + 1) % k].astype(int).tolist()
        train_idx = np.concatenate([b for i, b in enumerate(blocks) if i not in (fold, (fold + 1) % k)]).astype(int).tolist()
        if set(test_idx) & set(val_idx) or set(test_idx) & set(train_idx) or set(val_idx) & set(train_idx):
            raise RuntimeError(f"Overlapping split indices in fold {fold}")
        if args.cv_policy == "grouped_depth_interleaved":
            split_depths = [set(depths[idx].tolist()) for idx in (train_idx, val_idx, test_idx)]
            if split_depths[0] & split_depths[1] or split_depths[0] & split_depths[2] or split_depths[1] & split_depths[2]:
                raise RuntimeError(f"Depth leakage in fold {fold}")

        fold_dir = cv_dir / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        splits_path = fold_dir / "splits.json"
        splits_path.write_text(
            json.dumps(
                {
                    "cv_policy": args.cv_policy,
                    "k": k,
                    "fold": fold,
                    "splits": {"train": train_idx, "val": val_idx, "test": test_idx},
                },
                indent=2,
            )
        )
        env = env_base.copy()
        env["LMT_NN_OUT_DIR"] = str(fold_dir)
        env["LMT_NN_CHECKPOINT"] = str(fold_dir / "best_depthcnn.pth")
        env["LMT_NN_SPLITS_PATH"] = str(splits_path)
        complete = all(
            (fold_dir / name).is_file()
            for name in ("best_depthcnn.pth", "metrics_test.json", "stochastic_predictions.csv")
        )
        if not (args.resume and complete):
            subprocess.run(
                [sys.executable, str(root / "nn" / "resnet_optical.py"), "--config", str(config_path), "--train"],
                check=True,
                cwd=root,
                env=env,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(root / "nn" / "resnet_optical.py"),
                    "--config",
                    str(config_path),
                    "--eval",
                    "--eval-split",
                    "all",
                ],
                check=True,
                cwd=root,
                env=env,
            )
        else:
            print(f"[cv] fold {fold:02d}: reusing completed outputs", flush=True)

        prediction_path = fold_dir / "stochastic_predictions.csv"
        prediction_rows = _load_prediction_rows(prediction_path)
        if len(prediction_rows) != len(test_idx):
            raise RuntimeError(
                f"Fold {fold} prediction count {len(prediction_rows)} != test count {len(test_idx)}"
            )
        for idx, rec in zip(test_idx, prediction_rows):
            gt_mm = float(rec["gt_mm"])
            expected_mm = float(depths[idx]) * voxel_mm
            if abs(gt_mm - expected_mm) > 1e-5:
                raise RuntimeError(f"Fold {fold} prediction order mismatch at dataset index {idx}")
            pooled_rows.append(
                {
                    "fold": fold,
                    "dataset_index": idx,
                    "case_id": case_ids[idx],
                    "depth_voxels": int(depths[idx]),
                    "depth_mm": expected_mm,
                    "thickness_voxels": int(thickness[idx]),
                    "thickness_mm": float(thickness[idx]) * voxel_mm,
                    "conductivity_multiplier": float(gmult[idx]),
                    "pred_mm": float(rec["pred_mm"]),
                    "std_mm": float(rec.get("std_mm") or "nan"),
                }
            )
        metrics = json.loads((fold_dir / "metrics_test.json").read_text())
        fold_summaries.append({"fold": fold, "fold_dir": str(fold_dir), "metrics_test": metrics})

    pooled_csv = cv_dir / "pooled_test_predictions.csv"
    with pooled_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pooled_rows[0]))
        writer.writeheader()
        writer.writerows(pooled_rows)
    gt = np.asarray([float(row["depth_mm"]) for row in pooled_rows])
    pred = np.asarray([float(row["pred_mm"]) for row in pooled_rows])
    summary = {
        "k": k,
        "completed_folds": max_folds,
        "cv_policy": args.cv_policy,
        "n_dataset_cases": n,
        "n_selected_cases": int(selected_idx.size),
        "label_range": {"min_idx": depth_min, "max_idx": depth_max, "voxel_mm": voxel_mm},
        "pooled_test": _metric_arrays(gt, pred),
        "pooled_predictions_csv": str(pooled_csv),
        "folds": fold_summaries,
    }
    (cv_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[cv] wrote {cv_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
