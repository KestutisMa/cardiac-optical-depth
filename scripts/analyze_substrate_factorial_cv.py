#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def _metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    err = pred - gt
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((gt - float(np.mean(gt))) ** 2))
    r = float(np.corrcoef(gt, pred)[0, 1]) if gt.size > 1 and np.std(pred) > 0 else float("nan")
    return {
        "n": int(gt.size),
        "rmse_mm": float(np.sqrt(np.mean(err**2))),
        "mean_error_mm": float(np.mean(err)),
        "r2": float(1.0 - ss_res / max(ss_tot, 1e-12)),
        "r": r,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _bootstrap_by_depth(rows: list[dict[str, object]], *, draws: int, seed: int) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    depths = sorted({float(row["depth_mm"]) for row in rows})
    by_depth = {depth: [row for row in rows if float(row["depth_mm"]) == depth] for depth in depths}
    samples: dict[str, list[float]] = {"rmse_mm": [], "r2": [], "r": []}
    for _ in range(draws):
        chosen = rng.choice(depths, size=len(depths), replace=True)
        selected = [row for depth in chosen for row in by_depth[float(depth)]]
        gt = np.asarray([float(row["depth_mm"]) for row in selected])
        pred = np.asarray([float(row["pred_mm"]) for row in selected])
        metric = _metrics(gt, pred)
        for key in samples:
            samples[key].append(float(metric[key]))
    return {
        key: {
            "lower_95": float(np.nanpercentile(values, 2.5)),
            "upper_95": float(np.nanpercentile(values, 97.5)),
        }
        for key, values in samples.items()
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Aggregate and plot substrate-factorial grouped-depth CV results.")
    p.add_argument("--cv-dir", required=True)
    p.add_argument("--bootstrap-draws", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=20260829)
    args = p.parse_args()

    cv_dir = Path(args.cv_dir)
    pooled_path = cv_dir / "pooled_test_predictions.csv"
    with pooled_path.open(newline="") as f:
        raw_rows = list(csv.DictReader(f))
    rows: list[dict[str, object]] = []
    for raw in raw_rows:
        rows.append(
            {
                "fold": int(raw["fold"]),
                "dataset_index": int(raw["dataset_index"]),
                "case_id": raw["case_id"],
                "depth_mm": float(raw["depth_mm"]),
                "thickness_mm": float(raw["thickness_mm"]),
                "conductivity_multiplier": float(raw["conductivity_multiplier"]),
                "pred_mm": float(raw["pred_mm"]),
            }
        )
    if len(rows) != 279:
        raise ValueError(f"Expected 279 pooled predictions, found {len(rows)}")
    expected_depths = np.round(np.arange(0.5, 3.5 + 0.05, 0.1), 2)
    expected = {
        (float(depth), thickness, multiplier)
        for depth in expected_depths
        for thickness in (0.15, 0.25, 0.5)
        for multiplier in (0.15, 0.2, 0.4)
    }
    observed = {
        (round(float(row["depth_mm"]), 2), round(float(row["thickness_mm"]), 2), round(float(row["conductivity_multiplier"]), 2))
        for row in rows
    }
    if expected != observed:
        raise ValueError(f"Unexpected factorial rows; missing={sorted(expected-observed)}, extra={sorted(observed-expected)}")
    for depth in expected_depths:
        folds = {int(row["fold"]) for row in rows if round(float(row["depth_mm"]), 2) == float(depth)}
        if len(folds) != 1:
            raise ValueError(f"Depth {depth:.2f} mm crosses test folds: {folds}")

    def summarize(selected: list[dict[str, object]]) -> dict[str, object]:
        gt = np.asarray([float(row["depth_mm"]) for row in selected])
        pred = np.asarray([float(row["pred_mm"]) for row in selected])
        result: dict[str, object] = _metrics(gt, pred)
        result["depth_group_bootstrap_95ci"] = _bootstrap_by_depth(
            selected, draws=int(args.bootstrap_draws), seed=int(args.bootstrap_seed)
        )
        return result

    full = summarize(rows)
    le_3 = summarize([row for row in rows if float(row["depth_mm"]) <= 3.0 + 1e-9])
    settings: list[dict[str, object]] = []
    for multiplier in (0.15, 0.2, 0.4):
        for thickness in (0.15, 0.25, 0.5):
            selected = [
                row
                for row in rows
                if abs(float(row["conductivity_multiplier"]) - multiplier) < 1e-8
                and abs(float(row["thickness_mm"]) - thickness) < 1e-8
            ]
            settings.append(
                {
                    "conductivity_multiplier": multiplier,
                    "conductivity_reduction_percent": 100.0 * (1.0 - multiplier),
                    "thickness_mm": thickness,
                    **_metrics(
                        np.asarray([float(row["depth_mm"]) for row in selected]),
                        np.asarray([float(row["pred_mm"]) for row in selected]),
                    ),
                }
            )

    spreads = []
    for depth in expected_depths:
        preds = np.asarray([float(row["pred_mm"]) for row in rows if round(float(row["depth_mm"]), 2) == float(depth)])
        spreads.append({"depth_mm": float(depth), "prediction_range_mm": float(np.ptp(preds)), "prediction_sd_mm": float(np.std(preds, ddof=1))})
    spread_values = np.asarray([float(row["prediction_range_mm"]) for row in spreads])
    checkpoints = {
        path.parent.name: _sha256(path)
        for path in sorted(cv_dir.glob("fold_*/best_depthcnn.pth"))
    }
    report = {
        "design": {"n": len(rows), "depths": 31, "thicknesses": 3, "conductivity_multipliers": 3},
        "full_range": full,
        "depth_le_3_mm": le_3,
        "per_setting": settings,
        "within_depth_prediction_range_mm": {
            "median": float(np.median(spread_values)),
            "p95": float(np.percentile(spread_values, 95)),
            "maximum": float(np.max(spread_values)),
            "by_depth": spreads,
        },
        "checkpoint_sha256": checkpoints,
    }
    output_dir = cv_dir / "factorial_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    with (output_dir / "per_setting_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(settings[0]))
        writer.writeheader()
        writer.writerows(settings)

    import matplotlib.pyplot as plt

    colors = {0.15: "#0072B2", 0.25: "#D55E00", 0.5: "#009E73"}
    markers = {0.15: "o", 0.25: "s", 0.5: "^"}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.45), sharex=True, sharey=True)
    for ax, multiplier in zip(axes, (0.15, 0.2, 0.4)):
        for thickness in (0.15, 0.25, 0.5):
            selected = [
                row
                for row in rows
                if abs(float(row["conductivity_multiplier"]) - multiplier) < 1e-8
                and abs(float(row["thickness_mm"]) - thickness) < 1e-8
            ]
            ax.scatter(
                [float(row["depth_mm"]) for row in selected],
                [float(row["pred_mm"]) for row in selected],
                s=20,
                alpha=0.8,
                color=colors[thickness],
                marker=markers[thickness],
                label=f"{thickness:.2f} mm",
            )
        ax.plot([0.4, 3.6], [0.4, 3.6], color="#333333", linestyle="--", linewidth=1)
        ax.set_title(rf"$g_{{\mathrm{{mult}}}}={multiplier:g}$")
        ax.set_xlim(0.4, 3.6)
        ax.set_ylim(0.4, 3.6)
        ax.set_xlabel("True start depth (mm)")
        ax.grid(color="#dddddd", linewidth=0.5)
    axes[0].set_ylabel("Predicted start depth (mm)")
    axes[-1].legend(title="Layer thickness", frameon=False, fontsize=8, title_fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "factorial_predictions.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "factorial_predictions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"full_range": full, "depth_le_3_mm": le_3, "within_depth_prediction_range_mm": report["within_depth_prediction_range_mm"]}, indent=2))
    print(f"[analysis] wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
