#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run a simple baseline using a small set of hand-crafted features from the 1D emission signal, "
            "then fit a linear model (ridge/OLS) on the same CV folds as an existing NN CV run."
        )
    )
    p.add_argument(
        "--cv-run-dir",
        type=str,
        required=True,
        help="Path to a CV run directory (contains cv_meta.json and fold_*/splits.json).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: <cv-run-dir>/baseline_features).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        action="append",
        default=[],
        help="Ridge alpha value(s) to try. Repeatable. If omitted, uses a small default grid.",
    )
    p.add_argument("--digits", type=int, default=3, help="Decimal digits for printed metrics.")
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _fold_dirs(cv_dir: Path) -> list[Path]:
    return sorted([p for p in cv_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")])


def _metrics(gt: Any, pr: Any) -> dict[str, float]:
    import math
    import numpy as np

    g = np.asarray(gt, dtype=np.float64).reshape(-1)
    p = np.asarray(pr, dtype=np.float64).reshape(-1)
    if g.size == 0:
        return {"r": float("nan"), "r2": float("nan"), "rmse_mm": float("nan")}
    rmse = float(math.sqrt(float(np.mean((p - g) ** 2))))
    if g.size < 2:
        r = float("nan")
    else:
        g0 = g - float(np.mean(g))
        p0 = p - float(np.mean(p))
        denom = float(np.sqrt(np.sum(g0**2) * np.sum(p0**2)))
        r = float(np.sum(g0 * p0) / denom) if denom > 0 else float("nan")
    ss_res = float(np.sum((g - p) ** 2))
    ss_tot = float(np.sum((g - float(np.mean(g))) ** 2))
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))
    return {"r": r, "r2": r2, "rmse_mm": rmse}


@dataclass(frozen=True)
class _FeatureCfg:
    baseline_n: int = 10
    tail_n: int = 10


def _features_for_signal(sig: Any, dt_ms: float, cfg: _FeatureCfg) -> dict[str, float]:
    """
    Small, interpretable features from a single aligned+windowed signal.

    Notes:
    - Signals are already aligned by max dF/dt into a fixed window, so absolute activation time is not informative.
    - These features capture amplitude/shape differences around the aligned event.
    """
    import numpy as np

    x = np.asarray(sig, dtype=np.float64).reshape(-1)
    if x.size < 5:
        raise ValueError("Signal too short")
    b = float(np.mean(x[: cfg.baseline_n]))
    tail = float(np.mean(x[-cfg.tail_n :]))
    peak = float(np.max(x))
    trough = float(np.min(x))
    amp = float(peak - trough)
    d = np.diff(x, prepend=x[0])
    dmax = float(np.max(d))
    dmin = float(np.min(d))
    auc = float(np.sum(x) * float(dt_ms))
    return {
        "baseline": b,
        "tail": tail,
        "peak": peak,
        "amp": amp,
        "dmax": dmax,
        "dmin": dmin,
        "auc": auc,
    }


def _standardize_train_apply(
    x_train: Any, x_other: Any | None
) -> tuple[Any, Any | None, list[float], list[float]]:
    import numpy as np

    xt = np.asarray(x_train, dtype=np.float64)
    mu = xt.mean(axis=0)
    sd = xt.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    xtz = (xt - mu) / sd
    if x_other is None:
        return xtz, None, mu.tolist(), sd.tolist()
    xo = np.asarray(x_other, dtype=np.float64)
    xoz = (xo - mu) / sd
    return xtz, xoz, mu.tolist(), sd.tolist()


def _fit_ridge_closed_form(x: Any, y: Any, alpha: float) -> tuple[Any, float]:
    """
    Solve ridge regression with intercept:
      minimize ||Xw + b - y||^2 + alpha ||w||^2
    Returns (w, b).
    """
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.ndim != 2:
        raise ValueError(f"Expected X as (N,F), got {x.shape}")
    if y.shape[0] != x.shape[0]:
        raise ValueError("X/y length mismatch")

    alpha = float(alpha)
    if alpha < 0:
        raise ValueError("alpha must be >= 0")

    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    xc = x - x_mean
    yc = y - y_mean

    if alpha == 0.0:
        w = np.linalg.lstsq(xc, yc, rcond=None)[0]
    else:
        xtx = xc.T @ xc
        xty = xc.T @ yc
        reg = alpha * np.eye(x.shape[1], dtype=np.float64)
        try:
            w = np.linalg.solve(xtx + reg, xty)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(xtx + reg, xty, rcond=None)[0]
    b = y_mean - float(x_mean @ w)
    return w, b


def _predict(x: Any, w: Any, b: float) -> Any:
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    return x @ np.asarray(w, dtype=np.float64) + float(b)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(list(rows))


def main() -> int:
    args = _parse_args()
    cv_dir = Path(args.cv_run_dir).resolve()
    if not cv_dir.is_dir():
        raise FileNotFoundError(cv_dir)

    meta = _load_json(cv_dir / "cv_meta.json")
    data_dir = Path(str(meta.get("data_dir", "")))
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (cv_dir / "baseline_features")
    out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import matplotlib.pyplot as plt

    ds_meta = _load_json(data_dir / "meta.json") if (data_dir / "meta.json").is_file() else {}
    voxel_mm = float(((ds_meta.get("voxel") or {}).get("length_mm", 0.05)))
    dt_ms = float((((ds_meta.get("time_axis") or {}).get("dt_ms")) or 1.0))

    signals = np.load(data_dir / "signals.npy").astype(np.float64, copy=False)  # (N,L)
    depth_idx = np.load(data_dir / "targets_depth.npy").astype(np.int64, copy=False)  # (N,)
    gt_mm_all = depth_idx.astype(np.float64) * voxel_mm

    cfg = _FeatureCfg()
    feat_dicts = [_features_for_signal(sig, dt_ms=dt_ms, cfg=cfg) for sig in signals]
    feat_names = list(feat_dicts[0].keys())
    x_all = np.asarray([[fd[n] for n in feat_names] for fd in feat_dicts], dtype=np.float64)

    alphas = [float(a) for a in args.alpha] if args.alpha else [0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]

    pooled_rows: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []

    for fold_dir in _fold_dirs(cv_dir):
        splits_path = fold_dir / "splits.json"
        if not splits_path.is_file():
            continue
        splits = _load_json(splits_path).get("splits", {})
        train_idx = [int(i) for i in splits.get("train", [])]
        val_idx = [int(i) for i in splits.get("val", [])]
        test_idx = [int(i) for i in splits.get("test", [])]
        if not (train_idx and test_idx):
            raise ValueError(f"Empty splits in {splits_path}")

        x_train = x_all[train_idx]
        y_train = gt_mm_all[train_idx]
        x_val = x_all[val_idx] if val_idx else None
        y_val = gt_mm_all[val_idx] if val_idx else None

        # Z-score using TRAIN only.
        x_train_z, x_val_z, mu, sd = _standardize_train_apply(x_train, x_val)

        best_alpha = float(alphas[0])
        best_rmse = float("inf")
        for a in alphas:
            w, b = _fit_ridge_closed_form(x_train_z, y_train, alpha=float(a))
            if x_val_z is None or y_val is None or len(val_idx) == 0:
                best_alpha = float(a)
                best_rmse = float("nan")
                break
            pr_val = _predict(x_val_z, w, b)
            m = _metrics(y_val, pr_val)
            if float(m["rmse_mm"]) < float(best_rmse):
                best_rmse = float(m["rmse_mm"])
                best_alpha = float(a)

        w, b = _fit_ridge_closed_form(x_train_z, y_train, alpha=float(best_alpha))
        x_test = x_all[test_idx]
        x_test_z = (x_test - np.asarray(mu)) / np.asarray(sd)
        y_test = gt_mm_all[test_idx]
        pr_test = _predict(x_test_z, w, b)
        m_test = _metrics(y_test, pr_test)

        fold_summaries.append(
            {
                "fold": fold_dir.name,
                "alpha": best_alpha,
                "metrics_test": m_test,
                "feature_names": feat_names,
                "standardize_train": {"mean": mu, "std": sd},
                "weights": [float(x) for x in np.asarray(w).reshape(-1)],
                "bias": float(b),
            }
        )

        for idx, gt_mm, pr_mm in zip(test_idx, y_test.tolist(), pr_test.tolist()):
            pooled_rows.append({"sample_idx": int(idx), "gt_mm": float(gt_mm), "pred_mm": float(pr_mm), "fold": fold_dir.name})

    pooled_rows.sort(key=lambda r: r["sample_idx"])
    gt_pooled = [r["gt_mm"] for r in pooled_rows]
    pr_pooled = [r["pred_mm"] for r in pooled_rows]
    pooled_metrics = _metrics(gt_pooled, pr_pooled)

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "cv_run_dir": str(cv_dir),
                "data_dir": str(data_dir),
                "model": "linear_features_ridge",
                "features": {"names": feat_names, "config": {"baseline_n": cfg.baseline_n, "tail_n": cfg.tail_n, "dt_ms": dt_ms}},
                "alphas_tried": alphas,
                "voxel_mm": float(voxel_mm),
                "folds": fold_summaries,
                "pooled_test": pooled_metrics,
            },
            indent=2,
        )
        + "\n"
    )

    _write_csv(out_dir / "pooled_predictions.csv", pooled_rows, ["sample_idx", "gt_mm", "pred_mm", "fold"])
    _write_csv(out_dir / "features.csv", [dict(**{"sample_idx": i}, **fd) for i, fd in enumerate(feat_dicts)], ["sample_idx", *feat_names])

    # Figure: pooled scatter
    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    ax.scatter(gt_pooled, pr_pooled, s=18, alpha=0.8)
    mn = float(min(min(gt_pooled), min(pr_pooled)))
    mx = float(max(max(gt_pooled), max(pr_pooled)))
    ax.plot([mn, mx], [mn, mx], "--", lw=1.0, alpha=0.7)
    ax.set_xlabel("Ground truth depth (mm)")
    ax.set_ylabel("Predicted depth (mm)")
    ax.set_title("Baseline (hand-crafted features): pooled out-of-fold test")
    txt = rf"$r={pooled_metrics['r']:.3f}\quad R^2={pooled_metrics['r2']:.3f}\quad \mathrm{{RMSE}}={pooled_metrics['rmse_mm']:.3f}\,\mathrm{{mm}}$"
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top", ha="left", bbox=dict(boxstyle="round", pad=0.3, alpha=0.2))
    fig.tight_layout()
    fig.savefig(out_dir / "pooled_scatter.png", dpi=220)
    fig.savefig(out_dir / "pooled_scatter.pdf")
    plt.close(fig)

    digs = int(args.digits)
    print(
        f"pooled: r={pooled_metrics['r']:.{digs}f} R2={pooled_metrics['r2']:.{digs}f} RMSE={pooled_metrics['rmse_mm']:.{digs}f} mm"
    )
    print(f"features: {', '.join(feat_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

