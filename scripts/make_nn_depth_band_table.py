#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate depth-stratified performance metrics from pooled out-of-fold predictions "
            "(fold_*/stochastic_predictions.csv) under a CV run directory."
        )
    )
    p.add_argument(
        "--cv-run-dir",
        type=str,
        required=True,
        help="Path to a CV run directory (contains fold_*/ and summary.json).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: <cv-run-dir>/paper_tables).",
    )
    p.add_argument(
        "--band",
        type=str,
        action="append",
        default=[],
        help="Depth band as 'Name,lo_mm,hi_mm'. Repeatable. Default bands match the manuscript table.",
    )
    p.add_argument("--digits", type=int, default=3, help="Decimal digits for table values.")
    return p.parse_args()


def _fold_dirs(cv_dir: Path) -> list[Path]:
    return sorted([p for p in cv_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")])


def _load_pooled_test_predictions(cv_dir: Path) -> tuple[list[float], list[float]]:
    gt_all: list[float] = []
    pr_all: list[float] = []
    for fold_dir in _fold_dirs(cv_dir):
        path = fold_dir / "stochastic_predictions.csv"
        if not path.is_file():
            continue
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or ("gt_mm" not in reader.fieldnames) or ("pred_mm" not in reader.fieldnames):
                raise ValueError(f"Unexpected CSV header in {path}: {reader.fieldnames}")
            for row in reader:
                gt_all.append(float(row["gt_mm"]))
                pr_all.append(float(row["pred_mm"]))
    if not gt_all:
        raise FileNotFoundError("No pooled predictions found (expected fold_*/stochastic_predictions.csv)")
    return gt_all, pr_all


def _metrics(gt: list[float], pr: list[float]) -> dict[str, Any]:
    # Defer imports so the script can be inspected outside the container.
    import math
    import numpy as np

    g = np.asarray(gt, dtype=float)
    p = np.asarray(pr, dtype=float)
    n = int(g.size)
    if n == 0:
        return {"n": 0, "r": float("nan"), "r2": float("nan"), "rmse_mm": float("nan")}

    rmse = float(math.sqrt(float(np.mean((p - g) ** 2))))

    if n < 2:
        r = float("nan")
    else:
        # Pearson correlation (no SciPy dependency).
        g0 = g - float(np.mean(g))
        p0 = p - float(np.mean(p))
        denom = float(np.sqrt(np.sum(g0**2) * np.sum(p0**2)))
        r = float(np.sum(g0 * p0) / denom) if denom > 0 else float("nan")

    ss_res = float(np.sum((g - p) ** 2))
    ss_tot = float(np.sum((g - float(np.mean(g))) ** 2))
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))
    return {"n": n, "r": r, "r2": r2, "rmse_mm": rmse}


def _default_bands() -> list[tuple[str, float, float]]:
    return [
        ("Sub-epicardial", 0.0, 1.5),
        ("Mid-wall", 1.5, 3.0),
        ("Deep", 3.0, 4.0),
    ]


def _parse_bands(raw: list[str]) -> list[tuple[str, float, float]]:
    if not raw:
        return _default_bands()
    out: list[tuple[str, float, float]] = []
    for item in raw:
        parts = [p.strip() for p in str(item).split(",")]
        if len(parts) != 3:
            raise ValueError(f"--band must be 'Name,lo_mm,hi_mm', got: {item!r}")
        out.append((parts[0], float(parts[1]), float(parts[2])))
    return out


def main() -> int:
    args = _parse_args()
    cv_dir = Path(args.cv_run_dir).resolve()
    if not cv_dir.is_dir():
        raise FileNotFoundError(cv_dir)
    summary_path = cv_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (cv_dir / "paper_tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_all, pr_all = _load_pooled_test_predictions(cv_dir)

    # Defer imports so the script can be inspected outside the container.
    import numpy as np

    gt = np.asarray(gt_all, dtype=float)
    pr = np.asarray(pr_all, dtype=float)

    bands = _parse_bands(list(args.band))
    rows: list[dict[str, Any]] = []
    for i, (name, lo, hi) in enumerate(bands):
        if i == len(bands) - 1:
            mask = (gt >= float(lo)) & (gt <= float(hi))
        else:
            mask = (gt >= float(lo)) & (gt < float(hi))
        m = _metrics(gt[mask].tolist(), pr[mask].tolist())
        rows.append({"name": name, "lo_mm": float(lo), "hi_mm": float(hi), **m})

    digits = int(args.digits)

    def fmt(x: float) -> str:
        if x != x:  # NaN
            return "--"
        return f"{float(x):.{digits}f}"

    latex_lines = []
    for r in rows:
        band_label = f"{r['name']} ({r['lo_mm']:.1f}--{r['hi_mm']:.1f})"
        latex_lines.append(f"{band_label} & {fmt(r['r'])} & {fmt(r['r2'])} & {fmt(r['rmse_mm'])} \\\\")

    (out_dir / "depth_bands_metrics.json").write_text(json.dumps({"cv_run_dir": str(cv_dir), "rows": rows}, indent=2))
    (out_dir / "depth_bands_rows.tex").write_text("\n".join(latex_lines) + "\n")

    print("\n".join(latex_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

