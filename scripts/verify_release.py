"""Check saved predictions, dataset dimensions, fold separation and file integrity."""
from __future__ import annotations

import csv
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ROOT / "outputs/studies"
RUNS = {
    "primary": ("depth_sweep_v2", "cv_20260104T134115Z"),
    "optical": ("depth_sweep_optical_domains_v1", "cv_optical_domains_grouped_v1"),
    "factorial": ("substrate_factorial_r3", "cv_substrate_factorial_r3_grouped_depth"),
    "raw_only": ("depth_sweep_v2", "cv_channel_raw_only_20260523A"),
    "raw_diff": ("depth_sweep_v2", "cv_channel_raw_diff_20260523A"),
    "raw_log": ("depth_sweep_v2", "cv_channel_raw_log_20260523A"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def metrics(rows):
    gt = np.array([float(r.get("gt_mm", r.get("depth_mm"))) for r in rows])
    pred = np.array([float(r["pred_mm"]) for r in rows])
    require(len(rows) > 1 and np.isfinite(gt).all() and np.isfinite(pred).all(), "Invalid predictions")
    return {"n": len(rows), "rmse_mm": float(np.sqrt(np.mean((gt - pred) ** 2))),
            "r2": float(1 - np.sum((gt - pred) ** 2) / np.sum((gt - gt.mean()) ** 2)),
            "r": float(np.corrcoef(gt, pred)[0, 1])}


def compare(actual, expected, context):
    if "n" in expected:
        require(actual["n"] == int(expected["n"]), f"Count mismatch: {context}")
    for key in ("rmse_mm", "r2", "r"):
        require(np.isclose(actual[key], expected[key], atol=2e-6, rtol=0), f"{context}: {key} differs")


def verify_run(name, study, run):
    data = STUDIES / study / "derived/nn_dataset/emission_top_pixel_1d"
    cv = STUDIES / study / "derived/nn/emission_top_pixel_1d" / run
    signals = np.load(data / "signals.npy", allow_pickle=False)
    depth = np.load(data / "targets_depth.npy", allow_pickle=False)
    thick = np.load(data / "targets_thickness.npy", allow_pickle=False)
    ids = json.loads((data / "case_ids.json").read_text())
    selected = set(np.flatnonzero(thick > 0).tolist())
    n = {"depth_sweep_v2": 71, "depth_sweep_optical_domains_v1": 355, "substrate_factorial_r3": 279}[study]
    require(len(selected) == n and signals.shape == (len(ids), 100), f"{name}: invalid shape")
    require(np.isfinite(signals).all() and len(set(ids)) == len(ids), f"{name}: invalid dataset")
    require(len(depth) == len(ids) == len(thick), f"{name}: label shape mismatch")
    summary = json.loads((cv / "summary.json").read_text())
    rows, tested = [], []
    for fold in sorted(cv.glob("fold_*")):
        splits = json.loads((fold / "splits.json").read_text())["splits"]
        parts = [set(splits[k]) for k in ("train", "val", "test")]
        require(all(len(parts[i] & parts[j]) == 0 for i in range(3) for j in range(i)), f"{name}: index leakage")
        require(set.union(*parts) == selected, f"{name}: incomplete fold partition")
        if summary["cv_policy"] == "grouped_depth_interleaved":
            groups = [set(depth[list(part)].tolist()) for part in parts]
            require(all(not (groups[i] & groups[j]) for i in range(3) for j in range(i)), f"{name}: depth leakage")
        prediction_file = fold / "stochastic_predictions.csv"
        if prediction_file.exists():
            predictions = load_csv(prediction_file)
        else:
            predictions = [r for r in load_csv(cv / "pooled_test_predictions.csv") if r["fold"] == fold.name]
        test = splits["test"]
        require(len(test) == len(predictions), f"{name}: prediction count")
        for i, row in zip(test, predictions):
            require(np.isclose(float(row["gt_mm"]), depth[i] * 0.05, atol=1e-5), f"{name}: prediction order")
            if "sample_id" in row:
                require(row["sample_id"] == f"idx{i}", f"{name}: dataset index differs")
        tested.extend(test)
        rows.extend(predictions)
    require(len(tested) == n and set(tested) == selected, f"{name}: duplicate or absent tests")
    compare(metrics(rows), summary["pooled_test"], name)
    if study == "substrate_factorial_r3":
        mult = np.load(data / "targets_conductivity_multiplier.npy", allow_pickle=False)
        combinations = {(round(float(depth[i]) * .05, 2), round(float(thick[i]) * .05, 2), round(float(mult[i]), 2)) for i in selected}
        expected = set(product(np.round(np.arange(.5, 3.51, .1), 2), (.15, .25, .5), (.15, .2, .4)))
        require(combinations == expected and len(combinations) == n, "Incomplete factorial grid")
    return metrics(rows)


def main():
    provenance = json.loads((ROOT / "PROVENANCE.json").read_text())
    for relative, expected in provenance["original_file_sha256"].items():
        p = ROOT / relative
        if p.is_file() and p.suffix in (".npy", ".pth", ".csv"):
            require(hashlib.sha256(p.read_bytes()).hexdigest() == expected, f"Original artifact changed: {relative}")
    manifest = ROOT / "MANIFEST.sha256"
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            expected, relative = line.split("  ", 1)
            p = ROOT / relative
            require(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == expected, f"Checksum mismatch: {relative}")
    results = {name: verify_run(name, *run) for name, run in RUNS.items()}
    primary = STUDIES / "depth_sweep_v2/derived/nn/emission_top_pixel_1d/cv_20260104T134115Z"
    for kind in ("baseline_features",):
        p = primary / kind
        result = metrics(load_csv(p / "pooled_predictions.csv"))
        require(result["n"] == 71, f"{kind}: incomplete predictions")
        compare(result, json.loads((p / "summary.json").read_text())["pooled_test"], kind)
        results[kind] = result
    frozen = primary.parent / "optical_domain_shift_v1"
    rows = load_csv(frozen / "pooled_predictions.csv")
    summary = json.loads((frozen / "summary.json").read_text())
    for record in summary["metrics"]:
        subset = [r for r in rows if r["domain"] == record["domain"] and (record["range"] == "full" or float(r["gt_mm"]) <= 3.0)]
        compare(metrics(subset), record, "frozen " + record["domain"])
    hashes = json.loads((frozen / "checkpoint_hashes.json").read_text())
    for domain in hashes.values():
        for fold, expected in domain.items():
            require(hashlib.sha256((primary / fold / "best_depthcnn.pth").read_bytes()).hexdigest() == expected, "Frozen checkpoint changed")
    print(json.dumps(results, indent=2))
    print("PASS: archived metrics, datasets, fold separation and checkpoint hashes")


if __name__ == "__main__":
    main()
