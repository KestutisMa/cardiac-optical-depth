"""Evaluate released checkpoints with their recorded configuration, without retraining."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import fields
from pathlib import Path

import torch

from verify_release import ROOT, RUNS, STUDIES, load_csv, metrics

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "generated/.matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(ROOT / "nn"))
from resnet_optical import TrainConfig, evaluate  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", choices=RUNS, default="primary")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="generated/evaluation")
    args = p.parse_args()
    torch.set_num_threads(2)
    study, run = RUNS[args.run]
    cv = STUDIES / study / "derived/nn/emission_top_pixel_1d" / run
    destination = Path(args.output).resolve() / args.run
    if destination.is_relative_to(ROOT / "outputs"):
        p.error("Use an output outside the archived outputs directory")
    pooled = []
    allowed = {f.name for f in fields(TrainConfig)}
    archived = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in STUDIES.rglob("*") if p.is_file()}
    for fold in sorted(cv.glob("fold_*")):
        checkpoint = fold / "best_depthcnn.pth"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        cfg = {k: v for k, v in saved["config"].items() if k in allowed}
        out = destination / fold.name
        out.mkdir(parents=True, exist_ok=True)
        copied_checkpoint = out / checkpoint.name
        shutil.copyfile(checkpoint, copied_checkpoint)
        cfg.update(data_dir=str(STUDIES / study / "derived/nn_dataset/emission_top_pixel_1d"),
                   checkpoint=str(copied_checkpoint), device=args.device,
                   save_plot=str(out / "predictions.png"),
                   save_stochastic=str(out / "stochastic_predictions.csv"),
                   save_stochastic_raw=None)
        evaluate(TrainConfig(**cfg), eval_split="all")
        pooled.extend(load_csv(out / "stochastic_predictions.csv"))
    result = metrics(pooled)
    if any(hashlib.sha256(p.read_bytes()).hexdigest() != value for p, value in archived.items()):
        raise RuntimeError("An archived artifact changed during evaluation")
    (destination / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print("Historical predictions remain unchanged; stochastic/device differences may affect reruns.")


if __name__ == "__main__":
    main()
