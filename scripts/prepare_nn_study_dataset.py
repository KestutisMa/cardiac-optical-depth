#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare a study-level 1D dataset for NN training from /outputs/studies/<study_id>/cases/*"
    )
    p.add_argument("--study-id", type=str, required=True)
    p.add_argument(
        "--outputs-root",
        type=str,
        default=None,
        help="Outputs root (default: /outputs if present, else ./outputs)",
    )
    p.add_argument(
        "--dataset-name",
        type=str,
        default="emission_top_pixel_1d",
        help="Name under /outputs/studies/<study_id>/derived/nn_dataset/",
    )
    p.add_argument(
        "--roi-box",
        type=str,
        default=None,
        help="Optional ROI box as 'x0:x1,y0:y1' in detector pixel coordinates (exclusive end).",
    )
    p.add_argument(
        "--roi-pixel",
        type=str,
        default=None,
        help="Optional single pixel as 'x,y' in detector pixel coordinates (overrides --roi-box/--roi-margin).",
    )
    p.add_argument(
        "--roi-margin",
        type=int,
        default=None,
        help="Mean-ROI only: exclude this many pixels from each image border before taking the mean (ignored if --roi-box is set).",
    )
    # Default window matches the original OpenCM plotting/filtering scripts:
    # large pre-window (baseline) and short post-window after the upstroke.
    # Total length is (pre + post + 1) samples.
    p.add_argument("--align-pre", type=int, default=90, help="Samples before dF/dt peak (t<0)")
    p.add_argument("--align-post", type=int, default=9, help="Samples after dF/dt peak (t>0)")
    p.add_argument(
        "--ceemdan-trials",
        type=int,
        default=500,
        help="CEEMDAN trials (ensemble size).",
    )
    p.add_argument(
        "--ceemdan-noise-strength",
        type=float,
        default=0.1,
        help="CEEMDAN noise strength.",
    )
    p.add_argument(
        "--ceemdan-max-imf",
        type=int,
        default=None,
        help="Max IMFs to retain (excluding the first/noise IMF). Default: keep all except first.",
    )
    p.add_argument(
        "--ceemdan-seed",
        type=int,
        default=None,
        help="Optional base seed; a stable case-specific CEEMDAN seed is derived from it.",
    )
    p.add_argument("--max-cases", type=int, default=None, help="Limit number of cases (debug)")
    p.add_argument("--workers", type=int, default=1, help="Parallel workers for per-case processing (CEEMDAN is CPU-heavy).")
    p.add_argument("--progress-every", type=int, default=5, help="Print progress every N completed cases.")
    return p.parse_args()


@dataclass(frozen=True)
class _CaseRow:
    case_id: str
    signal: np.ndarray  # (L,) float32
    signal_raw: np.ndarray  # (L,) float32 (aligned+windowed, pre-CEEMDAN)
    depth: int
    thickness: int
    conductivity_multiplier: float
    time_ms: np.ndarray  # (L,) float32
    time_indices: np.ndarray  # (L,) int32
    dt_ms: float
    voxel_length_um: float
    resampled: bool
    peak_index: int


def _parse_roi_box(s: str) -> tuple[slice, slice]:
    # Format: "x0:x1,y0:y1"
    try:
        a, b = s.split(",", 1)
        x0, x1 = a.split(":")
        y0, y1 = b.split(":")
        xs = slice(int(x0), int(x1))
        ys = slice(int(y0), int(y1))
        return xs, ys
    except Exception as e:
        raise ValueError(f"Invalid --roi-box '{s}', expected 'x0:x1,y0:y1'") from e


def _parse_roi_pixel(s: str) -> tuple[int, int]:
    try:
        a, b = s.split(",", 1)
        return int(a), int(b)
    except Exception as e:
        raise ValueError(f"Invalid --roi-pixel '{s}', expected 'x,y'") from e


def _load_time_indices_from_emission_h5(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    with h5py.File(path, "r") as f:
        vox = f["voxels"]
        attrs = dict(vox.attrs.items())
        if "time_indices" not in f:
            raise KeyError(f"Missing time_indices dataset in {path}")
        time_indices = np.asarray(f["time_indices"][()], dtype=np.int32)
        return attrs, time_indices


def _load_vm_meta_from_h5(path: Path) -> tuple[float, float]:
    with h5py.File(path, "r") as f:
        vox = f["voxels"]
        dt_ms = vox.attrs.get("ep_spacedt_ms", None)
        if dt_ms is None:
            t_end_ms = vox.attrs.get("t_end_ms", None)
            if t_end_ms is None or int(vox.shape[0]) < 2:
                raise KeyError(f"Missing saved-frame interval metadata in {path}")
            dt_ms = float(t_end_ms) / float(int(vox.shape[0]) - 1)
        voxel_um = vox.attrs.get("voxel_length_um", None)
        if voxel_um is None:
            voxel_um = vox.attrs.get("voxel_um", None)
        if voxel_um is None:
            raise KeyError(f"Missing voxels.attrs['voxel_length_um'] (or 'voxel_um') in {path}")
        return float(dt_ms), float(voxel_um)


def _extract_signal(detector_top: np.ndarray, roi_x: slice | None, roi_y: slice | None) -> np.ndarray:
    if detector_top.ndim != 3:
        raise ValueError(f"Expected emission_detector_top as (T, X, Y), got shape={detector_top.shape}")
    if roi_x is None or roi_y is None:
        cropped = detector_top
        roi_desc = "full"
    else:
        cropped = detector_top[:, roi_x, roi_y]
        roi_desc = f"x={roi_x.start}:{roi_x.stop} y={roi_y.start}:{roi_y.stop}"
    if cropped.size == 0:
        raise ValueError(f"Empty ROI after cropping ({roi_desc}); detector shape={detector_top.shape}")
    return cropped.mean(axis=(1, 2)).astype(np.float32, copy=False)


def _extract_signal_pixel(detector_top: np.ndarray, *, x: int, y: int) -> np.ndarray:
    if detector_top.ndim != 3:
        raise ValueError(f"Expected emission_detector_top as (T, X, Y), got shape={detector_top.shape}")
    _, nx, ny = detector_top.shape
    if not (0 <= int(x) < nx and 0 <= int(y) < ny):
        raise ValueError(f"roi-pixel out of bounds: (x={x}, y={y}) for detector shape={detector_top.shape}")
    return detector_top[:, int(x), int(y)].astype(np.float32, copy=False)


def _ceemdan_denoise(
    signal: np.ndarray,
    *,
    trials: int,
    noise_strength: float,
    max_imf: int | None,
    seed: int | None,
) -> np.ndarray:
    """
    CEEMDAN denoising: drop the first (highest-frequency / noise) IMF and reconstruct.
    Matches the behavior in the original OpenCM helper scripts.
    """
    try:
        from PyEMD import CEEMDAN  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "CEEMDAN requires the `PyEMD` module (pip package is typically `EMD-signal`). "
            "If you're running on the host, run this script inside the container. "
            "If you're running inside the container, add `EMD-signal` to requirements.txt and rebuild."
        ) from e

    ceemd = CEEMDAN(trials=int(trials), noise_strength=float(noise_strength))
    if seed is not None:
        ceemd.noise_seed(int(seed))
    imfs = ceemd(signal.astype(np.float64))
    if imfs.ndim != 2 or imfs.shape[0] < 2:
        return signal.astype(np.float32, copy=False)
    to_keep = imfs[1:] if max_imf is None else imfs[1 : int(max_imf)]
    out = np.sum(to_keep, axis=0)
    return out.astype(np.float32, copy=False)


def _align_and_window(
    signal: np.ndarray,
    *,
    pre: int,
    post: int,
) -> tuple[np.ndarray, int]:
    """
    Align by max dF/dt (via diff) and extract a fixed window [peak-pre, peak+post].
    Pads if the window hits boundaries (see pad_mode).
    """
    pre = int(max(0, pre))
    post = int(max(0, post))
    if signal.size < 3:
        peak = int(np.argmax(signal))
    else:
        dfd = np.diff(signal.astype(np.float64), n=1)
        peak = int(np.argmax(dfd))  # 0..L-2; treat as peak at index in original signal
    start = peak - pre
    end = peak + post + 1
    left_pad = max(0, -start)
    right_pad = max(0, end - signal.size)
    s = max(0, start)
    e = min(signal.size, end)
    seg = signal[s:e].astype(np.float32, copy=False)
    if left_pad or right_pad:
        seg = np.pad(seg, (left_pad, right_pad), mode="edge")
    return seg, peak


@dataclass(frozen=True)
class _CaseTask:
    case_id: str
    raw_opencm_dir: Path


def _process_case(
    task: _CaseTask,
    *,
    roi_box: str | None,
    roi_pixel: tuple[int, int] | None,
    roi_margin: int | None,
    align_pre: int,
    align_post: int,
    ceemdan_trials: int,
    ceemdan_noise_strength: float,
    ceemdan_max_imf: int | None,
    ceemdan_seed: int | None,
    canonical_time_ms: np.ndarray,
) -> _CaseRow:
    case_id = task.case_id
    raw_opencm_dir = task.raw_opencm_dir

    det_path = raw_opencm_dir / "emission_detector_top.npy"
    emission_h5 = raw_opencm_dir / "emission_voxels.h5"
    vm_h5 = raw_opencm_dir / "vm_voxels.h5"

    det = np.load(det_path)  # (Tsel, X, Y)
    if roi_pixel is not None:
        px, py = int(roi_pixel[0]), int(roi_pixel[1])
        signal = _extract_signal_pixel(det, x=px, y=py)
    else:
        if roi_box is not None:
            roi_x, roi_y = _parse_roi_box(roi_box)
        else:
            margin = int(max(0, roi_margin or 0))
            roi_x = slice(margin, None if margin == 0 else -margin)
            roi_y = slice(margin, None if margin == 0 else -margin)
        signal = _extract_signal(det, roi_x=roi_x, roi_y=roi_y)

    attrs, time_indices = _load_time_indices_from_emission_h5(emission_h5)
    dt_ms, voxel_length_um = _load_vm_meta_from_h5(vm_h5)
    time_ms = (time_indices.astype(np.float32) * np.float32(dt_ms)).astype(np.float32, copy=False)

    depth = int(attrs.get("ischemia_depth", 0))
    thickness = int(attrs.get("ischemia_thickness", 0))
    conductivity_multiplier = float(attrs.get("conductivity_multiplier", float("nan")))

    resampled = False
    if time_ms.shape != canonical_time_ms.shape or not np.allclose(time_ms, canonical_time_ms, rtol=0.0, atol=1e-6):
        signal = np.interp(canonical_time_ms.astype(np.float64), time_ms.astype(np.float64), signal.astype(np.float64)).astype(np.float32)
        resampled = True

    # Always align + window, and always CEEMDAN (not optional).
    # For performance, we CEEMDAN-denoise the aligned window (small) and re-align once more.
    signal_raw, peak0 = _align_and_window(signal, pre=align_pre, post=align_post)
    case_seed = None
    if ceemdan_seed is not None:
        digest = hashlib.sha256(case_id.encode("utf-8")).digest()
        case_seed = (int(ceemdan_seed) + int.from_bytes(digest[:4], "big")) % (2**32)
    signal = _ceemdan_denoise(
        signal_raw,
        trials=int(ceemdan_trials),
        noise_strength=float(ceemdan_noise_strength),
        max_imf=ceemdan_max_imf,
        seed=case_seed,
    )
    signal, peak1 = _align_and_window(signal, pre=align_pre, post=align_post)
    peak_index = int(peak1 if peak1 != peak0 else peak0)

    return _CaseRow(
        case_id=case_id,
        signal=signal,
        signal_raw=signal_raw,
        depth=depth,
        thickness=thickness,
        conductivity_multiplier=conductivity_multiplier,
        time_ms=canonical_time_ms,
        time_indices=np.asarray(time_indices, dtype=np.int32),
        dt_ms=float(dt_ms),
        voxel_length_um=float(voxel_length_um),
        resampled=resampled,
        peak_index=peak_index,
    )


def main() -> int:
    args = _parse_args()
    repo_root = _repo_root()

    import sys

    sys.path.insert(0, str(repo_root / "src"))
    from lmt_optical.layout import StudyLayout, default_outputs_root  # noqa: E402

    outputs_root = Path(args.outputs_root) if args.outputs_root else default_outputs_root()
    layout = StudyLayout(outputs_root=outputs_root, study_id=args.study_id)
    layout.ensure()

    dataset_dir = layout.nn_dataset_dir / str(args.dataset_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    align_pre = int(args.align_pre)
    align_post = int(args.align_post)

    roi_box = str(args.roi_box) if args.roi_box else None
    roi_pixel_str = str(args.roi_pixel) if args.roi_pixel else None
    roi_pixel: tuple[int, int] | None = None
    roi_margin: int | None = None

    case_dirs = sorted([p for p in layout.cases_root.iterdir() if p.is_dir()])
    if args.max_cases is not None:
        case_dirs = case_dirs[: max(0, int(args.max_cases))]
    if not case_dirs:
        raise FileNotFoundError(f"No cases under {layout.cases_root}")

    skipped: list[dict[str, str]] = []
    tasks: list[_CaseTask] = []
    for case_dir in case_dirs:
        case_id = case_dir.name
        raw_opencm_dir = case_dir / "raw" / "opencm"
        if not (raw_opencm_dir / "emission_detector_top.npy").is_file():
            skipped.append({"case_id": case_id, "reason": "missing raw/opencm/emission_detector_top.npy"})
            continue
        if not (raw_opencm_dir / "emission_voxels.h5").is_file():
            skipped.append({"case_id": case_id, "reason": "missing raw/opencm/emission_voxels.h5"})
            continue
        if not (raw_opencm_dir / "vm_voxels.h5").is_file():
            skipped.append({"case_id": case_id, "reason": "missing raw/opencm/vm_voxels.h5"})
            continue
        tasks.append(_CaseTask(case_id=case_id, raw_opencm_dir=raw_opencm_dir))

    if not tasks:
        raise FileNotFoundError(
            f"No cases with required artifacts under {layout.cases_root} (need raw/opencm/emission_detector_top.npy, emission_voxels.h5, vm_voxels.h5)"
        )

    # ROI selection:
    # - If --roi-pixel is set: use that.
    # - Else if --roi-box is set: use mean in ROI.
    # - Else if --roi-margin is set: use mean in a cropped image.
    # - Else: default to a single center pixel (keeps the signal "raw", avoids ROI-mean smoothing).
    if roi_pixel_str is not None:
        roi_pixel = _parse_roi_pixel(roi_pixel_str)
        roi_box = None
        roi_margin = None
    elif roi_box is not None:
        _parse_roi_box(roi_box)
        roi_margin = None
    elif args.roi_margin is not None:
        roi_margin = int(max(0, args.roi_margin))
    else:
        det0 = np.load(tasks[0].raw_opencm_dir / "emission_detector_top.npy", mmap_mode="r")
        if det0.ndim != 3:
            raise ValueError(f"Expected emission_detector_top as (T, X, Y), got shape={det0.shape}")
        _, nx0, ny0 = det0.shape
        roi_pixel = (int(nx0 // 2), int(ny0 // 2))

    # Choose a canonical time axis from the first available case for reproducibility.
    t0 = tasks[0]
    attrs0, time_indices0 = _load_time_indices_from_emission_h5(t0.raw_opencm_dir / "emission_voxels.h5")
    dt_ms0, _ = _load_vm_meta_from_h5(t0.raw_opencm_dir / "vm_voxels.h5")
    canonical_time_ms = (time_indices0.astype(np.float32) * np.float32(dt_ms0)).astype(np.float32, copy=False)

    rows: list[_CaseRow] = []
    aligned_peak_indices: dict[str, int] = {}

    workers = int(max(1, args.workers))
    progress_every = int(max(1, args.progress_every))
    started = time.perf_counter()
    if roi_pixel is not None:
        roi_desc = f"pixel(x={roi_pixel[0]},y={roi_pixel[1]})"
    elif roi_box is not None:
        roi_desc = f"mean(box={roi_box})"
    else:
        roi_desc = f"mean(margin={roi_margin})"
    print(
        f"[nn_dataset] cases={len(tasks)} workers={workers} roi={roi_desc} ceemdan_trials={int(args.ceemdan_trials)}",
        flush=True,
    )

    if workers == 1:
        for i, task in enumerate(tasks, start=1):
            row = _process_case(
                task,
                roi_box=roi_box,
                roi_pixel=roi_pixel,
                roi_margin=roi_margin,
                align_pre=align_pre,
                align_post=align_post,
                ceemdan_trials=int(args.ceemdan_trials),
                ceemdan_noise_strength=float(args.ceemdan_noise_strength),
                ceemdan_max_imf=args.ceemdan_max_imf,
                ceemdan_seed=args.ceemdan_seed,
                canonical_time_ms=canonical_time_ms,
            )
            rows.append(row)
            aligned_peak_indices[row.case_id] = int(row.peak_index)
            if (i % progress_every) == 0 or i == len(tasks):
                elapsed = time.perf_counter() - started
                rate = i / max(elapsed, 1e-9)
                eta_s = (len(tasks) - i) / max(rate, 1e-9)
                print(f"[nn_dataset] progress {i}/{len(tasks)} elapsed={elapsed:.1f}s eta={eta_s:.1f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(
                    _process_case,
                    task,
                    roi_box=roi_box,
                    roi_pixel=roi_pixel,
                    roi_margin=roi_margin,
                    align_pre=align_pre,
                    align_post=align_post,
                    ceemdan_trials=int(args.ceemdan_trials),
                    ceemdan_noise_strength=float(args.ceemdan_noise_strength),
                    ceemdan_max_imf=args.ceemdan_max_imf,
                    ceemdan_seed=args.ceemdan_seed,
                    canonical_time_ms=canonical_time_ms,
                )
                for task in tasks
            ]
            done = 0
            for fut in as_completed(futs):
                row = fut.result()
                rows.append(row)
                aligned_peak_indices[row.case_id] = int(row.peak_index)
                done += 1
                if (done % progress_every) == 0 or done == len(tasks):
                    elapsed = time.perf_counter() - started
                    rate = done / max(elapsed, 1e-9)
                    eta_s = (len(tasks) - done) / max(rate, 1e-9)
                    print(f"[nn_dataset] progress {done}/{len(tasks)} elapsed={elapsed:.1f}s eta={eta_s:.1f}s", flush=True)

    # Preserve stable ordering for downstream metadata and debugging.
    rows = sorted(rows, key=lambda r: r.case_id)

    signals = np.stack([r.signal for r in rows], axis=0).astype(np.float32, copy=False)  # (N,L)
    signals_raw = np.stack([r.signal_raw for r in rows], axis=0).astype(np.float32, copy=False)  # (N,L)
    depths = np.asarray([r.depth for r in rows], dtype=np.int32)
    thicknesses = np.asarray([r.thickness for r in rows], dtype=np.int32)
    conductivity_multipliers = np.asarray([r.conductivity_multiplier for r in rows], dtype=np.float32)
    case_ids = [r.case_id for r in rows]
    time_indices = np.asarray(time_indices0, dtype=np.int32)
    time_ms = canonical_time_ms.astype(np.float32, copy=False)
    if time_ms.size < 2:
        frame_dt_ms = float(rows[0].dt_ms)
    else:
        frame_dt_ms = float(np.median(np.diff(time_ms.astype(np.float64))))
    time_ms = (np.arange(-align_pre, align_post + 1, dtype=np.float32) * np.float32(frame_dt_ms)).astype(np.float32)

    voxel_lengths = np.asarray([r.voxel_length_um for r in rows], dtype=np.float64)
    if float(voxel_lengths.max() - voxel_lengths.min()) > 1e-6:
        raise ValueError(f"Inconsistent voxel_length_um across cases: min={voxel_lengths.min()} max={voxel_lengths.max()}")

    np.save(dataset_dir / "signals.npy", signals)
    np.save(dataset_dir / "signals_raw.npy", signals_raw)
    np.save(dataset_dir / "targets_depth.npy", depths)
    np.save(dataset_dir / "targets_thickness.npy", thicknesses)
    np.save(dataset_dir / "targets_conductivity_multiplier.npy", conductivity_multipliers)
    # After alignment/windowing, "time indices" are relative to the aligned peak,
    # not the original VM frame indices used by the emission renderer.
    rel_time_indices = np.arange(-align_pre, align_post + 1, dtype=np.int32)
    np.save(dataset_dir / "time_indices.npy", rel_time_indices)
    np.save(dataset_dir / "time_ms.npy", time_ms)

    (dataset_dir / "case_ids.json").write_text(json.dumps(case_ids, indent=2))

    voxel_length_um = float(rows[0].voxel_length_um)
    voxel_mm = voxel_length_um / 1000.0

    meta = {
        "created_at": _utc_now_iso(),
        "study_id": args.study_id,
        "outputs_root": str(outputs_root),
        "cases_root": str(layout.cases_root),
        "dataset_name": str(args.dataset_name),
        "source": {"kind": "emission_detector_top", "path": "raw/opencm/emission_detector_top.npy"},
        "extraction": {
            "kind": "pixel_xy" if roi_pixel is not None else "mean_xy",
            "roi_pixel": {"x": int(roi_pixel[0]), "y": int(roi_pixel[1])} if roi_pixel is not None else None,
            "roi_box": roi_box,
            "roi_margin": roi_margin,
        },
        "preprocessing": {
            "denoise": {
                "kind": "ceemdan",
                "trials": int(args.ceemdan_trials),
                "noise_strength": float(args.ceemdan_noise_strength),
                "max_imf": args.ceemdan_max_imf,
                "base_seed": args.ceemdan_seed,
            },
            "align": {"kind": "max_dFdt", "pre": align_pre, "post": align_post, "padding": "edge"},
        },
        "split_recommendation": {
            "note": "Splits are a training protocol and are written per NN run (not in the dataset folder).",
        },
        "artifacts": {
            "signals_raw": "signals_raw.npy",
        },
        "labels": {
            "depth": "ischemia_depth",
            "thickness": "ischemia_thickness",
            "conductivity_multiplier": "conductivity_multiplier",
        },
        "time_axis": {
            "kind": "ms",
            "dt_ms_source": "raw/opencm/vm_voxels.h5:voxels.attrs['ep_spacedt_ms']",
            "time_indices_kind": "relative_to_peak",
            "dt_ms": float(rows[0].dt_ms),
            "length": int(time_ms.shape[0]),
        },
        "voxel": {"length_um": voxel_length_um, "length_mm": voxel_mm},
        "shape": {"N": int(signals.shape[0]), "L": int(signals.shape[1])},
        "resampling": {
            "kind": "linear_interp_to_canonical_time_ms",
            "note": "Applied only when a case time axis differs from the canonical study time axis; not intended as a denoiser.",
        },
        "resampled_cases": [r.case_id for r in rows if r.resampled],
        "skipped_cases": skipped,
    }
    meta["aligned_peak_indices"] = aligned_peak_indices
    (dataset_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # Also write a stable pointer for training scripts/tools.
    (layout.nn_dataset_dir / "LATEST").write_text(str(args.dataset_name))

    print(f"[nn_dataset] wrote {dataset_dir}", flush=True)
    print(f"[nn_dataset] N={signals.shape[0]} L={signals.shape[1]} ready", flush=True)
    if skipped:
        print(f"[nn_dataset] skipped {len(skipped)} case(s) missing artifacts", flush=True)
    total_s = time.perf_counter() - started
    print(f"[nn_dataset] total={total_s:.1f}s avg_per_case={total_s/max(1,len(rows)):.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
