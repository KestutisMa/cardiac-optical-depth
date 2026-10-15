"""
AP alignment utilities for DepthCNN pipeline.

This module provides functions to align a single-column AP CSV trace sampled at a known dt (ms)
to the model's grid (default dt=0.1 ms), enforcing a fixed length (default 100 samples) and
placing the maximum upstroke derivative (dV/dt max) at a prescribed anchor sample (default 10th sample).

It can also save an aligned CSV and a comparison plot (original vs aligned window).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def load_csv_signal_singlecol(csv_path: str) -> np.ndarray:
    """
    Load a single-column CSV as a float32 1D array.
    Robustly replaces NaNs/Infs with the median value.
    """
    try:
        arr = np.loadtxt(csv_path, delimiter=",")
    except Exception:
        arr = np.genfromtxt(csv_path, delimiter=",")
    if arr.ndim != 1:
        raise ValueError(f"{csv_path}: expected single-column CSV (got shape {arr.shape}).")
    sig = arr.astype(np.float32, copy=False)
    if not np.isfinite(sig).any():
        raise ValueError(f"{csv_path}: no finite values found.")
    med = float(np.nanmedian(sig)) if np.isfinite(np.nanmedian(sig)) else 0.0
    sig = np.nan_to_num(sig, nan=med, posinf=med, neginf=med).astype(np.float32, copy=False)
    return sig


def resample_uniform_linear(y: np.ndarray, src_dt_ms: float, dst_dt_ms: float) -> np.ndarray:
    """
    Linear-resample 1D series from src_dt_ms to dst_dt_ms (both in ms).

    Keeps the original duration (end time) and ensures last destination time does not exceed source.
    """
    if src_dt_ms <= 0 or dst_dt_ms <= 0:
        raise ValueError("src_dt_ms and dst_dt_ms must be positive.")
    n = y.size
    if n < 2:
        return y.astype(np.float32, copy=False)
    t_src = np.arange(n, dtype=np.float64) * float(src_dt_ms)
    T = t_src[-1]
    m = int(np.floor(T / float(dst_dt_ms))) + 1
    t_dst = np.arange(m, dtype=np.float64) * float(dst_dt_ms)
    t_dst = np.clip(t_dst, t_src[0], t_src[-1])
    y_dst = np.interp(t_dst, t_src, y.astype(np.float64)).astype(np.float32)
    return y_dst


def moving_average(y: np.ndarray, win: int) -> np.ndarray:
    """
    Simple centered moving average with odd window length, edge-padded.
    """
    win = int(max(1, win))
    if win == 1:
        return y.astype(np.float32, copy=False)
    if win % 2 == 0:
        win += 1
    k = np.ones(win, dtype=np.float32) / float(win)
    pad = win // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    out = np.convolve(ypad, k, mode="valid")
    return out.astype(np.float32, copy=False)


def argmax_dvdt(y: np.ndarray, dt_ms: float, smooth_ms: float = 0.2) -> int:
    """
    Find index of maximum upstroke derivative (dV/dt max) after light smoothing.

    Parameters
    ----------
    y : np.ndarray
        Signal on a uniform grid with spacing dt_ms.
    dt_ms : float
        Sampling interval in milliseconds.
    smooth_ms : float
        Moving-average window (in ms) applied before derivative.

    Returns
    -------
    int
        Index (0-based) in y of dV/dt maximum.
    """
    if y.size < 3:
        return int(np.argmax(y))
    win = max(1, int(round(smooth_ms / max(dt_ms, 1e-12))))
    y_s = moving_average(y, win)
    dvdt = np.gradient(y_s.astype(np.float64), float(dt_ms))
    idx = int(np.argmax(dvdt))
    return idx


def _safe_window(y: np.ndarray, start: int, length: int) -> np.ndarray:
    """
    Extract y[start:start+length] with edge padding if needed.
    """
    if length <= 0:
        return np.empty((0,), dtype=np.float32)
    end = start + length
    if start >= 0 and end <= y.size:
        return y[start:end].astype(np.float32, copy=False)
    left_pad = max(0, -start)
    right_pad = max(0, end - y.size)
    s = max(0, start)
    e = min(y.size, end)
    seg = y[s:e].astype(np.float32, copy=False)
    if left_pad or right_pad:
        seg = np.pad(seg, (left_pad, right_pad), mode="edge")
    return seg


def align_ap_signal_to_model_grid(
    ap: np.ndarray,
    dt_ms: float,
    *,
    target_len: int = 100,
    model_dt_ms: float = 0.1,
    peak_index_1based: int = 10,
    smooth_ms: float = 0.2,
) -> np.ndarray:
    """
    Align an AP trace so that sample #peak_index_1based (1-based) coincides
    with max dV/dt on a model grid (model_dt_ms), and output exactly target_len samples.

    Steps:
      1) Resample full AP from dt_ms -> model_dt_ms (linear interpolation).
      2) Find dV/dt max on the resampled signal.
      3) Extract a window of length target_len so that index (peak_index_1based-1)
         equals the dV/dt-peak index, padding edges if needed.
    """
    if peak_index_1based < 1 or peak_index_1based > target_len:
        raise ValueError("peak_index_1based must be in [1, target_len].")
    ap_resamp = resample_uniform_linear(ap, dt_ms, model_dt_ms)
    i_peak = argmax_dvdt(ap_resamp, model_dt_ms, smooth_ms=smooth_ms)
    anchor = peak_index_1based - 1
    start = int(i_peak - anchor)
    aligned = _safe_window(ap_resamp, start, target_len)
    return aligned


def align_ap_csv(
    csv_in: str,
    dt_ms: float,
    *,
    csv_out: Optional[str] = None,
    plot_out: Optional[str] = None,
    target_len: int = 100,
    model_dt_ms: float = 0.1,
    peak_index_1based: int = 10,
    smooth_ms: float = 0.2,
) -> str:
    """
    Align a single-column AP CSV sampled at dt_ms to a model grid (model_dt_ms),
    forcing length=target_len and placing dV/dt max at sample #peak_index_1based.
    Writes a single-column CSV with the aligned data and, optionally, a comparison plot.

    Returns
    -------
    str
        Path to the output CSV file.
    """
    inp = Path(csv_in)
    if csv_out is None:
        csv_out = str(inp.with_name(f"{inp.stem}_aligned_{target_len}x{str(model_dt_ms).replace('.','p')}ms.csv"))
    ap = load_csv_signal_singlecol(str(inp))
    aligned = align_ap_signal_to_model_grid(
        ap,
        dt_ms,
        target_len=target_len,
        model_dt_ms=model_dt_ms,
        peak_index_1based=peak_index_1based,
        smooth_ms=smooth_ms,
    )
    # Save aligned CSV (single column)
    np.savetxt(csv_out, aligned.reshape(-1, 1), delimiter=",", fmt="%.8f")

    # Optional plot: original full trace vs aligned window
    if plot_out:
        import matplotlib.pyplot as plt

        t_orig = np.arange(ap.size, dtype=np.float64) * float(dt_ms)
        t_al = np.arange(aligned.size, dtype=np.float64) * float(model_dt_ms)

        plt.figure(figsize=(9.5, 4.8))
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(t_orig, ap, lw=1.2)
        ax1.set_title(f"Original (dt={dt_ms} ms)")
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Signal (arb.)")

        ax2 = plt.subplot(1, 2, 2)
        ax2.plot(t_al, aligned, lw=1.2)
        ax2.axvline((peak_index_1based - 1) * model_dt_ms, ls="--", lw=0.9, alpha=0.8)
        ax2.set_title(f"Aligned window (L={target_len}, dt={model_dt_ms} ms)\n"
                      f"Anchor @ sample #{peak_index_1based}")
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Signal (arb.)")

        plt.tight_layout()
        plt.savefig(plot_out, dpi=180)

    return csv_out
