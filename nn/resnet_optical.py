#!/usr/bin/env python3
"""Train/Eval CLI for 1D depth regression using DepthCNN.
Imports core components from depth_core.py
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from depth_core import (
    set_global_seed, worker_init_fn,
    RandomAugment, EmissionDepthDataset,
    compute_global_minmax, compute_quantile_minmax,
    load_prepared_dataset,
    DepthCNN, init_kaiming,
)


# --------------------------
# Config dataclass
# --------------------------

@dataclass
class TrainConfig:
    data_dir: str
    checkpoint: str = "best_depthcnn.pth"
    seed: int = 11
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # master switch: disable uncertainty (simplifies training/eval + outputs)
    enable_uncertainty: bool = True

    # data
    batch_size: int = 32
    num_workers: int = 0
    stratified_split: bool = True
    quantile_norm: bool = True
    q_low: float = 0.001
    q_high: float = 0.999
    channel_mask: Optional[List[float]] = None

    # aug
    aug_p: float = 0.25
    aug_noise_min: float = 0.0
    aug_noise_max: float = 0.0025
    aug_max_shift: int = 2
    aug_amp_min: float = 0.985
    aug_amp_max: float = 1.015

    # schedule
    epochs: int = 180
    min_epochs: int = 40
    patience: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    plateau_factor: float = 0.5
    plateau_patience: int = 4
    plateau_min_lr: float = 1e-5

    # loss
    huber_beta: float = 0.75
    loss_weighting: bool = True
    num_bins: int = 12
    weight_pow: float = 2.0
    # heteroscedastic controls
    weight_logvar_term: bool = False
    logvar_min: float = -6.0
    logvar_max: float = 1.0
    # encourage higher uncertainty at higher depth (optional)
    monotonic_uncertainty: bool = False
    mono_lambda: float = 0.05

    # stochastic eval
    mc_samples: int = 50
    tta_samples: int = 1
    tta_noise: float = 0.0
    tta_max_shift: int = 0
    tta_amp_min: float = 1.0
    tta_amp_max: float = 1.0

    # model toggles
    attn_pool: bool = True
    heteroscedastic: bool = True
    tie_logvar_to_mu: bool = True
    weight_logvar_term: bool = True

    # label/depth mapping (index range and physical scale)
    depth_min_idx: int = 20
    depth_max_idx: int = 80
    voxel_mm: float = 0.05

    # viz/outputs
    plot_unc: str = "both"
    plot_raw_samples: bool = True
    samples_alpha: float = 0.12
    samples_jitter: float = 0.015
    save_plot: Optional[str] = "resnet_optical.png"
    save_stochastic: Optional[str] = "stochastic_predictions.csv"
    save_stochastic_raw: Optional[str] = "stochastic_samples_long.csv"
    calibrate_std_on_val: bool = True
    calibrate_std_by_depth: bool = True
    calibrate_std_monotonic: bool = True
    calib_bins: int = 6
    calib_min_bin: int = 5
    calib_target_coverage: float = 0.95


def _apply_channel_mask_tensor(x: torch.Tensor, channel_mask: Optional[object]) -> torch.Tensor:
    if channel_mask is None:
        return x
    if isinstance(channel_mask, str):
        values = [float(v.strip()) for v in channel_mask.split(",") if v.strip()]
    else:
        values = [float(v) for v in channel_mask]  # type: ignore[union-attr]
    if len(values) != 3:
        raise ValueError(f"channel_mask must have length 3, got {channel_mask!r}")
    if values == [1.0, 1.0, 1.0]:
        return x
    shape = (1, 3, 1) if x.ndim == 3 else (3, 1)
    mask = torch.tensor(values, dtype=x.dtype, device=x.device).view(*shape)
    return x * mask


# --------------------------
# Splits / loaders
# --------------------------

def make_splits(depths: np.ndarray, seed: int, stratified: bool = True) -> Tuple[List[int], List[int], List[int]]:
    """Split sample indices into train/val/test.

    If stratified=True, preserve the depth histogram per split to reduce
    seed sensitivity on small datasets.
    """
    N = int(depths.shape[0])
    all_idx = list(range(N))
    if not stratified:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(N)
        n1, n2 = int(0.7 * N), int(0.1 * N)
        train_idx, val_idx, test_idx = perm[:n1], perm[n1:n1 + n2], perm[n1 + n2 :]
        return [all_idx[i] for i in train_idx], [all_idx[i] for i in val_idx], [all_idx[i] for i in test_idx]

    from collections import defaultdict
    by_depth = defaultdict(list)
    for i in range(N):
        by_depth[int(depths[i])].append(int(i))

    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for d, lst in by_depth.items():
        if not lst:
            continue
        idx = np.arange(len(lst))
        rng.shuffle(idx)
        n = len(idx)
        # base fractional split
        n1 = int(np.floor(0.7 * n))
        n2 = int(np.floor(0.1 * n))
        # ensure we don't starve val/test when counts are small
        if n >= 3:
            if n1 < 1:
                n1 = 1
            if n - n1 < 2:
                # leave room for at least one val and one test
                n1 = max(1, n - 2)
            if n2 < 1:
                n2 = 1
            if n - n1 - n2 < 1:
                # borrow one from val if needed
                if n2 > 1:
                    n2 -= 1
                else:
                    n1 = max(1, n1 - 1)
        else:
            # too few samples to stratify meaningfully: all go to train except last to test
            n1 = max(1, n - 1)
            n2 = 0
        train += [lst[i] for i in idx[:n1]]
        val   += [lst[i] for i in idx[n1:n1 + n2]]
        test  += [lst[i] for i in idx[n1 + n2 :]]
    # If val is still empty across all depths, fall back to non-stratified global split
    if len(val) == 0 or len(test) == 0:
        perm = rng.permutation(N)
        if N == 1:
            train = [int(perm[0])]
            val = []
            test = []
        elif N == 2:
            train = [int(perm[0])]
            val = []
            test = [int(perm[1])]
        else:
            n1, n2 = max(1, int(0.7 * N)), max(1, int(0.1 * N))
            if n1 + n2 >= N:
                n1 = max(1, N - 2)
                n2 = 1
            train = [int(i) for i in perm[:n1]]
            val   = [int(i) for i in perm[n1:n1 + n2]]
            test  = [int(i) for i in perm[n1 + n2 :]]
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test


def build_dataloaders(cfg: TrainConfig, *,
                      fixed_train: Optional[List[int]] = None,
                      fixed_val: Optional[List[int]] = None,
                      fixed_test: Optional[List[int]] = None,
                      fixed_gmin: Optional[float] = None,
                      fixed_gmax: Optional[float] = None):
    prepared = load_prepared_dataset(cfg.data_dir)
    signals = prepared.signals
    depths = prepared.depth

    if fixed_train is None:
        splits_env = os.environ.get("LMT_NN_SPLITS_PATH")
        splits_path = Path(splits_env) if splits_env else None
        if splits_path is not None and splits_path.is_file():
            splits = json.loads(splits_path.read_text()).get("splits", {})
            train_idx = [int(i) for i in splits.get("train", [])]
            val_idx = [int(i) for i in splits.get("val", [])]
            test_idx = [int(i) for i in splits.get("test", [])]
            all_idx = set(train_idx) | set(val_idx) | set(test_idx)
            if not all_idx:
                raise ValueError(f"{splits_path} exists but contains empty splits")
            if len(all_idx) != len(train_idx) + len(val_idx) + len(test_idx):
                raise ValueError(f"{splits_path} has overlapping indices across splits")
        else:
            train_idx, val_idx, test_idx = make_splits(depths, cfg.seed, stratified=cfg.stratified_split)
    else:
        train_idx, val_idx, test_idx = fixed_train, fixed_val, fixed_test

    if fixed_gmin is None:
        if cfg.quantile_norm:
            gmin, gmax = compute_quantile_minmax(signals, train_idx, cfg.q_low, cfg.q_high)
        else:
            gmin, gmax = compute_global_minmax(signals, train_idx)
    else:
        gmin, gmax = fixed_gmin, fixed_gmax

    train_aug = RandomAugment(
        p=cfg.aug_p,
        noise_std_range=(cfg.aug_noise_min, cfg.aug_noise_max),
        max_shift=cfg.aug_max_shift,
        amp_jitter=(cfg.aug_amp_min, cfg.aug_amp_max),
    )

    dr = (cfg.depth_min_idx, cfg.depth_max_idx)
    train_ds = EmissionDepthDataset(signals, depths, train_idx, transform=train_aug, normalize=True,
                                    global_min=gmin, global_max=gmax, depth_range=dr,
                                    channel_mask=cfg.channel_mask)
    val_ds   = EmissionDepthDataset(signals, depths, val_idx,   transform=None,   normalize=True,
                                    global_min=gmin, global_max=gmax, depth_range=dr,
                                    channel_mask=cfg.channel_mask)
    test_ds  = EmissionDepthDataset(signals, depths, test_idx,  transform=None,   normalize=True,
                                    global_min=gmin, global_max=gmax, depth_range=dr,
                                    channel_mask=cfg.channel_mask)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, worker_init_fn=worker_init_fn,
                              pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, worker_init_fn=worker_init_fn,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, worker_init_fn=worker_init_fn,
                              pin_memory=True)
    return train_loader, val_loader, test_loader, train_idx, val_idx, test_idx, gmin, gmax


# --------------------------
# Train
# --------------------------

def train_and_save(cfg: TrainConfig) -> str:
    set_global_seed(cfg.seed)
    device = torch.device(cfg.device)

    # Splits & normalization
    tl, vl, te, train_idx, val_idx, test_idx, gmin, gmax = build_dataloaders(cfg)

    # Train directly on y∈[0,1] without z-scoring labels.
    # This removes seed sensitivity when the train split happens to be narrow in depth range.
    y_mean = 0.0
    y_std  = 1.0
    # Keep explicit copy of the configured label index range for checkpoint metadata
    dmin_cfg, dmax_cfg = int(cfg.depth_min_idx), int(cfg.depth_max_idx)

    model = DepthCNN(attn_pool=cfg.attn_pool, hetero=cfg.heteroscedastic,
                     tie_logvar_to_mu=cfg.tie_logvar_to_mu).to(device)
    model.apply(init_kaiming)
    with torch.no_grad():
        if hasattr(model, 'fc') and model.fc.bias is not None:
            model.fc.bias.zero_()

    opt = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=cfg.plateau_factor,
        patience=cfg.plateau_patience, min_lr=cfg.plateau_min_lr)
    huber = nn.SmoothL1Loss(beta=cfg.huber_beta)

    best_val = float('inf')
    best_epoch = -1
    epochs_no_improve = 0
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_rmse_mm": [],
        "lr": [],
    }

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tr_loss = 0.0
        n_seen = 0
        for x, y, _ in tl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_z = y  # identity scaling
            opt.zero_grad(set_to_none=True)
            out = model(x)
            if cfg.heteroscedastic:
                mu_z, logvar_z = out
                logvar_z = torch.clamp(logvar_z, min=cfg.logvar_min, max=cfg.logvar_max)
                if cfg.loss_weighting:
                    bins = torch.linspace(0.0, 1.0, cfg.num_bins + 1, device=y.device)
                    idx = torch.bucketize(y, bins[1:-1])
                    counts = torch.bincount(idx, minlength=cfg.num_bins).float() + 1e-6
                    inv_freq = 1.0 / counts
                    w = inv_freq.pow(cfg.weight_pow)[idx]
                    if cfg.weight_logvar_term:
                        nll = 0.5 * (torch.exp(-logvar_z) * (mu_z - y_z) ** 2 + logvar_z)
                        loss = (w * nll).mean()
                    else:
                        # weight only data-fit term to avoid driving logvar to a constant
                        loss = (0.5 * (w * torch.exp(-logvar_z) * (mu_z - y_z) ** 2 + logvar_z)).mean()
                else:
                    nll = 0.5 * (torch.exp(-logvar_z) * (mu_z - y_z) ** 2 + logvar_z)
                    loss = nll.mean()

                # Optional monotonicity regularizer: encourage log-variance to
                # increase with depth by penalizing negative covariance between
                # target depth (in [0,1]) and log-variance.
                if cfg.monotonic_uncertainty and cfg.mono_lambda > 0:
                    y_cent = y - y.mean()
                    lv_cent = logvar_z - logvar_z.mean()
                    cov = (y_cent * lv_cent).mean()
                    mono_penalty = torch.relu(-cov)  # penalize negative slope
                    loss = loss + cfg.mono_lambda * mono_penalty
            else:
                pred_z = out
                if cfg.loss_weighting:
                    bins = torch.linspace(0.0, 1.0, cfg.num_bins + 1, device=y.device)
                    idx = torch.bucketize(y, bins[1:-1])
                    counts = torch.bincount(idx, minlength=cfg.num_bins).float() + 1e-6
                    inv_freq = 1.0 / counts
                    w = inv_freq.pow(cfg.weight_pow)[idx]
                    loss = (w * F.smooth_l1_loss(pred_z, y_z, beta=cfg.huber_beta, reduction='none')).mean()
                else:
                    loss = huber(pred_z, y_z)
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            bs = x.size(0)
            tr_loss += loss.item() * bs
            n_seen += bs
        tr_loss /= max(1, n_seen)

        # validate (clean)
        model.eval()
        va_loss = 0.0
        n_val = 0
        preds_norm = []
        gts_norm = []
        with torch.no_grad():
            for x, y, _ in vl:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                y_z = y  # identity scaling
                out = model(x)
                if cfg.heteroscedastic:
                    mu_z, logvar_z = out
                    logvar_z = torch.clamp(logvar_z, min=cfg.logvar_min, max=cfg.logvar_max)
                    loss = (0.5 * (torch.exp(-logvar_z) * (mu_z - y_z) ** 2 + logvar_z)).mean()
                    pred = mu_z
                else:
                    loss = huber(out, y_z)
                    pred = out
                bs = x.size(0)
                va_loss += loss.item() * bs
                n_val += bs
                preds_norm.append(torch.clamp(pred, 0.0, 1.0).detach().cpu())
                gts_norm.append(torch.clamp(y, 0.0, 1.0).detach().cpu())
        va_loss /= max(1, n_val)

        history["epoch"].append(int(epoch))
        history["train_loss"].append(float(tr_loss))
        history["val_loss"].append(float(va_loss))
        try:
            pr = torch.cat(preds_norm, dim=0).numpy().reshape(-1)
            gt = torch.cat(gts_norm, dim=0).numpy().reshape(-1)
            pr_mm = (_denorm_depth(pr, depth_range=(dmin_cfg, dmax_cfg)) * float(cfg.voxel_mm)).astype(np.float64, copy=False)
            gt_mm = (_denorm_depth(gt, depth_range=(dmin_cfg, dmax_cfg)) * float(cfg.voxel_mm)).astype(np.float64, copy=False)
            rmse_mm = float(np.sqrt(np.mean((pr_mm - gt_mm) ** 2)))
        except Exception:
            rmse_mm = float("nan")
        history["val_rmse_mm"].append(float(rmse_mm))
        history["lr"].append(float(opt.param_groups[0].get("lr", cfg.lr)))

        print(f"Epoch {epoch:03d} | Train={tr_loss:.6f} | Val={va_loss:.6f}")
        sched.step(va_loss)
        if va_loss < best_val - 1e-8:
            best_val = va_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                'model_state': model.state_dict(),
                'epoch': epoch,
                'val_loss': va_loss,
                'config': cfg.__dict__,
                'norm': {'gmin': float(gmin), 'gmax': float(gmax)},
                'label_norm': {'mean': float(y_mean), 'std': float(y_std), 'kind': 'identity'},
                # store the label index range used during training for robust mm mapping
                'label_range': {'min_idx': int(dmin_cfg), 'max_idx': int(dmax_cfg), 'voxel_mm': float(cfg.voxel_mm)},
                'splits': {'train': train_idx, 'val': val_idx, 'test': test_idx},
            }, cfg.checkpoint)
        else:
            epochs_no_improve += 1
            if epoch >= cfg.min_epochs and epochs_no_improve >= cfg.patience:
                print(f"Early stopping at epoch {epoch} (best {best_epoch}, Val={best_val:.6f})")
                break
    print(f"Best model: epoch {best_epoch} | Val={best_val:.6f} -> {cfg.checkpoint}")

    # Save training curves next to the checkpoint.
    out_dir = Path(cfg.checkpoint).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    try:
        fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
        ax1.plot(history["epoch"], history["train_loss"], label="train", lw=1.7)
        ax1.plot(history["epoch"], history["val_loss"], label="val", lw=1.7)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.25)
        ax2 = ax1.twinx()
        ax2.plot(history["epoch"], history["lr"], label="lr", lw=1.0, alpha=0.35, color="black")
        ax2.set_ylabel("LR")
        ax1.set_title("Training curves")
        ax1.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / "learning_curves.png", dpi=170)
        plt.close(fig)
    except Exception:
        pass
    return cfg.checkpoint


# --------------------------
# Eval
# --------------------------

def _denorm_depth(y_norm: np.ndarray, depth_range: Tuple[int, int] = (20, 80)) -> np.ndarray:
    lo, hi = depth_range
    return y_norm * (hi - lo) + lo


def evaluate(cfg: TrainConfig, *, eval_split: str = "test") -> None:
    set_global_seed(cfg.seed)
    device = torch.device(cfg.device)

    if not os.path.exists(cfg.checkpoint):
        raise FileNotFoundError(cfg.checkpoint)
    ckpt = torch.load(cfg.checkpoint, map_location=device)

    train_idx = [int(i) for i in ckpt['splits']['train']]
    val_idx   = [int(i) for i in ckpt['splits']['val']]
    test_idx  = [int(i) for i in ckpt['splits']['test']]
    gmin = ckpt['norm']['gmin']
    gmax = ckpt['norm']['gmax']
    y_mean = ckpt.get('label_norm', {}).get('mean', 0.0)
    y_std  = max(ckpt.get('label_norm', {}).get('std', 1.0), 1e-6)

    # Depth index range used during training (for mm mapping)
    lr = ckpt.get('label_range', {}) or {}
    if lr:
        dmin = int(lr.get('min_idx', 20))
        dmax = int(lr.get('max_idx', 80))
        voxel_mm = float(lr.get('voxel_mm', 0.05))
    else:
        # Backward-compat for older checkpoints trained on 10..80 (0.5–4.0 mm)
        dmin, dmax, voxel_mm = 10, 80, 0.05

    # Ensure dataloaders use the same label mapping
    cfg.depth_min_idx, cfg.depth_max_idx, cfg.voxel_mm = dmin, dmax, voxel_mm
    tl_aug, vl, test_loader, *_ = build_dataloaders(
        cfg,
        fixed_train=train_idx,
        fixed_val=val_idx,
        fixed_test=test_idx,
        fixed_gmin=gmin,
        fixed_gmax=gmax,
    )

    # For reporting, evaluate TRAIN without augmentation.
    prepared = load_prepared_dataset(cfg.data_dir)
    train_eval_ds = EmissionDepthDataset(
        prepared.signals,
        prepared.depth,
        train_idx,
        transform=None,
        normalize=True,
        global_min=gmin,
        global_max=gmax,
        depth_range=(dmin, dmax),
        channel_mask=cfg.channel_mask,
    )
    train_eval_loader = DataLoader(
        train_eval_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        drop_last=False,
    )

    model = DepthCNN(attn_pool=cfg.attn_pool, hetero=cfg.heteroscedastic,
                     tie_logvar_to_mu=cfg.tie_logvar_to_mu).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    def enable_dropout_only(m: nn.Module):
        for mod in m.modules():
            if isinstance(mod, nn.Dropout):
                mod.train()

    def tta_perturb(x: torch.Tensor) -> torch.Tensor:
        if cfg.tta_samples <= 1:
            return x
        with torch.no_grad():
            xb = x.clone()
            if cfg.tta_max_shift:
                shift = np.random.randint(-cfg.tta_max_shift, cfg.tta_max_shift + 1)
                xb = torch.roll(xb, shifts=shift, dims=-1)
            if (cfg.tta_amp_min != 1.0) or (cfg.tta_amp_max != 1.0):
                scale = float(np.random.uniform(cfg.tta_amp_min, cfg.tta_amp_max))
                xb = xb * scale
            if cfg.tta_noise > 0:
                xb = xb + torch.randn_like(xb) * cfg.tta_noise
            return xb

    @torch.no_grad()
    def predict_with_uncertainty(x: torch.Tensor):
        mcK = max(int(cfg.mc_samples), 1)
        ttaK = max(int(cfg.tta_samples), 1)
        if mcK == 1 and ttaK == 1:
            out = model(x)
            if cfg.heteroscedastic:
                mu_z, logvar_z = out
                std_ale_z = torch.exp(0.5 * logvar_z).cpu().numpy()
                std_tot_z = std_ale_z
                std_epi_z = None
                return mu_z.cpu().numpy(), std_tot_z, std_ale_z, std_epi_z, None
            else:
                pr_z = out.cpu().numpy()
                return pr_z, None, None, None, None
        mus, vars_ = [], []
        for _ in range(mcK):
            model.eval()
            if mcK > 1:
                enable_dropout_only(model)
            for _ in range(ttaK):
                xt = tta_perturb(x)
                out = model(xt)
                if cfg.heteroscedastic:
                    mu_z, logvar_z = out
                    mus.append(mu_z.cpu().numpy())
                    vars_.append(torch.exp(logvar_z).cpu().numpy())
                else:
                    mus.append(out.cpu().numpy())
        M = np.stack(mus, axis=0)  # (S,B)
        mean_mu = M.mean(axis=0)
        var_ep = M.var(axis=0)
        if cfg.heteroscedastic and vars_:
            V = np.stack(vars_, axis=0)
            ale = V.mean(axis=0)
            std_ale = np.sqrt(ale)
            std_epi = np.sqrt(var_ep)
            std_total = np.sqrt(ale + var_ep)
        else:
            std_ale = None
            std_epi = np.sqrt(var_ep)
            std_total = np.sqrt(var_ep)
        return mean_mu, std_total, std_ale, std_epi, M

    def run_eval(loader, tag: str, *, scale_std=1.0):
        preds, gts = [], []
        stds_total, stds_ale, stds_epi = [], [], []
        all_samples = []
        for x, y, _ in loader:
            x = x.to(device)
            mean_z, std_z, std_ale_z, std_epi_z, samples_z = predict_with_uncertainty(x)
            mean_norm = np.clip(mean_z * y_std + y_mean, 0.0, 1.0)
            std_total_norm = (std_z * y_std) if std_z is not None else None
            std_ale_norm = (std_ale_z * y_std) if std_ale_z is not None else None
            std_epi_norm = (std_epi_z * y_std) if std_epi_z is not None else None
            preds.append(mean_norm)
            gts.append(y.numpy())
            if std_total_norm is not None:
                stds_total.append(std_total_norm)
            if std_ale_norm is not None:
                stds_ale.append(std_ale_norm)
            if std_epi_norm is not None:
                stds_epi.append(std_epi_norm)
            if samples_z is not None:
                all_samples.append(samples_z * y_std + y_mean)
        pr = np.concatenate(preds)
        gt = np.concatenate(gts)
        pr_mm = _denorm_depth(pr, depth_range=(dmin, dmax)) * voxel_mm
        gt_mm = _denorm_depth(gt, depth_range=(dmin, dmax)) * voxel_mm
        r, _ = pearsonr(gt_mm, pr_mm)
        ss_res = np.sum((gt_mm - pr_mm) ** 2)
        ss_tot = np.sum((gt_mm - np.mean(gt_mm)) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        rmse = float(np.sqrt(np.mean((pr_mm - gt_mm) ** 2)))
        std_mm = None
        std_ale_mm = None
        std_epi_mm = None
        if stds_total:
            std_norm = np.concatenate(stds_total)
            std_mm = std_norm * (dmax - dmin) * voxel_mm
        if stds_ale:
            std_ale_norm = np.concatenate(stds_ale)
            std_ale_mm = std_ale_norm * (dmax - dmin) * voxel_mm
        if stds_epi:
            std_epi_norm = np.concatenate(stds_epi)
            std_epi_mm = std_epi_norm * (dmax - dmin) * voxel_mm

        if std_mm is not None:
            # Apply optional scaling: float or callable returning per-sample scale
            if callable(scale_std):
                s = scale_std(pr_mm)
                if np.isscalar(s):
                    s0 = float(s)
                    std_mm = std_mm * s0
                    if std_ale_mm is not None:
                        std_ale_mm = std_ale_mm * s0
                    if std_epi_mm is not None:
                        std_epi_mm = std_epi_mm * s0
                else:
                    s = np.asarray(s, dtype=float)
                    if s.shape == std_mm.shape:
                        std_mm = std_mm * s
                        if std_ale_mm is not None and s.shape == std_ale_mm.shape:
                            std_ale_mm = std_ale_mm * s
                        if std_epi_mm is not None and s.shape == std_epi_mm.shape:
                            std_epi_mm = std_epi_mm * s
                    else:
                        s0 = float(np.mean(s))
                        std_mm = std_mm * s0
                        if std_ale_mm is not None:
                            std_ale_mm = std_ale_mm * s0
                        if std_epi_mm is not None:
                            std_epi_mm = std_epi_mm * s0
            else:
                s0 = float(scale_std)
                std_mm = std_mm * s0
                if std_ale_mm is not None:
                    std_ale_mm = std_ale_mm * s0
                if std_epi_mm is not None:
                    std_epi_mm = std_epi_mm * s0
        samples_gt_mm, samples_pr_mm = None, None
        if all_samples:
            P = np.concatenate(all_samples, axis=1)  # (S,N)
            samples_pr_mm = _denorm_depth(P, depth_range=(dmin, dmax)) * voxel_mm
            S = P.shape[0]
            samples_gt_mm = np.tile(gt_mm, S)
            samples_pr_mm = samples_pr_mm.reshape(-1)
        if std_mm is not None and std_mm.size == gt_mm.size:
            # quick sanity: correlation between uncertainty and depth (should trend positive)
            try:
                corr_std_depth = float(np.corrcoef(std_mm, gt_mm)[0,1])
            except Exception:
                corr_std_depth = float('nan')
            print(f"[{tag}] r={r:.4f} R2={r2:.4f} RMSE={rmse:.4f}mm | corr(std,depth)={corr_std_depth:.3f}")
        else:
            print(f"[{tag}] r={r:.4f} R2={r2:.4f} RMSE={rmse:.4f}mm")
        return gt_mm, pr_mm, r, r2, rmse, std_mm, std_ale_mm, std_epi_mm, samples_gt_mm, samples_pr_mm

    def _metrics_json_path(tag: str) -> Path:
        # By convention `cfg.checkpoint` is inside the run output dir.
        out_dir = Path(cfg.checkpoint).resolve().parent
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"metrics_{tag.lower()}.json"

    def _write_metrics(tag: str, *, r: float, r2: float, rmse: float, n: int) -> None:
        _metrics_json_path(tag).write_text(
            json.dumps(
                {
                    "split": str(tag).lower(),
                    "n": int(n),
                    "r": float(r),
                    "r2": float(r2),
                    "rmse_mm": float(rmse),
                    "label_range": {"min_idx": int(dmin), "max_idx": int(dmax), "voxel_mm": float(voxel_mm)},
                },
                indent=2,
            )
        )

    eval_split = str(eval_split).strip().lower()
    if eval_split not in {"train", "val", "test", "all"}:
        raise ValueError("--eval-split must be one of: train|val|test|all")

    # Optional std calibration on VAL (only used for TEST eval).
    scale: object = 1.0
    if cfg.calibrate_std_on_val and eval_split in {"test", "all"}:
        v_gt, v_pr, v_r, v_r2, v_rmse, v_std, _, _, _, _ = run_eval(vl, tag="VAL", scale_std=1.0)
        _write_metrics("val", r=float(v_r), r2=float(v_r2), rmse=float(v_rmse), n=int(v_gt.size))
        if v_std is not None:
            # Coverage-based scaling (robust): choose a multiplicative factor so that the
            # empirical two-sided (approx) 95% interval matches the target coverage on VAL.
            from statistics import NormalDist

            target = float(cfg.calib_target_coverage)
            target = float(min(max(target, 0.5), 0.995))
            # For a Normal(0,1): P(|Z| <= z_ref) = target  =>  z_ref = Phi^{-1}((1+target)/2)
            z_ref = float(NormalDist().inv_cdf((1.0 + target) / 2.0))
            # z = |residual| / std
            denom = np.maximum(v_std, 1e-9)
            z = np.abs(v_pr - v_gt) / denom
            q = float(np.quantile(z, target))
            gscale = q / max(1e-6, float(z_ref))
            # Allow shrinking std substantially; cap only extreme expansion.
            gscale = float(max(0.2, min(5.0, gscale)))
            print(f"[CAL] global std scale (target={target:.3f}) = {gscale:.3f}")
            scale = gscale

            # optional depth-wise scaling
            if cfg.calibrate_std_by_depth:
                bins = np.linspace(1.0, 4.0, max(2, int(cfg.calib_bins)) + 1)
                centers = 0.5 * (bins[:-1] + bins[1:])
                # IMPORTANT: use predictions for binning (valid at inference time).
                key = v_pr
                idx = np.digitize(key, bins) - 1
                s_bin = np.full(len(centers), np.nan, dtype=float)
                for b in range(len(centers)):
                    m = (idx == b)
                    if np.sum(m) >= max(3, int(cfg.calib_min_bin)):
                        denom_b = np.maximum(v_std[m], 1e-9)
                        z_b = np.abs(v_pr[m] - v_gt[m]) / denom_b
                        q_b = float(np.quantile(z_b, target))
                        s_bin[b] = q_b / max(1e-6, float(z_ref))
                # fill NaNs with nearest valid or global
                if np.all(~np.isfinite(s_bin)):
                    s_bin[:] = gscale
                else:
                    # forward/backward fill
                    last = gscale
                    for i in range(len(s_bin)):
                        if np.isfinite(s_bin[i]):
                            last = s_bin[i]
                        else:
                            s_bin[i] = last
                    last = s_bin[-1]
                    for i in range(len(s_bin)-1, -1, -1):
                        if np.isfinite(s_bin[i]):
                            last = s_bin[i]
                        else:
                            s_bin[i] = last
                # clip to reasonable range and make monotonic non-decreasing if requested
                s_bin = np.clip(s_bin, 0.2, 5.0)
                if cfg.calibrate_std_monotonic:
                    for i in range(1, len(s_bin)):
                        if s_bin[i] < s_bin[i-1]:
                            s_bin[i] = s_bin[i-1]
                # smooth lightly
                if len(s_bin) >= 3:
                    s_bin = np.convolve(s_bin, np.array([0.25,0.5,0.25]), mode='same')

                def scale_fn(pr_mm: np.ndarray):
                    return np.interp(pr_mm, centers, s_bin, left=s_bin[0], right=s_bin[-1])

                scale = scale_fn

    out_dir = Path(cfg.checkpoint).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, object]] = {}

    if eval_split in {"train", "all"}:
        gt_mm_tr, pr_mm_tr, r_tr, r2_tr, rmse_tr, std_tr, std_ale_tr, std_epi_tr, _, _ = run_eval(
            train_eval_loader, tag="TRAIN", scale_std=1.0
        )
        _write_metrics("train", r=float(r_tr), r2=float(r2_tr), rmse=float(rmse_tr), n=int(gt_mm_tr.size))
        results["train"] = {
            "gt_mm": gt_mm_tr,
            "pred_mm": pr_mm_tr,
            "std_mm": std_tr,
            "std_ale_mm": std_ale_tr,
            "std_epi_mm": std_epi_tr,
            "r": float(r_tr),
            "r2": float(r2_tr),
            "rmse": float(rmse_tr),
        }

    if eval_split in {"val", "all"}:
        gt_mm_va, pr_mm_va, r_va, r2_va, rmse_va, std_va, std_ale_va, std_epi_va, _, _ = run_eval(
            vl, tag="VAL", scale_std=1.0
        )
        _write_metrics("val", r=float(r_va), r2=float(r2_va), rmse=float(rmse_va), n=int(gt_mm_va.size))
        results["val"] = {
            "gt_mm": gt_mm_va,
            "pred_mm": pr_mm_va,
            "std_mm": std_va,
            "std_ale_mm": std_ale_va,
            "std_epi_mm": std_epi_va,
            "r": float(r_va),
            "r2": float(r2_va),
            "rmse": float(rmse_va),
        }

    if eval_split in {"test", "all"}:
        # Run TEST with calibrated std (if enabled)
        gt_mm, pr_mm, r, r2, rmse, std_mm, std_ale_mm, std_epi_mm, samples_gt_mm, samples_pr_mm = run_eval(
            test_loader, tag="TEST", scale_std=scale
        )
        _write_metrics("test", r=float(r), r2=float(r2), rmse=float(rmse), n=int(gt_mm.size))
        results["test"] = {
            "gt_mm": gt_mm,
            "pred_mm": pr_mm,
            "std_mm": std_mm,
            "std_ale_mm": std_ale_mm,
            "std_epi_mm": std_epi_mm,
            "r": float(r),
            "r2": float(r2),
            "rmse": float(rmse),
            "samples_gt_mm": samples_gt_mm,
            "samples_pr_mm": samples_pr_mm,
        }

    def _save_scatter(tag: str, gt_mm: np.ndarray, pr_mm: np.ndarray, *, r: float, r2: float, rmse: float, std_mm: np.ndarray | None) -> None:
        plt.figure(figsize=(7, 6))
        use_std = (std_mm is not None) and (cfg.plot_unc.lower() != 'none')
        if use_std and cfg.plot_unc.lower() in ("color", "both"):
            sc = plt.scatter(gt_mm, pr_mm, s=18, alpha=0.75, c=std_mm)
            cb = plt.colorbar(sc)
            cb.set_label("Predictive std (mm)")
        else:
            plt.scatter(gt_mm, pr_mm, s=18, alpha=0.75)
        if use_std and cfg.plot_unc.lower() in ("errorbar", "both"):
            yerr = 1.96 * std_mm
            plt.errorbar(gt_mm, pr_mm, yerr=yerr, fmt="none", ecolor="k", alpha=0.15, lw=0.6)
        mn, mx = float(min(gt_mm.min(), pr_mm.min())), float(max(gt_mm.max(), pr_mm.max()))
        plt.plot([mn, mx], [mn, mx], "--")
        txt = rf"$r={r:.3f}\quad R^2={r2:.3f}\quad \mathrm{{RMSE}}={rmse:.3f}\,\mathrm{{mm}}$"
        plt.text(
            0.05,
            0.95,
            txt,
            transform=plt.gca().transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", pad=0.3, alpha=0.2),
        )
        plt.xlabel("Ground Truth Depth (mm)")
        plt.ylabel("Predicted Depth (mm)")
        plt.title(f"GT vs prediction ({tag})")
        plt.tight_layout()
        plt.savefig(out_dir / f"gt_vs_pred_{tag}.png", dpi=170)
        plt.close()

    for tag, row in results.items():
        _save_scatter(
            tag=tag,
            gt_mm=row["gt_mm"],  # type: ignore[arg-type]
            pr_mm=row["pred_mm"],  # type: ignore[arg-type]
            r=float(row["r"]),
            r2=float(row["r2"]),
            rmse=float(row["rmse"]),
            std_mm=row.get("std_mm", None),  # type: ignore[arg-type]
        )

    if "test" not in results:
        return

    gt_mm = results["test"]["gt_mm"]  # type: ignore[assignment]
    pr_mm = results["test"]["pred_mm"]  # type: ignore[assignment]
    std_mm = results["test"].get("std_mm", None)  # type: ignore[assignment]
    std_ale_mm = results["test"].get("std_ale_mm", None)  # type: ignore[assignment]
    std_epi_mm = results["test"].get("std_epi_mm", None)  # type: ignore[assignment]
    r = float(results["test"]["r"])
    r2 = float(results["test"]["r2"])
    rmse = float(results["test"]["rmse"])
    samples_gt_mm = results["test"].get("samples_gt_mm", None)  # type: ignore[assignment]
    samples_pr_mm = results["test"].get("samples_pr_mm", None)  # type: ignore[assignment]

    # Plot
    if cfg.save_plot:
        plt.figure(figsize=(7,6))
        use_std = (std_mm is not None) and (cfg.plot_unc.lower() != 'none')
        if use_std and cfg.plot_unc.lower() in ('color','both'):
            sc = plt.scatter(gt_mm, pr_mm, s=18, alpha=0.75, c=std_mm)
            cb = plt.colorbar(sc); cb.set_label('Predictive std (mm)')
        else:
            plt.scatter(gt_mm, pr_mm, s=18, alpha=0.75)
        if use_std and cfg.plot_unc.lower() in ('errorbar','both'):
            yerr = 1.96 * std_mm
            plt.errorbar(gt_mm, pr_mm, yerr=yerr, fmt='none', ecolor='k', alpha=0.15, lw=0.6)
        if cfg.plot_raw_samples and (samples_gt_mm is not None) and (samples_pr_mm is not None):
            jitter = (np.random.rand(samples_gt_mm.size) - 0.5) * 2.0 * cfg.samples_jitter
            plt.scatter(samples_gt_mm + jitter, samples_pr_mm, alpha=cfg.samples_alpha, s=6, lw=0)
        mn, mx = float(min(gt_mm.min(), pr_mm.min())), float(max(gt_mm.max(), pr_mm.max()))
        # fix axes to the expected 1–4 mm range for readability
        try:
            plt.xlim(1.0, 4.0)
            plt.ylim(1.0, 4.0)
        except Exception:
            pass
        plt.plot([mn, mx], [mn, mx], '--')
        txt = rf"$r={r:.3f}\quad R^2={r2:.3f}\quad \mathrm{{RMSE}}={rmse:.3f}\,\mathrm{{mm}}$"
        plt.text(0.05, 0.95, txt, transform=plt.gca().transAxes, va='top', ha='left', bbox=dict(boxstyle='round', pad=0.3, alpha=0.2))
        plt.xlabel('Ground Truth Depth (mm)')
        plt.ylabel('Predicted Depth (mm)')
        plt.title('Depth Regression: GT vs Prediction (with uncertainty)')
        plt.tight_layout(); plt.savefig(cfg.save_plot, dpi=200)
        print(f"Saved plot to {cfg.save_plot}")

    # CSVs
    if cfg.save_stochastic:
        import csv
        std_values = std_mm if std_mm is not None else np.full_like(gt_mm, np.nan, dtype=float)
        with open(cfg.save_stochastic, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['gt_mm','pred_mm','std_mm','std_ale_mm','std_epi_mm','ci95_minus','ci95_plus'])
            std_ale_list = std_ale_mm.tolist() if std_ale_mm is not None else [None] * len(gt_mm)
            std_epi_list = std_epi_mm.tolist() if std_epi_mm is not None else [None] * len(gt_mm)
            for g, p, s, sa, se in zip(gt_mm.tolist(), pr_mm.tolist(), std_values.tolist(), std_ale_list, std_epi_list):
                w.writerow([g, p, s, sa, se, p - 1.96*s, p + 1.96*s])
        print(f"Saved test predictions to {cfg.save_stochastic}")
    if cfg.save_stochastic_raw and (samples_gt_mm is not None) and (samples_pr_mm is not None):
        import csv
        with open(cfg.save_stochastic_raw, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['gt_mm','pred_mm'])
            for g, p in zip(samples_gt_mm.tolist(), samples_pr_mm.tolist()):
                w.writerow([g, p])
        print(f"Saved raw stochastic samples to {cfg.save_stochastic_raw}")



# ----------------------------------------------------------------------------
# Config-driven predictor (robust to older checkpoints missing 'norm'/'splits')
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict_depth_from_csv_cfg(
    cfg: TrainConfig,
    csv_path: str,
    csv_dt_ms: float,
    *,
    auto_align: bool = True,
    aligned_csv_out: Optional[str] = None,
    aligned_plot_out: Optional[str] = None,
    window_plot_out: Optional[str] = None,
    target_len: int = 100,
    model_dt_ms: float = 0.1,
    anchor_idx_1based: int = 10,
    smooth_ms: float = 0.2,
) -> dict:
    """
    Predict slow-conduction depth using TrainConfig (no PredictOptions needed).
    Falls back to dataset-derived normalization and input length if the checkpoint
    lacks 'norm' and/or 'splits'.

    Returns dict: {'pred_mm', 'std_mm', 'ci95', 'used_csv'}.
    """
    set_global_seed(cfg.seed)
    device = torch.device(cfg.device)

    if not os.path.exists(cfg.checkpoint):
        raise FileNotFoundError(cfg.checkpoint)
    ckpt = torch.load(cfg.checkpoint, map_location=device)

    # -------------- Normalization (gmin/gmax) --------------
    if 'norm' in ckpt and ('gmin' in ckpt['norm']) and ('gmax' in ckpt['norm']):
        gmin = float(ckpt['norm']['gmin']); gmax = float(ckpt['norm']['gmax'])
    else:
        # compute from prepared dataset according to cfg.quantile_norm
        prepared = load_prepared_dataset(cfg.data_dir)
        all_idx = list(range(int(prepared.signals.shape[0])))
        if not all_idx:
            raise RuntimeError("No training samples found to compute normalization and checkpoint lacks 'norm'.")
        if cfg.quantile_norm:
            gmin, gmax = compute_quantile_minmax(prepared.signals, all_idx, cfg.q_low, cfg.q_high)
        else:
            gmin, gmax = compute_global_minmax(prepared.signals, all_idx)
        gmin, gmax = float(gmin), float(gmax)

    # -------------- Label normalization --------------------
    # If missing, assume model outputs are already normalized in [0,1].
    if 'label_norm' in ckpt and isinstance(ckpt['label_norm'], dict):
        y_mean = float(ckpt['label_norm'].get('mean', 0.0))
        y_std  = float(max(ckpt['label_norm'].get('std', 1.0), 1e-6))
    else:
        y_mean, y_std = 0.0, 1.0

    # -------------- Model ----------------------------------
    def _extract_state_dict(obj):
        if isinstance(obj, torch.nn.Module):
            return obj.state_dict()
        if isinstance(obj, dict):
            for k in ('model_state', 'state_dict', 'model', 'net', 'weights', 'params'):
                v = obj.get(k)
                if isinstance(v, dict):
                    return v
                if isinstance(v, torch.nn.Module):
                    return v.state_dict()
            # flat state dict
            if obj and all(isinstance(k, str) for k in obj.keys()) and any(isinstance(v, torch.Tensor) for v in obj.values()):
                return obj
        raise KeyError("No state_dict found in checkpoint.")

    def _strip_module_prefix(sd: dict) -> dict:
        if any(k.startswith('module.') for k in sd.keys()):
            return {k.replace('module.', '', 1): v for k, v in sd.items()}
        return sd

    def _infer_arch(sd: dict, cfg_fallback: TrainConfig):
        keys = list(sd.keys())
        hetero = None
        if any(k.startswith('fc_mu') or k.startswith('fc_logvar') for k in keys):
            hetero = True
        elif any(k.startswith('fc.') for k in keys):
            hetero = False
        else:
            hetero = bool(getattr(cfg_fallback, 'heteroscedastic', True))
        attn_pool = True if any(k.startswith('pool_attn.') for k in keys) else bool(getattr(cfg_fallback, 'attn_pool', True))
        return bool(attn_pool), bool(hetero)

    sd_raw = _extract_state_dict(ckpt)
    sd = _strip_module_prefix(sd_raw)
    attn_pool, hetero = _infer_arch(sd, cfg)
    model = DepthCNN(attn_pool=attn_pool, hetero=hetero).to(device)
    try:
        model.load_state_dict(sd, strict=True)
    except Exception:
        # last resort: try alternate hetero flag, else non-strict
        try:
            model = DepthCNN(attn_pool=attn_pool, hetero=not hetero).to(device)
            model.load_state_dict(sd, strict=True)
        except Exception:
            model = DepthCNN(attn_pool=attn_pool, hetero=hetero).to(device)
            model.load_state_dict(sd, strict=False)
    model.eval()

    # -------------- Input length inference -----------------
    target_len_effective = int(target_len)
    try:
        prepared = load_prepared_dataset(cfg.data_dir)
        target_len_effective = int(prepared.signals.shape[1])
    except Exception:
        pass

    # -------------- Load/align CSV -------------------------
    used_csv = csv_path
    if auto_align:
        from ap_align import align_ap_csv, load_csv_signal_singlecol
        used_csv = align_ap_csv(
            csv_in=csv_path,
            dt_ms=float(csv_dt_ms),
            csv_out=aligned_csv_out,
            plot_out=aligned_plot_out,
            target_len=int(target_len_effective),
            model_dt_ms=float(model_dt_ms),
            peak_index_1based=int(anchor_idx_1based),
            smooth_ms=float(smooth_ms),
        )
        x_raw = load_csv_signal_singlecol(used_csv)
    else:
        from ap_align import load_csv_signal_singlecol, resample_uniform_linear, _safe_window  # type: ignore
        x_full = load_csv_signal_singlecol(csv_path)
        if abs(float(csv_dt_ms) - float(model_dt_ms)) > 1e-12:
            xr = resample_uniform_linear(x_full, float(csv_dt_ms), float(model_dt_ms))
        else:
            xr = x_full
        if xr.size != target_len_effective:
            start = max(0, (xr.size - target_len_effective) // 2)
            x_raw = _safe_window(xr, start, target_len_effective)  # type: ignore
        else:
            x_raw = xr

    # -------------- Normalize + build 3 channels ------------
    x_clipped = np.clip(x_raw, gmin, gmax)
    x_norm = (x_clipped - gmin) / max(gmax - gmin, 1e-12)
    raw = torch.from_numpy(x_norm.astype(np.float32))
    diff = torch.cat([raw[:1], raw[1:] - raw[:-1]])
    logp = torch.log1p(torch.clamp(raw, min=0.0))
    x_t = torch.stack([raw, diff, logp], dim=0).unsqueeze(0).to(device=device)
    channel_mask = (ckpt.get("config", {}) or {}).get("channel_mask", getattr(cfg, "channel_mask", None))
    x_t = _apply_channel_mask_tensor(x_t, channel_mask)

    # -------------- Forward (with/without uncertainty) -----
    if hetero:
        out = model(x_t)
        if isinstance(out, tuple) and len(out) == 2:
            mu_z, logvar_z = out
            mu = float(mu_z.item())
            std = float(torch.exp(0.5 * logvar_z).item())
        else:
            mu = float(out.item())
            std = None
    else:
        out = model(x_t)
        mu = float(out.item())
        std = None

    # -------------- Map to mm ------------------------------
    mu_norm = float(np.clip(mu * y_std + y_mean, 0.0, 1.0))
    # Use label range saved in the checkpoint if available
    lr = ckpt.get('label_range', {}) if isinstance(ckpt, dict) else {}
    if lr:
        dmin = int(lr.get('min_idx', 20))
        dmax = int(lr.get('max_idx', 80))
        voxel_mm = float(lr.get('voxel_mm', 0.05))
    else:
        dmin, dmax, voxel_mm = 10, 80, 0.05
    pred_mm = float((_denorm_depth(mu_norm, depth_range=(dmin, dmax)) * voxel_mm))
    std_mm = None
    ci95 = None
    if std is not None:
        std_norm = std * y_std
        std_mm = float(std_norm * (dmax - dmin) * voxel_mm)
        ci95 = (pred_mm - 1.96 * std_mm, pred_mm + 1.96 * std_mm)

    # -------------- Optional plot --------------------------
    if window_plot_out:
        import matplotlib.pyplot as plt
        t = np.arange(x_raw.size, dtype=np.float64) * float(model_dt_ms)
        plt.figure(figsize=(7.5, 3.6))
        plt.plot(t, x_raw, lw=1.1)
        anchor_t = (anchor_idx_1based - 1) * model_dt_ms
        if len(t) > 0 and 0 <= anchor_t <= t[-1]:
            plt.axvline(anchor_t, ls='--', lw=0.9, alpha=0.7)
        title = f"Window to model (L={x_raw.size}, dt={model_dt_ms} ms) | Pred depth = {pred_mm:.2f} mm"
        if ci95: title += f" | 95% CI [{ci95[0]:.2f}, {ci95[1]:.2f}] mm"
        plt.title(title)
        plt.xlabel("Time (ms)"); plt.ylabel("Signal (arb.)")
        plt.tight_layout(); plt.savefig(window_plot_out, dpi=170)

    return {"pred_mm": pred_mm, "std_mm": std_mm, "ci95": ci95, "used_csv": used_csv}

# ----------------------------------------------------------------------------
# Canonical minimal API (no TrainConfig): predict_depth_from_csv(...)
# ----------------------------------------------------------------------------

# ============================================================================
# Minimal, TrainConfig-free prediction helper (re-added)
# ============================================================================
from dataclasses import dataclass

@dataclass
class PredictOptions:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mc_samples: int = 50
    tta_samples: int = 1
    tta_noise: float = 0.0
    tta_max_shift: int = 0
    tta_amp_min: float = 1.0
    tta_amp_max: float = 1.0

def _infer_model_length_from_dataset(data_dir: str) -> int:
    prepared = load_prepared_dataset(data_dir)
    return int(prepared.signals.shape[1])

@torch.no_grad()
def predict_depth_from_csv_ckpt(
    checkpoint: str,
    csv_path: str,
    csv_dt_ms: float,
    *,
    auto_align: bool = True,
    aligned_csv_out: Optional[str] = None,
    aligned_plot_out: Optional[str] = None,
    window_plot_out: Optional[str] = None,
    target_len: int = 100,
    model_dt_ms: float = 0.1,
    anchor_idx_1based: int = 10,
    smooth_ms: float = 0.2,
    opts: Optional[PredictOptions] = None,
) -> dict:
    """
    Predict slow-conduction depth from a CSV using a saved checkpoint.

    Best-practice behavior:
    - Loads model and label normalization from checkpoint when present.
    - Robust to checkpoints missing 'norm' and 'splits':
      falls back to dataset-derived normalization (if data_dir available),
      else per-signal quantile min/max from the provided CSV.
    - Optionally aligns/resamples the input to the model grid.

    Returns dict with keys:
      - pred_mm (float), std_mm (Optional[float]), ci95 (Optional[Tuple[float,float]]), used_csv (str)
    """
    if opts is None:
        opts = PredictOptions()

    device = torch.device(opts.device)
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(checkpoint)
    ckpt = torch.load(checkpoint, map_location=device)

    # Training-time info
    cfg_saved = ckpt.get("config", {}) or {}

    # 1) Normalization for input (gmin/gmax)
    gmin = gmax = None
    if isinstance(ckpt.get("norm"), dict) and "gmin" in ckpt["norm"] and "gmax" in ckpt["norm"]:
        gmin = float(ckpt["norm"]["gmin"])  # preferred path
        gmax = float(ckpt["norm"]["gmax"])  # preferred path
    else:
        # Fallback A: compute from training dataset (if we know where it is)
        data_dir = cfg_saved.get("data_dir")
        use_quant = bool(cfg_saved.get("quantile_norm", True))
        q_low = float(cfg_saved.get("q_low", 0.001))
        q_high = float(cfg_saved.get("q_high", 0.999))
        if isinstance(data_dir, str) and os.path.isdir(data_dir):
            prepared = load_prepared_dataset(data_dir)
            all_idx = list(range(int(prepared.signals.shape[0])))
            if all_idx:
                if use_quant:
                    gmin, gmax = compute_quantile_minmax(prepared.signals, all_idx, q_low, q_high)
                else:
                    gmin, gmax = compute_global_minmax(prepared.signals, all_idx)
                gmin, gmax = float(gmin), float(gmax)
        # Fallback B: per-signal quantile range from the provided CSV
        if gmin is None or gmax is None:
            from ap_align import load_csv_signal_singlecol  # lightweight read
            x_full = load_csv_signal_singlecol(csv_path)
            if x_full.size == 0:
                raise RuntimeError("Cannot infer normalization: empty CSV and checkpoint lacks 'norm'.")
            lo = float(np.quantile(x_full, q_low))
            hi = float(np.quantile(x_full, q_high))
            # Avoid degenerate case
            if abs(hi - lo) < 1e-9:
                lo = float(x_full.min())
                hi = float(x_full.max())
            gmin, gmax = lo, hi

    # 2) Label normalization for mapping to [0,1]
    # If missing, assume outputs already in [0,1].
    if 'label_norm' in ckpt and isinstance(ckpt['label_norm'], dict):
        y_mean = float(ckpt['label_norm'].get('mean', 0.0))
        y_std  = float(max(ckpt['label_norm'].get('std', 1.0), 1e-6))
    else:
        y_mean, y_std = 0.0, 1.0

    # 3) Model definition and weights (robust extraction)
    def _extract_state_dict(obj):
        if isinstance(obj, torch.nn.Module):
            return obj.state_dict()
        if isinstance(obj, dict):
            for k in ('model_state', 'state_dict', 'model', 'net', 'weights', 'params'):
                v = obj.get(k)
                if isinstance(v, dict):
                    return v
                if isinstance(v, torch.nn.Module):
                    return v.state_dict()
            if obj and all(isinstance(k, str) for k in obj.keys()) and any(isinstance(v, torch.Tensor) for v in obj.values()):
                return obj
        raise KeyError("No state_dict found in checkpoint.")

    def _strip_module_prefix(sd: dict) -> dict:
        if any(k.startswith('module.') for k in sd.keys()):
            return {k.replace('module.', '', 1): v for k, v in sd.items()}
        return sd

    def _infer_arch(sd: dict, cfg_fallback: dict):
        keys = list(sd.keys())
        hetero = None
        if any(k.startswith('fc_mu') or k.startswith('fc_logvar') for k in keys):
            hetero = True
        elif any(k.startswith('fc.') for k in keys):
            hetero = False
        else:
            hetero = bool(cfg_fallback.get('heteroscedastic', True))
        attn_pool = True if any(k.startswith('pool_attn.') for k in keys) else bool(cfg_fallback.get('attn_pool', True))
        return bool(attn_pool), bool(hetero)

    sd_raw = _extract_state_dict(ckpt)
    sd = _strip_module_prefix(sd_raw)
    attn_pool, hetero = _infer_arch(sd, cfg_saved)
    model = DepthCNN(attn_pool=attn_pool, hetero=hetero).to(device)
    try:
        model.load_state_dict(sd, strict=True)
    except Exception:
        try:
            model = DepthCNN(attn_pool=attn_pool, hetero=not hetero).to(device)
            model.load_state_dict(sd, strict=True)
        except Exception:
            model = DepthCNN(attn_pool=attn_pool, hetero=hetero).to(device)
            model.load_state_dict(sd, strict=False)
    model.eval()

    # 4) Expected input length (optional inference from training dataset)
    data_dir = cfg_saved.get("data_dir")
    if isinstance(data_dir, str) and os.path.isdir(data_dir):
        try:
            target_len = int(_infer_model_length_from_dataset(data_dir))
        except Exception:
            pass

    # 5) Load/align the CSV to the model grid
    used_csv = csv_path
    if auto_align:
        from ap_align import align_ap_csv, load_csv_signal_singlecol
        used_csv = align_ap_csv(
            csv_in=csv_path, dt_ms=float(csv_dt_ms),
            csv_out=aligned_csv_out, plot_out=aligned_plot_out,
            target_len=int(target_len), model_dt_ms=float(model_dt_ms),
            peak_index_1based=int(anchor_idx_1based), smooth_ms=float(smooth_ms),
        )
        x_raw = load_csv_signal_singlecol(used_csv)
    else:
        from ap_align import load_csv_signal_singlecol, resample_uniform_linear, _safe_window  # type: ignore
        x_full = load_csv_signal_singlecol(csv_path)
        if abs(float(csv_dt_ms) - float(model_dt_ms)) > 1e-12:
            xr = resample_uniform_linear(x_full, float(csv_dt_ms), float(model_dt_ms))
        else:
            xr = x_full
        if xr.size != target_len:
            start = max(0, (xr.size - target_len) // 2)
            x_raw = _safe_window(xr, start, target_len)  # type: ignore
        else:
            x_raw = xr

    # 6) Normalize + build 3-channel input
    x_clipped = np.clip(x_raw, gmin, gmax)
    x_norm = (x_clipped - gmin) / max(gmax - gmin, 1e-12)
    raw = torch.from_numpy(x_norm.astype(np.float32))
    diff = torch.cat([raw[:1], raw[1:] - raw[:-1]])
    logp = torch.log1p(torch.clamp(raw, min=0.0))
    x_t = torch.stack([raw, diff, logp], dim=0).unsqueeze(0).to(device=device)
    x_t = _apply_channel_mask_tensor(x_t, cfg_saved.get("channel_mask"))

    # 7) Inference with optional MC Dropout + TTA
    def _enable_dropout_only(m: nn.Module):
        for mod in m.modules():
            if isinstance(mod, nn.Dropout):
                mod.train()

    def _tta_perturb(x: torch.Tensor) -> torch.Tensor:
        if opts.tta_samples <= 1:
            return x
        with torch.no_grad():
            xb = x.clone()
            if opts.tta_max_shift:
                shift = np.random.randint(-opts.tta_max_shift, opts.tta_max_shift + 1)
                xb = torch.roll(xb, shifts=int(shift), dims=-1)
            if (opts.tta_amp_min != 1.0) or (opts.tta_amp_max != 1.0):
                scale = float(np.random.uniform(opts.tta_amp_min, opts.tta_amp_max))
                xb = xb * scale
            if opts.tta_noise > 0:
                xb = xb + torch.randn_like(xb) * opts.tta_noise
            return xb

    mcK = max(int(opts.mc_samples), 1)
    ttaK = max(int(opts.tta_samples), 1)
    if mcK == 1 and ttaK == 1:
        out = model(x_t)
        if hetero:
            mu_z, logvar_z = out
            mu = float(mu_z.item())
            std = float(torch.exp(0.5 * logvar_z).item())
        else:
            mu = float(out.item())
            std = None
    else:
        mus, vars_ = [], []
        for _ in range(mcK):
            model.eval()
            if mcK > 1:
                _enable_dropout_only(model)
            for _ in range(ttaK):
                xt = _tta_perturb(x_t)
                out = model(xt)
                if hetero:
                    mu_z, logvar_z = out
                    mus.append(float(mu_z.item()))
                    vars_.append(float(torch.exp(logvar_z).item()))
                else:
                    mus.append(float(out.item()))
        m_arr = np.array(mus, dtype=np.float64)
        mu = float(m_arr.mean())
        var_ep = float(m_arr.var())
        if hetero and len(vars_) > 0:
            ale = float(np.mean(np.array(vars_, dtype=np.float64)))
            std = float((ale + var_ep) ** 0.5)
        else:
            std = float(max(var_ep, 0.0) ** 0.5)

    # 8) Map back to mm
    mu_norm = float(np.clip(mu * y_std + y_mean, 0.0, 1.0))
    # Map back to mm using the label range discovered from the checkpoint
    lr = ckpt.get('label_range', {}) if isinstance(ckpt, dict) else {}
    if lr:
        dmin = int(lr.get('min_idx', 20))
        dmax = int(lr.get('max_idx', 80))
        voxel_mm = float(lr.get('voxel_mm', 0.05))
    else:
        dmin, dmax, voxel_mm = 10, 80, 0.05
    pred_mm = float((_denorm_depth(mu_norm, depth_range=(dmin, dmax)) * voxel_mm))
    std_mm = None
    ci95 = None
    if std is not None:
        std_norm = std * y_std
        std_mm = float(std_norm * (dmax - dmin) * voxel_mm)
        ci95 = (pred_mm - 1.96 * std_mm, pred_mm + 1.96 * std_mm)

    # 9) Optional plot of the window sent to the model
    if window_plot_out:
        import matplotlib.pyplot as plt
        t = np.arange(x_raw.size, dtype=np.float64) * float(model_dt_ms)
        plt.figure(figsize=(7.5, 3.6))
        plt.plot(t, x_raw, lw=1.1)
        anchor_t = (anchor_idx_1based - 1) * model_dt_ms
        if len(t) > 0 and 0 <= anchor_t <= t[-1]:
            plt.axvline(anchor_t, ls='--', lw=0.9, alpha=0.7)
        plt.title(
            f"Window to model (L={x_raw.size}, dt={model_dt_ms} ms)\n"
            f"Pred depth = {pred_mm:.2f} mm" + (f" | 95% CI [{ci95[0]:.2f}, {ci95[1]:.2f}] mm" if ci95 else "")
        )
        plt.xlabel("Time (ms)"); plt.ylabel("Signal (arb.)")
        plt.tight_layout(); plt.savefig(window_plot_out, dpi=170)

    return {"pred_mm": pred_mm, "std_mm": std_mm, "ci95": ci95, "used_csv": used_csv}

def predict_depth_from_csv(
    checkpoint: str,
    csv_path: str,
    csv_dt_ms: float,
    *,
    auto_align: bool = True,
    aligned_csv_out: Optional[str] = None,
    aligned_plot_out: Optional[str] = None,
    window_plot_out: Optional[str] = None,
    target_len: int = 100,
    model_dt_ms: float = 0.1,
    anchor_idx_1based: int = 10,
    smooth_ms: float = 0.2,
    opts: Optional[PredictOptions] = None,
) -> dict:
    """
    Predict slow-conduction depth from a CSV without TrainConfig.

    Parameters mirror predict_depth_from_csv_ckpt(...), but with a simpler name.
    """
    return predict_depth_from_csv_ckpt(
        checkpoint=checkpoint,
        csv_path=csv_path,
        csv_dt_ms=csv_dt_ms,
        auto_align=auto_align,
        aligned_csv_out=aligned_csv_out,
        aligned_plot_out=aligned_plot_out,
        window_plot_out=window_plot_out,
        target_len=target_len,
        model_dt_ms=model_dt_ms,
        anchor_idx_1based=anchor_idx_1based,
        smooth_ms=smooth_ms,
        opts=opts,
    )


# --------------------------
# CLI
# --------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train/Eval DepthCNN (1D)')
    p.add_argument('--config', type=str, required=True, help='YAML config path')
    m = p.add_mutually_exclusive_group(required=True)
    m.add_argument('--train', action='store_true')
    m.add_argument('--eval', action='store_true')
    p.add_argument(
        "--eval-split",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
        help="When using --eval: which split(s) to evaluate (default: test).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, 'r') as f:
        cfg_dict = yaml.safe_load(f) or {}

    env_data_dir = os.environ.get("LMT_NN_DATA_DIR")
    if env_data_dir:
        cfg_dict["data_dir"] = env_data_dir
    env_checkpoint = os.environ.get("LMT_NN_CHECKPOINT")
    if env_checkpoint:
        cfg_dict["checkpoint"] = env_checkpoint
    env_depth_min_idx = os.environ.get("LMT_NN_DEPTH_MIN_IDX")
    if env_depth_min_idx:
        cfg_dict["depth_min_idx"] = int(env_depth_min_idx)
    env_depth_max_idx = os.environ.get("LMT_NN_DEPTH_MAX_IDX")
    if env_depth_max_idx:
        cfg_dict["depth_max_idx"] = int(env_depth_max_idx)
    env_voxel_mm = os.environ.get("LMT_NN_VOXEL_MM")
    if env_voxel_mm:
        cfg_dict["voxel_mm"] = float(env_voxel_mm)
    env_channel_mask = os.environ.get("LMT_NN_CHANNEL_MASK")
    if env_channel_mask:
        cfg_dict["channel_mask"] = [float(v.strip()) for v in env_channel_mask.split(",") if v.strip()]
    env_out_dir = os.environ.get("LMT_NN_OUT_DIR")
    if env_out_dir:
        out_dir = Path(env_out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for k in ("save_plot", "save_stochastic", "save_stochastic_raw"):
            if k in cfg_dict and cfg_dict[k]:
                p = Path(str(cfg_dict[k]))
                if not p.is_absolute():
                    cfg_dict[k] = str(out_dir / p.name)
    cfg = TrainConfig(**cfg_dict)

    # Use point-estimate regression when uncertainty is disabled.
    if not bool(getattr(cfg, "enable_uncertainty", True)):
        cfg.heteroscedastic = False
        cfg.tie_logvar_to_mu = False
        cfg.weight_logvar_term = False
        cfg.monotonic_uncertainty = False
        cfg.mc_samples = 1
        cfg.tta_samples = 1
        cfg.plot_unc = "none"
        cfg.calibrate_std_on_val = False
        cfg.calibrate_std_by_depth = False
        cfg.calibrate_std_monotonic = False
        cfg.save_stochastic_raw = None

    if hasattr(cfg, 'data_dir') and (not os.path.isdir(cfg.data_dir)):
        raise FileNotFoundError(cfg.data_dir)
    if 'device' in cfg_dict:
        torch.cuda.is_available()  # trigger CUDA init for clarity in logs
    if args.train:
        train_and_save(cfg)
    else:
        evaluate(cfg, eval_split=str(args.eval_split))
    
    # result = predict_depth_from_csv(
    # checkpoint="best_depthcnn.pth",      # path to your trained model
    # csv_path="experiment_raw.csv",       # single-column AP CSV
    # csv_dt_ms=0.2,                       # known sampling interval (ms)
    # auto_align=True,                     # align to 0.1 ms grid, 100 samples, dV/dt max at sample #10
    # aligned_csv_out="experiment_aligned.csv",       # (optional) save aligned CSV
    # aligned_plot_out="experiment_alignment.png",    # (optional) plot original vs aligned
    # window_plot_out="exp_window_to_model.png",      # (optional) plot actually fed window
    # target_len=100,
    # model_dt_ms=0.1,
    # anchor_idx_1based=10,
    # smooth_ms=0.2,
    # opts=PredictOptions(device="cuda", mc_samples=50, tta_samples=1),
    # )

    # print("Predicted depth (mm):", result["pred_mm"])
    # print("Std (mm):", result["std_mm"])
    # print("95% CI (mm):", result["ci95"])
    # print("CSV used:", result["used_csv"])


if __name__ == '__main__':
    main()
