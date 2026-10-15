"""
Core components for 1D depth regression.
Includes: seeding helpers, augmentation, dataset, min/max utilities,
model blocks (ResBlock1D), attention pooling, and DepthCNN (toggleable
heteroscedastic head).

This module is imported by the training/eval CLI script.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# --------------------------
# Reproducibility utilities
# --------------------------

def set_global_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)


# --------------------------
# Data pipeline
# --------------------------

@dataclass(frozen=True)
class PreparedEmission1DDataset:
    signals: np.ndarray  # (N, L) float32
    depth: np.ndarray    # (N,) int/float
    thickness: np.ndarray  # (N,) int
    time_indices: Optional[np.ndarray]  # (L,) int or None


def load_prepared_dataset(data_dir: str) -> PreparedEmission1DDataset:
    """
    Load the dataset produced by `scripts/prepare_nn_dataset.py`.

    Expected files under `data_dir`:
      - signals.npy (N,L) float32
      - targets_depth.npy (N,) int32
      - targets_thickness.npy (N,) int32
      - time_indices.npy (optional, L) int32
    """
    base = os.path.abspath(data_dir)
    signals_path = os.path.join(base, "signals.npy")
    depth_path = os.path.join(base, "targets_depth.npy")
    thickness_path = os.path.join(base, "targets_thickness.npy")
    time_indices_path = os.path.join(base, "time_indices.npy")

    if not os.path.isfile(signals_path):
        raise FileNotFoundError(signals_path)
    if not os.path.isfile(depth_path):
        raise FileNotFoundError(depth_path)
    if not os.path.isfile(thickness_path):
        raise FileNotFoundError(thickness_path)

    signals = np.load(signals_path).astype(np.float32, copy=False)
    depth = np.load(depth_path)
    thickness = np.load(thickness_path)
    time_indices = np.load(time_indices_path) if os.path.isfile(time_indices_path) else None

    if signals.ndim != 2:
        raise ValueError(f"Expected signals.npy as (N,L), got shape={signals.shape}")
    if depth.shape[0] != signals.shape[0] or thickness.shape[0] != signals.shape[0]:
        raise ValueError("signals/targets length mismatch")

    return PreparedEmission1DDataset(
        signals=signals,
        depth=depth,
        thickness=thickness,
        time_indices=time_indices,
    )


class RandomAugment:
    """Stochastic augmentation for 1D signals (0..1 scale).
    - amplitude jitter (multiplicative)
    - small circular time shift
    - additive Gaussian noise
    """
    def __init__(self,
                 p: float = 0.5,
                 noise_std_range: Tuple[float, float] = (0.0, 0.005),
                 max_shift: int = 2,
                 amp_jitter: Tuple[float, float] = (0.98, 1.02),
                 clamp: bool = True) -> None:
        assert 0.0 <= p <= 1.0
        self.p = p
        self.noise_std_range = noise_std_range
        self.max_shift = max_shift
        self.amp_jitter = amp_jitter
        self.clamp = clamp

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) > self.p:
            return signal
        scale = float(np.random.uniform(*self.amp_jitter))
        signal = signal * scale
        if self.max_shift:
            shift = np.random.randint(-self.max_shift, self.max_shift + 1)
            signal = torch.roll(signal, shifts=shift, dims=-1)
        std = float(np.random.uniform(*self.noise_std_range))
        if std > 0:
            signal = signal + torch.randn_like(signal) * std
        if self.clamp:
            signal = torch.clamp(signal, 0.0, 1.0)
        return signal


class EmissionDepthDataset(Dataset):
    """Loads prepared signals, normalizes with provided min/max, optional augment.
    Returns (3,L) features: raw, first-diff, log1p; and normalized depth y∈[0,1].
    """
    def __init__(self,
                 signals: np.ndarray,
                 depths: np.ndarray,
                 indices: List[int],
                 transform: Optional[RandomAugment] = None,
                 normalize: bool = True,
                 global_min: Optional[float] = None,
                 global_max: Optional[float] = None,
                 depth_range: Tuple[int, int] = (20, 80),
                 channel_mask: Optional[Tuple[float, float, float]] = None) -> None:
        self.signals = signals
        self.depths = depths
        self.indices = indices
        self.transform = transform
        self.normalize = normalize
        self.min_depth, self.max_depth = depth_range
        self.global_min = global_min
        self.global_max = global_max
        self.use_global = normalize and (global_min is not None) and (global_max is not None)
        if channel_mask is None:
            self.channel_mask = (1.0, 1.0, 1.0)
        else:
            mask = tuple(float(v) for v in channel_mask)
            if len(mask) != 3:
                raise ValueError(f"channel_mask must have length 3, got {channel_mask!r}")
            self.channel_mask = mask

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        i = int(self.indices[idx])
        arr = self.signals[i].astype(np.float32, copy=False)  # (L,)
        sig = torch.from_numpy(arr).unsqueeze(0)  # (1,L)
        if self.normalize:
            if self.use_global:
                denom = (self.global_max - self.global_min) + 1e-6
                sig = (sig - self.global_min) / denom
            else:
                mn = sig.min(-1, keepdim=True)[0]
                mx = sig.max(-1, keepdim=True)[0]
                sig = (sig - mn) / (mx - mn + 1e-6)
        if self.transform is not None:
            sig = self.transform(sig)
        raw = sig[0]
        diff = torch.cat([raw[:1], raw[1:] - raw[:-1]])
        logp = torch.log1p(torch.clamp(raw, min=0.0))
        x = torch.stack([raw, diff, logp], dim=0)  # (3,L)
        if self.channel_mask != (1.0, 1.0, 1.0):
            mask = torch.tensor(self.channel_mask, dtype=x.dtype, device=x.device).view(3, 1)
            x = x * mask
        n = float(self.depths[i])
        n = max(self.min_depth, min(float(n), self.max_depth))
        y = torch.tensor((n - self.min_depth) / (self.max_depth - self.min_depth), dtype=torch.float32)
        sample_id = f"idx{i}"
        return x, y, sample_id


# --------------------------
# Min/Max utilities
# --------------------------

def compute_global_minmax(signals: np.ndarray, indices: List[int]) -> Tuple[float, float]:
    sel = signals[np.asarray(indices, dtype=np.int64)]
    all_vals = sel.reshape(-1)
    gmin, gmax = float(all_vals.min()), float(all_vals.max())
    if not np.isfinite(gmin) or not np.isfinite(gmax):
        raise ValueError("Non-finite min/max.")
    return gmin, gmax


def compute_quantile_minmax(signals: np.ndarray, indices: List[int], q_low: float, q_high: float) -> Tuple[float, float]:
    sel = signals[np.asarray(indices, dtype=np.int64)]
    all_vals = sel.reshape(-1)
    gmin, gmax = np.quantile(all_vals, [q_low, q_high]).astype(float)
    if not np.isfinite(gmin) or not np.isfinite(gmax) or gmax <= gmin:
        raise ValueError("Invalid quantile min/max.")
    return gmin, gmax


# --------------------------
# Model
# --------------------------

class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1) -> None:
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.need_proj = in_ch != out_ch
        if self.need_proj:
            self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.need_proj:
            identity = self.proj(identity)
        return F.relu(x + identity)


class AttentionPool1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(16, channels // 2)
        self.attn = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.attn(x)            # (B,1,L)
        w = torch.softmax(logits, dim=-1)
        return (x * w).sum(dim=-1)       # (B,C)


class DepthCNN(nn.Module):
    """ResNet(1D) backbone + attention (or GAP) + linear heads.
    If hetero=True, returns (mu, logvar). Optional tying of log-variance to
    the predicted mean to encourage monotonic uncertainty with depth.
    """
    def __init__(self, attn_pool: bool = True, hetero: bool = False,
                 tie_logvar_to_mu: bool = False) -> None:
        super().__init__()
        self.hetero = hetero
        self.tie_logvar_to_mu = tie_logvar_to_mu
        self.rb1 = ResBlock1D(3, 32, dilation=1)
        self.rb2 = ResBlock1D(32, 64, dilation=2)
        self.rb3 = ResBlock1D(64, 64, dilation=4)
        self.rb4 = ResBlock1D(64, 64, dilation=8)
        self.pool_attn = AttentionPool1D(64) if attn_pool else None
        self.pool_gap = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(0.3)
        if hetero:
            self.fc_mu = nn.Linear(64, 1)
            self.fc_logvar = nn.Linear(64, 1)
            if tie_logvar_to_mu:
                # Non-negative coefficient multiplied by detached mean
                self.logvar_mu_coeff = nn.Parameter(torch.tensor(0.0))
        else:
            self.fc = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor):
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        if self.pool_attn is not None:
            x = self.pool_attn(x)
        else:
            x = self.pool_gap(x).squeeze(-1)
        x = self.drop(x)
        if self.hetero:
            mu = self.fc_mu(x).squeeze(-1)
            if self.tie_logvar_to_mu:
                extra = F.softplus(self.logvar_mu_coeff) * torch.clamp(mu.detach(), 0.0, 1.0)
                logvar = self.fc_logvar(x).squeeze(-1) + extra
            else:
                logvar = self.fc_logvar(x).squeeze(-1)
            return mu, logvar
        else:
            return self.fc(x).squeeze(-1)


def init_kaiming(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
