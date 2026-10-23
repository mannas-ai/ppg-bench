from typing import Any, Mapping

import numpy as np
import torch
from scipy.signal import find_peaks, welch
from scipy.integrate import trapezoid
from torch import nn

from ..models import ModelPort


PRV_FEATURES = ["mean_NN", "mean_HR", "SDNN", "RMSSD", "pNN50", "LF", "HF", "LF_HF"]


class PRVCtrlEncoder(nn.Module):
    """Compute a deterministic PRV feature vector directly from each PPG window."""

    def __init__(self, fs: float):
        """Store the sampling rate used to convert peak locations into intervals."""
        super().__init__()
        self.fs = float(fs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return PRV features shaped (batch, 8) for frozen-probe evaluation."""
        if x.ndim == 3:
            if x.shape[1] != 1:
                raise ValueError(f"PRVCtrl expects (B, T) or (B, 1, T), got {tuple(x.shape)}")
            x = x[:, 0, :]
        if x.ndim != 2:
            raise ValueError(f"PRVCtrl expects (B, T) or (B, 1, T), got {tuple(x.shape)}")

        x_np = x.detach().cpu().numpy()
        features = np.stack([self._features_for_window(row) for row in x_np], axis=0)
        return torch.as_tensor(features, dtype=x.dtype, device=x.device)

    def _features_for_window(self, ppg: np.ndarray) -> np.ndarray:
        """Compute PRV features from one PPG window using peak-to-peak intervals."""
        out = np.zeros(len(PRV_FEATURES), dtype=np.float32)
        ppg = np.asarray(ppg, dtype=np.float32).reshape(-1)
        if ppg.size < max(3, int(self.fs)):
            return out

        if not np.isfinite(ppg).all():
            return out
        scale = float(np.std(ppg))
        if scale <= 1e-8:
            return out
        ppg = (ppg - float(np.mean(ppg))) / scale

        peaks = self._detect_ppg_peaks(ppg)
        if peaks.size < 3:
            return out

        nn_ms = np.diff(peaks).astype(np.float32) / self.fs * 1000.0
        nn_ms = nn_ms[(nn_ms >= 300.0) & (nn_ms <= 2000.0)]
        if nn_ms.size < 2:
            return out

        diff_ms = np.diff(nn_ms)
        mean_nn = float(np.mean(nn_ms))
        out[0] = mean_nn
        out[1] = 60000.0 / mean_nn
        out[2] = float(np.std(nn_ms, ddof=1))
        out[3] = float(np.sqrt(np.mean(diff_ms**2))) if diff_ms.size else 0.0
        out[4] = float(100.0 * np.mean(np.abs(diff_ms) > 50.0)) if diff_ms.size else 0.0

        lf, hf = self._frequency_features(nn_ms)
        out[5] = lf
        out[6] = hf
        out[7] = lf / hf if hf > 1e-8 else 0.0
        return out

    def _detect_ppg_peaks(self, ppg: np.ndarray) -> np.ndarray:
        """Find systolic-like peaks, falling back to inverted peaks if needed."""
        min_distance = max(1, int(0.30 * self.fs))
        prominence = max(0.1, 0.25 * float(np.std(ppg)))
        pos_peaks, _ = find_peaks(ppg, distance=min_distance, prominence=prominence)
        neg_peaks, _ = find_peaks(-ppg, distance=min_distance, prominence=prominence)
        return pos_peaks if pos_peaks.size >= neg_peaks.size else neg_peaks

    def _frequency_features(self, nn_ms: np.ndarray) -> tuple[float, float]:
        """Estimate LF and HF power from an interpolated interval series."""
        if nn_ms.size < 30 or float(np.sum(nn_ms)) < 120000.0:
            return 0.0, 0.0

        t = np.cumsum(nn_ms) / 1000.0
        grid = np.arange(float(t[0]), float(t[-1]), 0.25)
        if grid.size < 16:
            return 0.0, 0.0

        interval_series = np.interp(grid, t, nn_ms)
        interval_series = interval_series - float(np.mean(interval_series))
        freq, power = welch(interval_series, fs=4.0, nperseg=min(256, interval_series.size))

        lf_mask = (freq >= 0.04) & (freq < 0.15)
        hf_mask = (freq >= 0.15) & (freq < 0.40)
        lf = float(trapezoid(power[lf_mask], freq[lf_mask])) if np.any(lf_mask) else 0.0
        hf = float(trapezoid(power[hf_mask], freq[hf_mask])) if np.any(hf_mask) else 0.0
        return lf, hf


class PRVCtrlPort(ModelPort):
    """A checkpoint-free PRV feature baseline computed from PPG windows."""

    name = "PRVCtrl"

    def prepare_batch(
        self,
        batch: Mapping[str, Any],
        model_cfg: Mapping[str, Any],
        device: str,
    ) -> torch.Tensor:
        """Move benchmark PPG windows to the target device without filtering."""
        return batch["x"].to(device=device, dtype=torch.float32)

    def load_encoder(self) -> nn.Module:
        """Build the PRV feature encoder using target_fs from the model config."""
        if "target_fs" not in self.model_init_cfg:
            raise KeyError("PRVCtrl requires target_fs in its model config")
        return PRVCtrlEncoder(fs=float(self.model_init_cfg["target_fs"]))
