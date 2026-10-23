
from typing import Any, Mapping

import numpy as np
import torch
from scipy.signal import cheby2, filtfilt
from torch import nn

from ..models import ModelPort
from .pulseppg_net import Net


PULSEPPG_CFG = {
    "in_channels": 1,
    "base_filters": 128,
    "kernel_size": 11,
    "stride": 2,
    "groups": 1,
    "n_block": 12,
    "finalpool": "max",
}


def _filter_pulseppg_ppg(x: torch.Tensor, fs: float) -> torch.Tensor:
    """Apply PulsePPG's 0.5-12 Hz Chebyshev II PPG bandpass."""
    x_np = x.detach().cpu().numpy()
    b, a = cheby2(4, 20, [0.5, 12.0], btype="bandpass", fs=fs)
    filtered = filtfilt(b, a, x_np, axis=-1).astype(np.float32)
    return torch.from_numpy(filtered).to(device=x.device, dtype=x.dtype)


class PulsePPGPort(ModelPort):
    def __init__(self, model_init_cfg: Mapping[str, Any]):
        super().__init__(model_init_cfg)

    def prepare_batch(
        self,
        batch: Mapping[str, Any],
        model_cfg: Mapping[str, Any],
        device: str,
    ) -> torch.Tensor:
        # PulsePPG expects channel-first PPG: (B, 1, T).
        # Use per-subject z-score normalisation followed by its native bandpass.
        x = batch["x"].to(device=device, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"PulsePPG expects input shaped (B, T) or (B, 1, T), got {tuple(x.shape)}")

        if "ppg_mean" not in batch or "ppg_std" not in batch:
            raise KeyError('PulsePPG requires "ppg_mean" and "ppg_std" in the batch')
        mean = batch["ppg_mean"].to(device=device, dtype=torch.float32).view(-1, 1, 1)
        std = batch["ppg_std"].to(device=device, dtype=torch.float32).view(-1, 1, 1)
        x = (x - mean) / (std + 1e-6)
        fs = float(self.model_init_cfg["target_fs"])
        return _filter_pulseppg_ppg(x, fs)
    
    def load_encoder(self) -> nn.Module:
        ckpt_path = self.model_init_cfg.get("ckpt_path")
        if ckpt_path is None:
            raise KeyError("Must specify checkpoint path for PulsePPG")

        model = Net(**PULSEPPG_CFG)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["net"])
        return model
