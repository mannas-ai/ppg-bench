from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from scipy.signal import butter, sosfiltfilt

from ..models import ModelPort, external_module


ANYPPG_CFG = {
    "in_channels": 1,
    "base_filters": 64,
    "ratio": 1.0,
    "filter_list": [64, 160, 160, 400, 400, 512],
    "m_blocks_list": [2, 2, 2, 3, 3, 1],
    "kernel_size": 3,
    "stride": 2,
    "groups_width": 16,
    "use_bn": True,
    "use_do": True,
    "verbose": False,
}


def _filter_anyppg_ppg(x: torch.Tensor, fs: float) -> torch.Tensor:
    """Apply AnyPPG's official 0.5-8 Hz Butterworth PPG bandpass before z-scoring."""
    x_np = x.detach().cpu().numpy()
    sos = butter(3, [0.5, 8.0], btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, x_np, axis=-1).astype(np.float32)
    return torch.from_numpy(filtered).to(device=x.device, dtype=x.dtype)


class AnyPPGPort(ModelPort):
    def __init__(self, model_init_cfg: Mapping[str, Any]):
        super().__init__(model_init_cfg)
    
    def prepare_batch(
        self,
        batch: Mapping[str, Any],
        model_cfg: Mapping[str, Any],
        device: str,
    ) -> torch.Tensor:
        # AnyPPG expects (B, 1, T) with official bandpass filtering and window-level z-score normalization.
        x = batch["x"].to(dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"AnyPPG expects input shaped (B, T) or (B, 1, T), got {tuple(x.shape)}")

        fs = float(self.model_init_cfg["target_fs"])
        x = _filter_anyppg_ppg(x, fs).to(device=device)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / (std + 1e-6)
    
    def load_encoder(self) -> nn.Module:
        ckpt_path = self.model_init_cfg.get("ckpt_path")
        if ckpt_path is None:
            raise KeyError("Must specify checkpoint path for AnyPPG!")

        source = external_module(self.model_init_cfg.get("source_root"), "load_anyppg.resnet1d")
        model = source.Net1D(**ANYPPG_CFG)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model
