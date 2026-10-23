from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from scipy.signal import cheby2, filtfilt, sosfiltfilt

from ..models import ModelPort
from .papagei_net import ResNet1D, ResNet1DMoE

def load_model_without_module_prefix(model, checkpoint_path):
    # Copied from PaPaGei
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Create a new state_dict with the `module.` prefix removed
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith('module.'):
            new_key = k[7:]  # Remove `module.` prefix
        else:
            new_key = k
        new_state_dict[new_key] = v
    
    # Load the new state_dict into the model
    model.load_state_dict(new_state_dict)

    return model


MODEL_CONFIG_PAPAGEI_S = {
                'base_filters': 32,
                'kernel_size': 3,
                'stride': 2,
                'groups': 1,
                'n_block': 18,
                'n_classes': 512,
                'n_experts': 3
                }


MODEL_CONFIG_PAPAGEI_P = {'base_filters': 32,
                'kernel_size': 3,
                'stride': 2,
                'groups': 1,
                'n_block': 18,
                'n_classes': 512,
                }


def _filter_papagei_ppg(x: torch.Tensor, fs: float) -> torch.Tensor:
    """Apply PaPaGei/pyPPG 0.5-12 Hz Chebyshev II filtering plus PPG smoothing."""
    x_np = x.detach().cpu().numpy()
    sos = cheby2(4, 20, [0.5, 12.0], btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, x_np, axis=-1)

    if fs >= 75:
        win = round(fs * 50 / 1000)
        b = np.ones(win, dtype=np.float32) / win
        filtered = filtfilt(b, 1, filtered, axis=-1)

    return torch.from_numpy(filtered.astype(np.float32)).to(device=x.device, dtype=x.dtype)


class PaPaGei_EncoderWrapper(nn.Module):
    def __init__(self, ckpt_path, cfg, model_type):
        super().__init__()
        if model_type == 'S':
            self.encoder = ResNet1DMoE(in_channels=1, 
            base_filters=cfg['base_filters'], 
            kernel_size=cfg['kernel_size'],
            stride=cfg['stride'],
            groups=cfg['groups'],
            n_block=cfg['n_block'],
            n_classes=cfg['n_classes'],
            n_experts=cfg['n_experts'])
        
        elif model_type == 'P':
            self.encoder = ResNet1D(in_channels=1, 
                base_filters=cfg['base_filters'], 
                kernel_size=cfg['kernel_size'],
                stride=cfg['stride'],
                groups=cfg['groups'],
                n_block=cfg['n_block'],
                n_classes=cfg['n_classes'])        
        else:
            raise ValueError(f"Invalid model type {model_type} for PaPaGei_EncoderWrapper, expected 'S' or 'P'")
    
        self.encoder = load_model_without_module_prefix(self.encoder, ckpt_path)

    def forward(self, x):
        # output idx 0 specified in PaPaGei
        return self.encoder(x)[0]

class PaPaGei_S_Port(ModelPort):
    def __init__(self, model_init_cfg: Mapping[str, Any]):
        super().__init__(model_init_cfg)
    
    def prepare_batch(
        self,
        batch: Mapping[str, Any],
        model_cfg: Mapping[str, Any],
        device: str,
    ) -> torch.Tensor:
        # PaPaGei_S expects (B, 1, T) with official pyPPG-style filtering and window-level z-score normalization.
        x = batch["x"].to(dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"PaPaGei_S expects input shaped (B, T) or (B, 1, T), got {tuple(x.shape)}")

        fs = float(self.model_init_cfg["target_fs"])
        x = _filter_papagei_ppg(x, fs).to(device=device)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / (std + 1e-6)
    
    def load_encoder(self) -> nn.Module:
        return PaPaGei_EncoderWrapper(self.model_init_cfg.get("ckpt_path"), MODEL_CONFIG_PAPAGEI_S, model_type='S')
    
class PaPaGei_P_Port(ModelPort):
    def __init__(self, model_init_cfg: Mapping[str, Any]):
        super().__init__(model_init_cfg)
    
    def prepare_batch(
        self,
        batch: Mapping[str, Any],
        model_cfg: Mapping[str, Any],
        device: str,
    ) -> torch.Tensor:
        # PaPaGei_P expects (B, 1, T) with official pyPPG-style filtering and window-level z-score normalization.
        x = batch["x"].to(dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"PaPaGei_P expects input shaped (B, T) or (B, 1, T), got {tuple(x.shape)}")

        fs = float(self.model_init_cfg["target_fs"])
        x = _filter_papagei_ppg(x, fs).to(device=device)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / (std + 1e-6)
    
    def load_encoder(self) -> nn.Module:
        return PaPaGei_EncoderWrapper(self.model_init_cfg.get("ckpt_path"), MODEL_CONFIG_PAPAGEI_P, model_type='P')
  
