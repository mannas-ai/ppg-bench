"""SigmaPPG pretraining-student encoder port.

This port intentionally exposes only the original masked-PPG student encoder.
It does not use SigmaPPG's downstream wrapper, pooling layer, or task head.
"""

from functools import partial
from typing import Any, Mapping

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from torch import nn

from ..models import ModelPort, external_module


class SigmaPPGEncoderWrapper(nn.Module):
    """Mean-pool SigmaPPG's final pre-LM-head patch-token representations."""

    def __init__(self, student: nn.Module):
        super().__init__()
        self.student = student

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one mean-pooled representation per input without the CLS token or LM head."""
        patch_tokens = self.student(
            x,
            input_chans=[0, 1],
            return_patch_tokens=True,
        )
        return patch_tokens.mean(dim=1)


class SigmaPPGPort(ModelPort):
    """Load SigmaPPG's patch-100 pretraining student encoder."""

    def load_encoder(self) -> nn.Module:
        """Build the original student and strictly load its patch-100 checkpoint weights."""
        ckpt_path = self.model_init_cfg.get("ckpt_path")
        if ckpt_path is None:
            raise KeyError("Must specify ckpt_path for SigmaPPG")

        source = external_module(
            self.model_init_cfg.get("source_root"), "pretraining.pretraining_model"
        )
        encoder = source.NeuralTransformerForMaskedPPGModeling(
            PPG_size=12000,
            patch_size=100,
            in_chans=1,
            out_chans=12,
            vocab_size=4096,
            embed_dim=360,
            depth=18,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_norm=partial(nn.LayerNorm, eps=1e-6),
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop_rate=0.08,
            attn_drop_rate=0.08,
            drop_path_rate=0.1,
            init_values=0.1,
            use_abs_pos_emb=True,
            use_rel_pos_bias=False,
        )

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise ValueError("SigmaPPG checkpoint must contain a 'model' state dictionary")
        student_state = {
            key.removeprefix("student."): value
            for key, value in checkpoint["model"].items()
            if key.startswith("student.")
        }
        if not student_state:
            raise ValueError("SigmaPPG checkpoint contains no 'student.' parameters")
        encoder.load_state_dict(student_state, strict=True)
        return SigmaPPGEncoderWrapper(encoder)

    def prepare_batch(
        self,
        batch: Mapping[str, Any],
        model_cfg: Mapping[str, Any],
        device: str,
    ) -> torch.Tensor:
        """Apply SigmaPPG's pretraining filter and per-segment min-max scaling."""
        del model_cfg

        x = batch["x"]
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim != 3:
            raise ValueError(f"SigmaPPG expects input shaped (B, T) or (B, 1, T), got {tuple(x.shape)}")
        if x.shape[1] != 1:
            raise ValueError(f"SigmaPPG PPG port expects one channel, got {x.shape[1]}")

        patch_size = 100
        usable_length = (x.shape[-1] // patch_size) * patch_size
        if usable_length == 0:
            raise ValueError(
                f"SigmaPPG requires at least one {patch_size}-sample patch; got {x.shape[-1]} samples"
            )
        x = x[..., :usable_length]

        input_dtype = x.dtype
        target_fs = float(self.model_init_cfg.get("target_fs", 50))
        sos = butter(2, (0.5, 8.0), btype="bandpass", fs=target_fs, output="sos")

        filtered = np.ascontiguousarray(
            sosfiltfilt(
                sos,
                x.detach().to(device="cpu", dtype=torch.float32).numpy(),
                axis=-1,
            )
        )
        filtered = torch.as_tensor(filtered, dtype=input_dtype, device=device)

        segment_min = filtered.amin(dim=-1, keepdim=True)
        segment_range = filtered.amax(dim=-1, keepdim=True) - segment_min
        normalized = 2.0 * (filtered - segment_min) / segment_range.clamp_min(
            torch.finfo(filtered.dtype).eps
        ) - 1.0
        return torch.where(segment_range > 0, normalized, torch.zeros_like(normalized))
