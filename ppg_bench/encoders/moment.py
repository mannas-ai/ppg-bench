from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from ..models import ModelPort
from momentfm import MOMENTPipeline

class MOMENT_EncoderWrapper(nn.Module):
    def __init__(self, model_init_cfg: Mapping[str, Any]):
        super().__init__()
        ckpt_path = model_init_cfg.get("ckpt_path")
        if ckpt_path is None:
            raise KeyError("Must specify local checkpoint path for MOMENT")

        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"MOMENT checkpoint path does not exist: {ckpt_path}")

        model_kwargs = {
            "task_name": "embedding",
            "freeze_encoder": bool(model_init_cfg.get("freeze_encoder", False)),
            "freeze_embedder": bool(model_init_cfg.get("freeze_embedder", False)),
            "freeze_head": bool(model_init_cfg.get("freeze_head", False)),
        }
        self.model = MOMENTPipeline.from_pretrained(
            str(ckpt_path),
            local_files_only=True,
            model_kwargs=model_kwargs,
        )
        self.model.init()
    
    def forward(self, x):
        # In embedding mode, reduction="mean" averages channels and valid patches.
        output = self.model(x_enc=x, reduction="mean")
        return output.embeddings


class MOMENTPort(ModelPort):
    def __init__(self, model_init_cfg: Mapping[str, Any]):
        super().__init__(model_init_cfg)
    
    def prepare_batch(
        self,
        batch: Mapping[str, Any],
        model_cfg: Mapping[str, Any],
        device: str,
    ) -> torch.Tensor:
        # MOMENT expects (B, C, T). Our PPG windows are single-channel.
        x = batch["x"]
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)

        x = x.to(device=device, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim != 3:
            raise ValueError(f"MOMENT expects input shaped (B, T) or (B, 1, T), got {tuple(x.shape)}")

        if x.shape[1] != 1:
            raise ValueError(f"MOMENT PPG port expects one channel, got {x.shape[1]}")
        return x

    def load_encoder(self) -> nn.Module:
        return MOMENT_EncoderWrapper(self.model_init_cfg)
