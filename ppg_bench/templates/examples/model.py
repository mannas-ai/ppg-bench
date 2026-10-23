"""Replace this example encoder with your pretrained model."""

import torch
from torch import nn

from ppg_bench.models import ModelPort


class MyModel(ModelPort):
    def load_encoder(self):
        # Return a fresh encoder each time: finetuning reloads it for each trial.
        encoder = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        checkpoint = self.model_init_cfg.get("ckpt_path")
        if checkpoint:
            encoder.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True)
            )
        return encoder

    def prepare_batch(self, batch, model_cfg, device):
        # x arrives as (batch, samples), at the fs specified in model.yaml.
        x = batch["x"].float().to(device)
        x = (x - x.mean(-1, keepdim=True)) / (
            x.std(-1, keepdim=True, unbiased=False) + 1e-6
        )
        return x.unsqueeze(1)
