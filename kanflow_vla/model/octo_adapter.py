"""
Frozen Octo adapter for KANFlow-VLA quick evaluation.

This uses a pretrained Octo transformer as a feature extractor and converts its
`readout_action` embedding into a fixed-size condition vector for the existing
RWKV+GroupKAN+CFM action head.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class OctoConditionEncoder(nn.Module):
    """Use a pretrained Octo checkpoint as a frozen condition encoder."""

    def __init__(
        self,
        checkpoint_path: str = "hf://rail-berkeley/octo-small-1.5",
        output_dim: int = 256,
        platform: str = "cpu",
    ):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.output_dim = output_dim
        self.platform = platform

        if platform == "cpu":
            os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
            os.environ.setdefault("JAX_PLATFORMS", "cpu")
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

        from octo.model.octo_model import OctoModel

        self._octo = OctoModel.load_pretrained(checkpoint_path)
        self.token_dim = int(self._octo.config["model"]["token_embedding_size"])

    @staticmethod
    def _maybe_unnormalize(images: torch.Tensor) -> torch.Tensor:
        """Undo ImageNet normalization if the tensor looks normalized."""
        if images.numel() == 0:
            return images

        img_min = float(images.min().detach().cpu())
        img_max = float(images.max().detach().cpu())
        if img_min >= 0.0 and img_max <= 1.0:
            return images.clamp(0.0, 1.0)

        mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3, 1, 1)
        std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3, 1, 1)
        return (images * std + mean).clamp(0.0, 1.0)

    def _prepare_images(self, images: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert normalized torch images into Octo observation images.

        Input shape is canonicalized to (B, T, V, C, H, W).
        """
        images = self._maybe_unnormalize(images.float())

        if images.ndim == 4:
            images = images[:, None, None, :, :, :]
        elif images.ndim == 5:
            images = images[:, :, None, :, :, :]
        elif images.ndim != 6:
            raise ValueError(f"Unsupported image shape for Octo adapter: {tuple(images.shape)}")

        bsz, timesteps, views, channels, _, _ = images.shape
        primary = images[:, :, 0]
        wrist = images[:, :, 1 if views > 1 else 0]

        primary = F.interpolate(
            primary.reshape(bsz * timesteps, channels, primary.shape[-2], primary.shape[-1]),
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        )
        wrist = F.interpolate(
            wrist.reshape(bsz * timesteps, channels, wrist.shape[-2], wrist.shape[-1]),
            size=(128, 128),
            mode="bilinear",
            align_corners=False,
        )

        primary = primary.reshape(bsz, timesteps, channels, 256, 256)
        wrist = wrist.reshape(bsz, timesteps, channels, 128, 128)

        primary = (
            primary.permute(0, 1, 3, 4, 2).mul(255.0).round().clamp(0, 255).byte().cpu().numpy()
        )
        wrist = (
            wrist.permute(0, 1, 3, 4, 2).mul(255.0).round().clamp(0, 255).byte().cpu().numpy()
        )
        return primary, wrist

    @lru_cache(maxsize=256)
    def _cached_task(self, texts_key: tuple[str, ...]):
        return self._octo.create_tasks(texts=list(texts_key))

    def forward(
        self,
        images: torch.Tensor,
        task_texts: list[str] | tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        """Produce a torch condition vector from Octo transformer outputs."""
        if images.ndim == 4:
            batch_size = images.shape[0]
            timesteps = 1
        else:
            batch_size = images.shape[0]
            timesteps = images.shape[1]

        if task_texts is None:
            task_texts = [""] * batch_size
        else:
            task_texts = list(task_texts)

        if len(task_texts) != batch_size:
            if len(task_texts) == 1:
                task_texts = task_texts * batch_size
            else:
                raise ValueError(
                    f"task_texts batch mismatch: expected {batch_size}, got {len(task_texts)}"
                )

        image_primary, image_wrist = self._prepare_images(images)
        timestep_pad_mask = np.ones((batch_size, timesteps), dtype=bool)
        observations = {
            "image_primary": image_primary,
            "image_wrist": image_wrist,
            "timestep_pad_mask": timestep_pad_mask,
        }
        tasks = self._cached_task(tuple(task_texts))

        outputs = self._octo.run_transformer(
            observations,
            tasks,
            timestep_pad_mask,
            train=False,
        )
        readout = np.asarray(outputs["readout_action"].tokens[:, -1, 0, :], dtype=np.float32)

        if readout.shape[-1] > self.output_dim:
            readout = readout[:, : self.output_dim]
        elif readout.shape[-1] < self.output_dim:
            pad = np.zeros((readout.shape[0], self.output_dim - readout.shape[-1]), dtype=np.float32)
            readout = np.concatenate([readout, pad], axis=-1)

        return torch.from_numpy(readout).to(device=images.device, dtype=torch.float32)
