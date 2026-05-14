# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F
import torchvision

from threedgrut.utils.logger import logger


@dataclass(frozen=True)
class VisualizationSpec:
    name: str
    output_key: str
    transform: Callable[[torch.Tensor], torch.Tensor]


class TrainingVisualizer:
    """Save periodic training visualizations to disk."""

    def __init__(self, output_dir: str | os.PathLike, frequency: int):
        self.frequency = int(frequency)
        self.enabled = self.frequency > 0
        self.output_dir = Path(output_dir) / "visualizations"
        self.specs = [
            VisualizationSpec("rgb", "pred_rgb", lambda image: image.clip(0.0, 1.0)),
            VisualizationSpec("normal", "pred_normals", lambda image: (0.5 * (image + 1.0)).clip(0.0, 1.0)),
        ]

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📸 Training visualizations will be saved to: {self.output_dir}")

    def should_visualize(self, step: int) -> bool:
        return self.enabled and step > 0 and step % self.frequency == 0

    @torch.no_grad()
    def save(self, step: int, outputs: dict) -> None:
        if not self.should_visualize(step):
            return

        images = self._collect_images(outputs)
        if not images:
            return

        image = self._concat_images(images)
        torchvision.utils.save_image(image, self.output_dir / f"{step:05d}.png")

    def _collect_images(self, outputs: dict) -> list[torch.Tensor]:
        images = []
        for spec in self.specs:
            image = self._to_image_batch(outputs.get(spec.output_key))
            if image is None:
                continue

            images.append(spec.transform(image))

        return images

    @staticmethod
    def _concat_images(images: list[torch.Tensor]) -> torch.Tensor:
        height, width = images[0].shape[-2:]
        resized_images = []

        for image in images:
            if image.shape[-2:] != (height, width):
                image = F.interpolate(image, size=(height, width), mode="bilinear", align_corners=False)
            resized_images.append(image)

        return torch.cat(resized_images, dim=-1)

    @staticmethod
    def _to_image_batch(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None

        tensor = tensor.detach()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        if tensor.ndim != 4:
            return None

        if tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(0, 3, 1, 2)
        elif tensor.shape[1] not in (1, 3, 4):
            return None

        return tensor.float().cpu()
