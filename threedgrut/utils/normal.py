# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import torch.nn.functional as F


class NormalUtils:
    """Utilities for normal processing."""

    def __init__(self):
        pass

    @torch.cuda.nvtx.range("generate_pseudo_normal")
    def depth_to_pseudo_normal(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        T_to_world: torch.Tensor,
        pred_dist: torch.Tensor,
        valid: torch.Tensor,
        foreground_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rays_o = self._to_channel_last_batch(rays_o, channels=3)
        rays_d = self._to_channel_last_batch(rays_d, channels=3)
        pred_dist = self._to_channel_last_batch(pred_dist, channels=1)
        valid = self._to_channel_last_batch(valid, channels=1).bool()

        if foreground_mask is not None:
            foreground_mask = self._prepare_mask(foreground_mask, pred_dist)
            valid = valid & foreground_mask

        pred_dist = torch.where(valid, pred_dist, torch.zeros_like(pred_dist))
        points = rays_o + pred_dist * rays_d

        world_rotation = T_to_world[:, :3, :3]
        world_translation = T_to_world[:, None, None, :3, 3]
        points = torch.einsum("bij,bhwj->bhwi", world_rotation, points) + world_translation

        pseudo_normal = torch.zeros_like(points)
        pseudo_normal_mask = torch.zeros_like(valid)
        if pred_dist.shape[1] >= 3 and pred_dist.shape[2] >= 3:
            valid_normal_mask = (
                valid[:, 1:-1, 1:-1]
                & valid[:, 1:-1, 2:]
                & valid[:, 1:-1, :-2]
                & valid[:, 2:, 1:-1]
                & valid[:, :-2, 1:-1]
            )

            dx = points[:, 1:-1, 2:] - points[:, 1:-1, :-2]
            dy = points[:, 2:, 1:-1] - points[:, :-2, 1:-1]
            dx = torch.where(valid_normal_mask.expand_as(dx), dx, torch.zeros_like(dx))
            dy = torch.where(valid_normal_mask.expand_as(dy), dy, torch.zeros_like(dy))

            normals = F.normalize(torch.cross(dy, dx, dim=-1), dim=-1, eps=1e-6)
            normals = self._remap_pseudo_normal_axes(normals)
            normals = torch.where(valid_normal_mask.expand_as(normals), normals, torch.zeros_like(normals))

            pseudo_normal[:, 1:-1, 1:-1] = normals
            pseudo_normal_mask[:, 1:-1, 1:-1] = valid_normal_mask

        return pseudo_normal, pseudo_normal_mask

    @staticmethod
    def _prepare_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        mask = NormalUtils._to_channel_last_batch(mask, channels=1)
        if mask.shape[-3:-1] != reference.shape[-3:-1]:
            mask = mask.permute(0, 3, 1, 2).float()
            mask = F.interpolate(mask, size=reference.shape[1:3], mode="nearest")
            mask = mask.permute(0, 2, 3, 1)

        return mask > 0.0

    @staticmethod
    def _to_channel_last_batch(tensor: torch.Tensor, channels: int) -> torch.Tensor:
        if tensor.ndim == 2:
            tensor = tensor[None, ..., None]
        elif tensor.ndim == 3:
            if tensor.shape[-1] == channels:
                tensor = tensor.unsqueeze(0)
            elif tensor.shape[0] == channels:
                tensor = tensor.permute(1, 2, 0).unsqueeze(0)
            else:
                tensor = tensor.unsqueeze(-1)
        elif tensor.ndim == 4:
            if tensor.shape[-1] == channels:
                pass
            elif tensor.shape[1] == channels:
                tensor = tensor.permute(0, 2, 3, 1)
            else:
                raise ValueError(f"Expected tensor with {channels} channels, got shape {tuple(tensor.shape)}")
        else:
            raise ValueError(f"Expected a 2D, 3D, or 4D tensor, got shape {tuple(tensor.shape)}")

        if tensor.shape[-1] != channels:
            raise ValueError(f"Expected tensor with {channels} channels, got shape {tuple(tensor.shape)}")

        return tensor

    def _remap_pseudo_normal_axes(self, normals: torch.Tensor) -> torch.Tensor:
        return normals
