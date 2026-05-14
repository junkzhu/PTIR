# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import torch.nn.functional as F


@torch.no_grad()
def normal_mae(
    pred_normal: torch.Tensor,
    gt_normal: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    average_full_image: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Normal MAE: mean angular error in degrees.

    Args:
        pred_normal: [..., 3], normal in world/camera space.
        gt_normal: Same shape as pred_normal, in the same space.
        valid_mask: Optional bool/float mask shaped [...] or [..., 1].
        average_full_image: If true, masked-out pixels contribute zero error
            but remain in the denominator.
        eps: Normalization epsilon.
    """
    pred = F.normalize(pred_normal, dim=-1, eps=eps)
    gt = F.normalize(gt_normal, dim=-1, eps=eps)

    dot = (pred * gt).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)
    angular_error = torch.acos(dot) * 180.0 / torch.pi

    if valid_mask is not None:
        valid_mask = valid_mask.bool()
        if valid_mask.ndim == angular_error.ndim + 1:
            valid_mask = valid_mask.squeeze(-1)
        if average_full_image:
            angular_error = torch.where(valid_mask, angular_error, torch.zeros_like(angular_error))
            return angular_error.mean()
        angular_error = angular_error[valid_mask]

    if angular_error.numel() == 0:
        return pred_normal.new_tensor(0.0)

    return angular_error.mean()


class NormalUtils:
    """Utilities for normal processing."""

    def __init__(self):
        pass

    def initialize_shading_normal_from_rotation(self, rotation: torch.Tensor) -> torch.Tensor:
        rotation_matrix = self._quaternion_to_so3(rotation)
        local_z_axis = rotation_matrix[:, :, 2]
        return F.normalize(local_z_axis, dim=-1, eps=1e-6)

    @torch.cuda.nvtx.range("generate_pseudo_normal")
    def depth_to_pseudo_normal(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        T_to_world: torch.Tensor,
        pred_dist: torch.Tensor,
        valid: torch.Tensor,
        pred_opacity: Optional[torch.Tensor] = None,
        foreground_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rays_o = self._to_channel_last_batch(rays_o, channels=3)
        rays_d = self._to_channel_last_batch(rays_d, channels=3)
        pred_dist = self._to_channel_last_batch(pred_dist, channels=1)
        valid = self._to_channel_last_batch(valid, channels=1).bool()
        if pred_opacity is not None:
            pred_opacity = self._to_channel_last_batch(pred_opacity.detach(), channels=1).clamp(0.0, 1.0)

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
            if pred_opacity is not None:
                normals = normals * pred_opacity[:, 1:-1, 1:-1]
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

    @staticmethod
    def _quaternion_to_so3(rotation: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.norm(rotation, dim=1, keepdim=True).clamp_min(1e-8)
        q = rotation / norm

        w = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]

        rotation_matrix = torch.zeros((q.shape[0], 3, 3), dtype=rotation.dtype, device=rotation.device)
        rotation_matrix[:, 0, 0] = 1 - 2 * (y * y + z * z)
        rotation_matrix[:, 0, 1] = 2 * (x * y - w * z)
        rotation_matrix[:, 0, 2] = 2 * (x * z + w * y)
        rotation_matrix[:, 1, 0] = 2 * (x * y + w * z)
        rotation_matrix[:, 1, 1] = 1 - 2 * (x * x + z * z)
        rotation_matrix[:, 1, 2] = 2 * (y * z - w * x)
        rotation_matrix[:, 2, 0] = 2 * (x * z - w * y)
        rotation_matrix[:, 2, 1] = 2 * (y * z + w * x)
        rotation_matrix[:, 2, 2] = 1 - 2 * (x * x + y * y)
        return rotation_matrix
