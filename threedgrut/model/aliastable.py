# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_ALIAS_TABLE_SIZE = (64, 128)
ENVIRONMENT_TYPE_2D = "2d"
ENVIRONMENT_TYPE_CUBE = "cube"
ENVIRONMENT_TYPE_SPHERICAL_GAUSSIAN = "spherical_gaussian"
ENVIRONMENT_TYPE_OPTIONS = (
    ENVIRONMENT_TYPE_2D,
    ENVIRONMENT_TYPE_CUBE,
    ENVIRONMENT_TYPE_SPHERICAL_GAUSSIAN,
)
EQUIRECTANGULAR_ENVIRONMENT_TYPES = (
    ENVIRONMENT_TYPE_2D,
    ENVIRONMENT_TYPE_SPHERICAL_GAUSSIAN,
)


@dataclass(frozen=True)
class EnvAliasTable:
    width: int
    height: int
    numCells: int
    prob: torch.Tensor
    alias: torch.Tensor
    pdf: torch.Tensor


def build_alias_table(
    weights: torch.Tensor, eps: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a Vose alias table from non-negative weights."""
    weights_tensor = torch.as_tensor(weights)
    if weights_tensor.numel() == 0:
        raise ValueError("weights must contain at least one element.")
    if eps < 0.0:
        raise ValueError(f"eps must be non-negative, got {eps}.")

    device = weights_tensor.device
    flat_weights = (
        weights_tensor.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    )
    flat_weights = torch.where(
        torch.isfinite(flat_weights) & (flat_weights > 0.0),
        flat_weights,
        torch.zeros_like(flat_weights),
    )
    if eps > 0.0:
        flat_weights = flat_weights + eps

    num_entries = flat_weights.numel()
    total_weight = float(flat_weights.sum().item())
    if total_weight <= 0.0:
        probabilities = np.ones(num_entries, dtype=np.float32)
        aliases = np.arange(num_entries, dtype=np.int64)
    else:
        scaled_weights = flat_weights.numpy() * (float(num_entries) / total_weight)
        probabilities = np.empty(num_entries, dtype=np.float32)
        aliases = np.arange(num_entries, dtype=np.int64)

        small = [int(index) for index in np.nonzero(scaled_weights < 1.0)[0]]
        large = [int(index) for index in np.nonzero(scaled_weights >= 1.0)[0]]

        while small and large:
            small_index = small.pop()
            large_index = large.pop()

            probabilities[small_index] = np.float32(
                np.clip(scaled_weights[small_index], 0.0, 1.0)
            )
            aliases[small_index] = large_index

            scaled_weights[large_index] -= 1.0 - scaled_weights[small_index]
            if scaled_weights[large_index] < 1.0:
                small.append(large_index)
            else:
                large.append(large_index)

        for index in small:
            probabilities[index] = 1.0
            aliases[index] = index
        for index in large:
            probabilities[index] = 1.0
            aliases[index] = index

    prob = (
        torch.from_numpy(probabilities)
        .to(device=device, dtype=torch.float32)
        .contiguous()
    )
    alias = (
        torch.from_numpy(aliases.astype(np.int32))
        .to(device=device, dtype=torch.int32)
        .contiguous()
    )
    return prob, alias


def _environment_luminance(
    environment: torch.Tensor,
    luminance_weights: tuple[float, float, float],
) -> torch.Tensor:
    tensor = torch.as_tensor(environment)
    if tensor.ndim != 3 or tensor.shape[-1] < 3:
        raise ValueError(
            f"Environment must have shape [H, W, C>=3], got {tuple(tensor.shape)}"
        )

    rgb = tensor[..., :3].detach().to(dtype=torch.float32)
    rgb = torch.where(torch.isfinite(rgb) & (rgb > 0.0), rgb, torch.zeros_like(rgb))
    weights = rgb.new_tensor(luminance_weights)
    return torch.sum(rgb * weights, dim=-1)


def _resize_2d_environment_for_alias_table(
    environment: torch.Tensor,
    target_size: Optional[tuple[int, int]],
) -> torch.Tensor:
    tensor = torch.as_tensor(environment).detach()
    if target_size is None:
        return tensor
    if len(target_size) != 2:
        raise ValueError(
            f"target_size must be a (height, width) pair, got {target_size}."
        )

    target_height, target_width = int(target_size[0]), int(target_size[1])
    if target_height <= 0 or target_width <= 0:
        raise ValueError(f"target_size entries must be positive, got {target_size}.")
    if tensor.ndim != 3 or tensor.shape[-1] < 3:
        raise ValueError(
            f"Environment must have shape [H, W, C>=3], got {tuple(tensor.shape)}"
        )
    if tuple(tensor.shape[:2]) == (target_height, target_width):
        return tensor

    tensor = tensor.to(dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    resized = F.interpolate(tensor, size=(target_height, target_width), mode="area")
    return resized.squeeze(0).permute(1, 2, 0).contiguous()


def _equirect_solid_angles(
    height: int, width: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    row_edges = (
        torch.arange(height + 1, dtype=dtype, device=device) / float(height) - 0.5
    ) * torch.pi
    row_solid_angles = (2.0 * torch.pi / float(width)) * (
        torch.sin(row_edges[1:]) - torch.sin(row_edges[:-1])
    )
    return (
        torch.clamp(row_solid_angles, min=0.0)
        .reshape(height, 1)
        .expand(height, width)
        .contiguous()
    )


def _cubemap_solid_angles(
    face_size: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    edges = torch.linspace(-1.0, 1.0, face_size + 1, dtype=dtype, device=device)
    u0 = edges[:-1].reshape(1, face_size)
    u1 = edges[1:].reshape(1, face_size)
    v0 = edges[:-1].reshape(face_size, 1)
    v1 = edges[1:].reshape(face_size, 1)

    def area_element(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.atan2(u * v, torch.sqrt(u * u + v * v + 1.0))

    solid_angle = (
        area_element(u1, v1)
        - area_element(u0, v1)
        - area_element(u1, v0)
        + area_element(u0, v0)
    )
    return torch.clamp(solid_angle, min=0.0).repeat(6, 1).contiguous()


def _environment_solid_angles(
    height: int,
    width: int,
    environment_type: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    normalized_type = str(environment_type).lower()
    if normalized_type == ENVIRONMENT_TYPE_CUBE:
        if height != 6 * width:
            raise ValueError(
                f"Cubemap environment must have shape [6*N, N, C], got H={height}, W={width}."
            )
        return _cubemap_solid_angles(width, device, dtype)
    return _equirect_solid_angles(height, width, device, dtype)


def environment_importance_weights(
    environment: Optional[torch.Tensor],
    environment_type: str = "2d",
    include_solid_angle: bool = True,
    luminance_weights: tuple[float, float, float] = (0.2126, 0.7152, 0.0722),
) -> Optional[torch.Tensor]:
    """Build per-texel importance weights from environment luminance."""
    if environment is None:
        return None

    normalized_type = str(environment_type).lower()
    if normalized_type not in ENVIRONMENT_TYPE_OPTIONS:
        raise ValueError(
            f"environment_type must be one of {list(ENVIRONMENT_TYPE_OPTIONS)}, got '{environment_type}'."
        )

    weights = _environment_luminance(environment, luminance_weights)
    if not include_solid_angle:
        return weights.contiguous()

    height, width = weights.shape
    solid_angles = _environment_solid_angles(
        height, width, normalized_type, weights.device, weights.dtype
    )
    return (weights * solid_angles).contiguous()


def build_environment_alias_table(
    environment: Optional[torch.Tensor],
    environment_type: str = "2d",
    target_size: Optional[tuple[int, int]] = DEFAULT_ALIAS_TABLE_SIZE,
    luminance_weights: tuple[float, float, float] = (0.2126, 0.7152, 0.0722),
    eps: float = 0.0,
) -> Optional[EnvAliasTable]:
    """Build an EnvAliasTable for environment-map importance sampling."""
    if environment is None:
        return None
    if eps < 0.0:
        raise ValueError(f"eps must be non-negative, got {eps}.")

    normalized_type = str(environment_type).lower()
    if normalized_type not in ENVIRONMENT_TYPE_OPTIONS:
        raise ValueError(
            f"environment_type must be one of {list(ENVIRONMENT_TYPE_OPTIONS)}, got '{environment_type}'."
        )

    if normalized_type in EQUIRECTANGULAR_ENVIRONMENT_TYPES:
        environment = _resize_2d_environment_for_alias_table(environment, target_size)

    luminance = _environment_luminance(environment, luminance_weights)
    if eps > 0.0:
        luminance = luminance + eps

    height, width = luminance.shape
    solid_angles = _environment_solid_angles(
        height, width, normalized_type, luminance.device, luminance.dtype
    )
    sample_weights = (luminance * solid_angles).contiguous()

    total_weight = sample_weights.sum()
    if not bool(torch.isfinite(total_weight)) or float(total_weight.item()) <= 0.0:
        sample_weights = solid_angles
        total_weight = sample_weights.sum()

    pdf = torch.where(
        solid_angles > 0.0,
        sample_weights
        / torch.clamp(
            total_weight * solid_angles, min=torch.finfo(sample_weights.dtype).tiny
        ),
        torch.zeros_like(sample_weights),
    )
    prob, alias = build_alias_table(sample_weights)
    return EnvAliasTable(
        width=int(width),
        height=int(height),
        numCells=int(height * width),
        prob=prob,
        alias=alias,
        pdf=pdf.reshape(-1).to(dtype=torch.float32).contiguous(),
    )


__all__ = [
    "DEFAULT_ALIAS_TABLE_SIZE",
    "ENVIRONMENT_TYPE_2D",
    "ENVIRONMENT_TYPE_CUBE",
    "ENVIRONMENT_TYPE_SPHERICAL_GAUSSIAN",
    "ENVIRONMENT_TYPE_OPTIONS",
    "EnvAliasTable",
    "build_alias_table",
    "build_environment_alias_table",
    "environment_importance_weights",
]
