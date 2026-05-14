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

import torch
import torch.nn.functional as F
from fused_ssim import fused_ssim


@torch.cuda.nvtx.range("l1_loss")
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


@torch.cuda.nvtx.range("l2_loss")
def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


@torch.cuda.nvtx.range("ssim")
def ssim(img1, img2, window_size=11, size_average=True):
    # predicted_image, gt_image: [BS, CH, H, W], predicted_image is differentiable
    return fused_ssim(img1, img2, padding="valid")


@torch.cuda.nvtx.range("pseudo_normal_loss")
def pseudo_normal_loss(render_normal, pseudo_normal, valid_mask=None, eps=1e-6):
    """
    Args:
        render_normal: [B, H, W, 3] or [H, W, 3], world-space rendered normal.
        pseudo_normal: Same shape as render_normal, world-space pseudo normal.
        valid_mask: Optional bool/float mask shaped [B, H, W, 1], [B, H, W], [H, W, 1], or [H, W].
        eps: Normalization epsilon.
    """
    n_render = F.normalize(render_normal, dim=-1, eps=eps)
    n_pseudo = F.normalize(pseudo_normal.detach(), dim=-1, eps=eps)

    loss = 1.0 - (n_render * n_pseudo).sum(dim=-1)

    if valid_mask is not None:
        valid_mask = valid_mask.bool()
        if valid_mask.ndim == loss.ndim + 1:
            valid_mask = valid_mask.squeeze(-1)
        loss = loss[valid_mask]

    if loss.numel() == 0:
        return render_normal.sum() * 0.0

    return loss.mean()
