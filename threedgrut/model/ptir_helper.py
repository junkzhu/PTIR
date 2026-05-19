# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch

from threedgrut.utils.logger import logger

if TYPE_CHECKING:
    from threedgrut.model.model import MixtureOfGaussians


def init_model_from_training_checkpoint(
    model: "MixtureOfGaussians",
    checkpoint_path: str | Path,
    setup_optimizer: bool = False,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Load a training checkpoint and initialize a model from it.

    This is intended for PTIR-style initialization from stage1 training weights,
    not for full resume training semantics.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Training checkpoint not found: {checkpoint_path}")

    load_location = map_location if map_location is not None else getattr(model, "device", None)
    logger.info(f"🤸 Loading training checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=load_location, weights_only=False)
    model.init_from_checkpoint(checkpoint, setup_optimizer=setup_optimizer)
    return checkpoint
