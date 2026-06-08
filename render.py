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

import argparse
from pathlib import Path

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="path to the pretrained checkpoint",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="",
        help="Path to the training data, if not provided taken from ckpt",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        type=str,
        help="Output path. Required unless --relight is set; --relight defaults to the checkpoint run directory.",
    )
    parser.add_argument(
        "--save-gt",
        action="store_false",
        help="If set, the GT images will not be saved [True by default]",
    )
    parser.add_argument(
        "--compute-extra-metrics",
        action="store_false",
        help="If set, extra image metrics will not be computed [True by default]",
    )
    parser.add_argument(
        "--relight",
        action="store_true",
        help="If set, render the scaled-albedo checkpoint under every environment map in --environment-dir.",
    )
    parser.add_argument(
        "--environment-dir",
        type=str,
        default=None,
        help="Folder containing environment maps used by --relight.",
    )
    args = parser.parse_args()

    if args.relight and not args.environment_dir:
        parser.error("--environment-dir is required when --relight is set")
    if not args.relight and not args.out_dir:
        parser.error("--out-dir is required unless --relight is set")

    from threedgrut.render import Renderer

    out_dir = args.out_dir
    if args.relight and out_dir is None:
        out_dir = str(Path(args.checkpoint).resolve().parent)

    if args.relight:
        renderer = Renderer.from_checkpoint(
            checkpoint_path=str(Path(out_dir) / "ckpt_last_scaled.pt"),
            path=args.path,
            out_dir=out_dir,
            save_gt=False,
            computes_extra_metrics=False,
            create_run_dir=False,
        )
        renderer.render_relight_all(environment_dir=args.environment_dir)
    else:
        renderer = Renderer.from_checkpoint(
            checkpoint_path=args.checkpoint,
            path=args.path,
            out_dir=out_dir,
            save_gt=args.save_gt,
            computes_extra_metrics=args.compute_extra_metrics,
        )
        renderer.render_all()
