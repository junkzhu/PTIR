from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate RGB2X diffusion priors for RGB images."
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Dataset root used to preserve relative output paths.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        nargs="*",
        default=None,
        help="Specific RGB image paths. If omitted, recursively finds png/jpg/jpeg under dataset_root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "rgb2x" / "outputs",
        help="Output root for generated priors.",
    )
    parser.add_argument(
        "--aovs",
        type=str,
        default="normal",
        help="Comma-separated AOVs. Currently default is normal; extensible to albedo,roughness,metallic,irradiance.",
    )
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate priors even if output files exist.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only the local model cache under rgb2x/model_cache.",
    )
    return parser.parse_args()


def _find_images(dataset_root: Path) -> list[Path]:
    valid_exts = {".png", ".jpg", ".jpeg"}
    return sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in valid_exts
        and "_prior_" not in path.stem
        and not path.stem.endswith("_mask")
    )


def main() -> None:
    args = _parse_args()
    from threedgrut.utils.rgb2x_prior import generate_rgb2x_priors

    dataset_root = args.dataset_root.expanduser().resolve()
    image_paths = args.images if args.images is not None else _find_images(dataset_root)
    aovs = tuple(aov.strip() for aov in args.aovs.split(",") if aov.strip())

    generate_rgb2x_priors(
        image_paths=image_paths,
        dataset_root=dataset_root,
        output_root=args.output_dir,
        aovs=aovs,
        input_size=args.input_size,
        inference_steps=args.inference_steps,
        seed=args.seed,
        skip_existing=not args.overwrite,
        local_files_only=args.local_files_only,
    )


if __name__ == "__main__":
    main()
