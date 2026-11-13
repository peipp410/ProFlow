"""Use StarDist nuclei segmentation to generate spatial spot-center coordinates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


TIFF_SUFFIXES = {".tif", ".tiff"}
DEFAULT_ROOT = Path("/home/peijiazheng/agent/server/skills/spatial")


def read_rgb_tiff(path: Path):
    """Read a TIFF as a 2-D RGB image, handling common channel-first layouts."""
    import numpy as np
    import tifffile

    image = np.squeeze(np.asarray(tifffile.imread(path)))
    if image.ndim == 4 and image.shape[-1] in (3, 4):
        image = image[0]
    elif image.ndim == 4 and image.shape[1] in (3, 4):
        image = np.moveaxis(image[0], 0, -1)

    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] in (3, 4):
        image = image[..., :3]
    elif image.ndim == 3 and image.shape[0] in (3, 4):
        image = np.moveaxis(image[:3], 0, -1)
    else:
        raise ValueError(f"Expected a 2-D RGB H&E TIFF, got shape {image.shape}: {path}")

    if image.shape[-1] != 3:
        raise ValueError(f"Expected three RGB channels, got shape {image.shape}: {path}")
    return image


def predict_nuclei(image_rgb, model, n_tiles=None, prob_thresh=None, nms_thresh=None):
    """Normalize an RGB tile and return its StarDist instance-label image."""
    from csbdeep.utils import normalize

    normalized = normalize(image_rgb, 1, 99.8, axis=(0, 1, 2))
    kwargs: dict[str, Any] = {"axes": "YXC"}
    if n_tiles is not None:
        kwargs["n_tiles"] = n_tiles
    if prob_thresh is not None:
        kwargs["prob_thresh"] = prob_thresh
    if nms_thresh is not None:
        kwargs["nms_thresh"] = nms_thresh
    labels, _ = model.predict_instances(normalized, **kwargs)
    return labels


def nucleus_spots_from_labels(labels):
    """Return one bounding-box midpoint and morphology record per nucleus."""
    import numpy as np

    spots = []
    label_ids = np.unique(labels)
    for label_id in label_ids[label_ids > 0]:
        ys, xs = np.nonzero(labels == label_id)
        if xs.size == 0:
            continue
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        spots.append(
            {
                "label": int(label_id),
                "x": (x_min + x_max) / 2.0,
                "y": (y_min + y_max) / 2.0,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "area": int(xs.size),
            }
        )
    return spots


def quadrant_specs(height: int, width: int, overlap: int = 64):
    """Split a large image into four overlapping crops with disjoint owner cores."""
    if overlap < 0:
        raise ValueError("quadrant overlap must be >= 0")

    y_mid, x_mid = height // 2, width // 2
    y_ranges = ((0, y_mid), (y_mid, height))
    x_ranges = ((0, x_mid), (x_mid, width))
    specs = []
    for core_y0, core_y1 in y_ranges:
        for core_x0, core_x1 in x_ranges:
            if core_y0 == core_y1 or core_x0 == core_x1:
                continue
            specs.append(
                {
                    "x0": max(0, core_x0 - overlap),
                    "y0": max(0, core_y0 - overlap),
                    "x1": min(width, core_x1 + overlap),
                    "y1": min(height, core_y1 + overlap),
                    "core_x0": core_x0,
                    "core_y0": core_y0,
                    "core_x1": core_x1,
                    "core_y1": core_y1,
                }
            )
    return specs


def spots_owned_by_core(spots, spec):
    """Translate crop-local boxes to slide coordinates and deduplicate overlap."""
    translated = []
    for spot in spots:
        global_spot = dict(spot)
        for key in ("x", "x_min", "x_max"):
            global_spot[key] += spec["x0"]
        for key in ("y", "y_min", "y_max"):
            global_spot[key] += spec["y0"]
        if (
            spec["core_x0"] <= global_spot["x"] < spec["core_x1"]
            and spec["core_y0"] <= global_spot["y"] < spec["core_y1"]
        ):
            translated.append(global_spot)
    return translated


def save_centers(spots, txt_path: Path, csv_path: Path):
    with txt_path.open("w", encoding="utf-8") as handle:
        for spot in spots:
            handle.write(f"{spot['x']:.2f},{spot['y']:.2f}\n")

    fields = ["label", "x", "y", "x_min", "y_min", "x_max", "y_max", "area"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(spots)


def save_thumbnail(image_rgb, spots, output_path: Path, scale_factor: float = 0.2):
    """Save a PNG thumbnail with center points overlaid in red."""
    import cv2
    import numpy as np

    if not 0 < scale_factor <= 1:
        raise ValueError("thumbnail scale must be in (0, 1]")
    height, width = image_rgb.shape[:2]
    thumb_width = max(1, int(round(width * scale_factor)))
    thumb_height = max(1, int(round(height * scale_factor)))
    thumbnail = cv2.resize(image_rgb, (thumb_width, thumb_height))
    red_value = np.iinfo(thumbnail.dtype).max if np.issubdtype(thumbnail.dtype, np.integer) else 1.0
    red = tuple(int(v) if np.issubdtype(thumbnail.dtype, np.integer) else float(v) for v in (red_value, 0, 0))
    radius = max(1, int(round(4 * scale_factor)))
    for spot in spots:
        x = int(round(spot["x"] * scale_factor))
        y = int(round(spot["y"] * scale_factor))
        cv2.circle(thumbnail, (x, y), radius, red, thickness=-1)
    if not cv2.imwrite(str(output_path), cv2.cvtColor(thumbnail, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write thumbnail: {output_path}")


def iter_tiff_files(input_path: Path):
    if input_path.is_file():
        if input_path.suffix.lower() not in TIFF_SUFFIXES:
            raise ValueError(f"Input file is not a TIFF: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    return sorted(path for path in input_path.iterdir() if path.suffix.lower() in TIFF_SUFFIXES)


def parse_n_tiles(value: str):
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Use two values, for example 2,2 or 4x4")
    try:
        y_tiles, x_tiles = (int(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("n_tiles values must be integers") from exc
    if y_tiles < 1 or x_tiles < 1:
        raise argparse.ArgumentTypeError("n_tiles values must be >= 1")
    return y_tiles, x_tiles


def process_single_tiff(
    file_path: Path,
    output_dir: Path,
    model,
    n_tiles=None,
    prob_thresh=None,
    nms_thresh=None,
    quadrant_overlap: int = 64,
    thumbnail_scale: float = 0.2,
):
    image_rgb = read_rgb_tiff(file_path)
    height, width = image_rgb.shape[:2]
    spots = []

    for spec in quadrant_specs(height, width, overlap=quadrant_overlap):
        crop = image_rgb[spec["y0"] : spec["y1"], spec["x0"] : spec["x1"]]
        labels = predict_nuclei(
            crop,
            model,
            n_tiles=n_tiles,
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
        )
        local_spots = nucleus_spots_from_labels(labels)
        owned = spots_owned_by_core(local_spots, spec)
        for spot in owned:
            spot["label"] = len(spots) + 1
            spots.append(spot)
        del crop, labels, local_spots

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = file_path.stem
    txt_path = output_dir / f"valid_centers_{stem}.txt"
    csv_path = output_dir / f"nucleus_spots_{stem}.csv"
    thumbnail_path = output_dir / f"thumbnail_with_centers_{stem}.png"
    save_centers(spots, txt_path, csv_path)
    save_thumbnail(image_rgb, spots, thumbnail_path, scale_factor=thumbnail_scale)
    print(f"{file_path}: found {len(spots)} nuclei")
    print(f"  centers:   {txt_path}")
    print(f"  details:   {csv_path}")
    print(f"  thumbnail: {thumbnail_path}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Use StarDist H&E nuclei segmentation to export one coordinate per nucleus."
    )
    parser.add_argument(
        "--input_dir",
        default=str(DEFAULT_ROOT / "data/images"),
        help="One H&E TIFF file or a directory containing TIFF files.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_ROOT / "data/coords"),
        help="Directory for valid_centers_*.txt, nucleus_spots_*.csv, and thumbnails.",
    )
    parser.add_argument("--model", default="2D_versatile_he", help="Pretrained StarDist model name.")
    parser.add_argument(
        "--n_tiles",
        type=parse_n_tiles,
        default=None,
        help="Optional StarDist tile grid as Y,X (for example 2,2) to reduce memory use.",
    )
    parser.add_argument(
        "--quadrant_overlap",
        type=int,
        default=64,
        help="Overlap in pixels around quadrant boundaries; default: 64.",
    )
    parser.add_argument("--prob_thresh", type=float, default=None, help="Optional StarDist probability threshold.")
    parser.add_argument("--nms_thresh", type=float, default=None, help="Optional StarDist NMS threshold.")
    parser.add_argument(
        "--thumbnail_scale", type=float, default=0.2, help="Thumbnail scale factor in (0, 1]."
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.quadrant_overlap < 0:
        raise SystemExit("--quadrant_overlap must be >= 0")
    if not 0 < args.thumbnail_scale <= 1:
        raise SystemExit("--thumbnail_scale must be in (0, 1]")

    input_path = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    tiff_files = iter_tiff_files(input_path)
    if not tiff_files:
        print(f"No TIFF files found in {input_path}")
        return

    from stardist.models import StarDist2D

    model = StarDist2D.from_pretrained(args.model)
    if model is None:
        raise RuntimeError(f"Could not load pretrained StarDist model: {args.model}")

    failures = []
    for file_path in tiff_files:
        try:
            process_single_tiff(
                file_path=file_path,
                output_dir=output_dir,
                model=model,
                n_tiles=args.n_tiles,
                prob_thresh=args.prob_thresh,
                nms_thresh=args.nms_thresh,
                quadrant_overlap=args.quadrant_overlap,
                thumbnail_scale=args.thumbnail_scale,
            )
        except Exception as exc:
            failures.append((file_path, exc))
            print(f"Failed to process {file_path}: {exc}")

    if failures:
        failed_files = ", ".join(path.name for path, _ in failures)
        raise RuntimeError(f"Failed to process {len(failures)} of {len(tiff_files)} TIFF files: {failed_files}")


if __name__ == "__main__":
    main()
