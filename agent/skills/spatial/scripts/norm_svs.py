"""Normalize GDC-downloaded H&E SVS slides into RGB TIFFs for spatial inference.

Inputs: a GDC TSV manifest, <input_dir>/<id>/<filename>, and a fitted
normalizer pickle whose transform(uint8 RGB array) returns uint8 RGB.
Outputs: <output_dir>/<slide_stem>.tif and normalization_report.csv.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import importlib
import logging
from pathlib import Path
import pickle
import tempfile


LOGGER = logging.getLogger(__name__)
SKILL_ROOT = Path(__file__).resolve().parents[1]
REPORT_FIELDS = (
    "id", "filename", "input_path", "output_path", "status", "level",
    "downsample", "width", "height", "fallback_blocks", "message",
)


def read_manifest(path: Path) -> list[tuple[str, str]]:
    """Validate a slide-only GDC manifest before writing any outputs."""
    entries = []
    output_names = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not {"id", "filename"}.issubset(reader.fieldnames or []):
            raise ValueError("Manifest must be tab-separated with id and filename columns")
        for line_number, row in enumerate(reader, start=2):
            file_id = (row.get("id") or "").strip()
            filename = (row.get("filename") or "").strip()
            for field, value in (("id", file_id), ("filename", filename)):
                if not value or value in {".", ".."} or "/" in value or "\\" in value:
                    raise ValueError(f"Manifest line {line_number}: invalid {field}: {value!r}")
            if Path(filename).suffix.lower() != ".svs":
                raise ValueError(f"Manifest line {line_number}: expected an SVS file: {filename}")
            output_name = f"{Path(filename).stem}.tif".casefold()
            if output_name in output_names:
                raise ValueError(f"Duplicate output filename in manifest: {output_name}")
            output_names.add(output_name)
            entries.append((file_id, filename))
    if not entries:
        raise ValueError("Manifest contains no slides")
    return entries


def load_normalizer(path: Path):
    """Load a trusted, already-fitted normalizer; do not fit a new reference."""
    with path.open("rb") as handle:
        normalizer = pickle.load(handle)
    if not callable(getattr(normalizer, "transform", None)):
        raise ValueError("Normalizer pickle must provide a callable transform(image) method")
    return normalizer


def normalize_quadrants(image, normalizer, on_block_error: str = "error"):
    """Normalize four nonempty quadrants, preserving their size and placement."""
    import numpy as np

    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected an H x W x 3 uint8 RGB image")
    height, width = image.shape[:2]
    if min(height, width) < 1:
        raise ValueError("Cannot normalize an empty image")
    if on_block_error not in {"error", "keep"}:
        raise ValueError("on_block_error must be error or keep")
    result = np.empty_like(image)
    failures = []
    for row_name, y0, y1 in (("top", 0, height // 2), ("bottom", height // 2, height)):
        for col_name, x0, x1 in (("left", 0, width // 2), ("right", width // 2, width)):
            if y0 == y1 or x0 == x1:
                continue
            name = f"{row_name}_{col_name}"
            original = image[y0:y1, x0:x1]
            try:
                # A copy also preserves the original if transform mutates then fails.
                normalized = np.asarray(normalizer.transform(original.copy()))
                if normalized.shape != original.shape or normalized.dtype != np.uint8:
                    raise ValueError("transform must return uint8 RGB with the input shape")
            except Exception as exc:
                message = f"{name}: {type(exc).__name__}: {exc}"
                if on_block_error == "error":
                    raise RuntimeError(f"Normalization failed for {message}") from exc
                failures.append(message)
                normalized = original
                LOGGER.warning("Keeping original block: %s", message)
            result[y0:y1, x0:x1] = normalized
    return result, failures


def normalize_slide(source: Path, destination: Path, normalizer, level: int,
                    on_block_error: str, overwrite: bool) -> dict:
    """Read one pyramid level and atomically write a non-pyramidal RGB TIFF."""
    import numpy as np
    import openslide
    from PIL import Image
    import tifffile

    if not source.is_file():
        raise FileNotFoundError(f"SVS file not found: {source}")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output exists; use --overwrite to replace it: {destination}")
    with openslide.OpenSlide(str(source)) as slide:
        if not 0 <= level < slide.level_count:
            raise ValueError(f"Level {level} is unavailable; valid range is 0..{slide.level_count - 1}")
        width, height = slide.level_dimensions[level]
        downsample = float(slide.level_downsamples[level])
        with slide.read_region((0, 0), level, (width, height)) as rgba:
            # OpenSlide returns RGBA; place transparent regions on white background.
            with Image.new("RGB", rgba.size, "white") as rgb:
                rgb.paste(rgba, mask=rgba.getchannel("A"))
                image = np.array(rgb)

    normalized, failures = normalize_quadrants(image, normalizer, on_block_error)
    metadata = {
        "source_filename": source.name,
        "source_id": source.parent.name,
        "source_level": level,
        "source_downsample": downsample,
        "fallback_blocks": len(failures),
        "axes": "YXS",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tif", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        tifffile.imwrite(
            temporary, normalized, photometric="rgb", metadata=metadata,
            bigtiff=normalized.nbytes >= 2**32 - 2**25,
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "status": "partial" if failures else "success",
        "downsample": downsample, "width": width, "height": height,
        "fallback_blocks": len(failures), "message": "; ".join(failures),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, required=True, help="GDC SVS manifest (TSV)")
    parser.add_argument("--input_dir", type=Path, default=SKILL_ROOT / "data/raw",
                        help="gdc-client download root containing <id>/<filename>")
    parser.add_argument("--output_dir", type=Path, default=SKILL_ROOT / "data/images",
                        help="Directory for normalized TIFFs and normalization_report.csv")
    parser.add_argument("--normalizer_path", type=Path, required=True,
                        help="Trusted pickle of a fitted normalizer with transform(image)")
    parser.add_argument("--level", type=int, default=1, help="SVS pyramid level (0 is full resolution)")
    parser.add_argument("--on_block_error", choices=("error", "keep"), default="error",
                        help="Fail the slide, or keep failed blocks and report a partial result")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing TIFF outputs")
    args = parser.parse_args(argv)
    if args.level < 0:
        parser.error("--level must be >= 0")
    for name in ("manifest", "input_dir", "output_dir", "normalizer_path"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        entries = read_manifest(args.manifest)
        if not args.input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
        # Fail once, before starting a batch, when required packages are unavailable.
        for module_name in ("numpy", "openslide", "PIL.Image", "tifffile"):
            importlib.import_module(module_name)

        normalizer = load_normalizer(args.normalizer_path)
        report_path = args.output_dir / "normalization_report.csv"
        if report_path.resolve() in {args.manifest, args.normalizer_path}:
            raise ValueError("Report path must not overwrite the manifest or normalizer")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        counts = Counter()
        # The report describes this run; replace the previous run's report.
        with report_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
            writer.writeheader()
            for file_id, filename in entries:
                source = args.input_dir / file_id / filename
                destination = args.output_dir / f"{Path(filename).stem}.tif"
                record = {
                    "id": file_id, "filename": filename,
                    "input_path": str(source), "output_path": "",
                    "status": "failed", "level": args.level,
                    "downsample": "", "width": "", "height": "",
                    "fallback_blocks": 0, "message": "",
                }
                try:
                    if destination.resolve() in {source.resolve(), args.manifest, args.normalizer_path}:
                        raise ValueError("TIFF output must not overwrite an input file")
                    LOGGER.info("Processing %s", filename)
                    record.update(normalize_slide(
                        source, destination, normalizer, args.level,
                        args.on_block_error, args.overwrite,
                    ))
                    record["output_path"] = str(destination)
                except Exception as exc:
                    record["message"] = f"{type(exc).__name__}: {exc}"
                    LOGGER.error("%s: %s", filename, record["message"])
                counts[record["status"]] += 1
                writer.writerow(record)
                handle.flush()
        LOGGER.info("Finished: %d success, %d partial, %d failed. Report: %s",
                    counts["success"], counts["partial"], counts["failed"], report_path)
        return 1 if counts["failed"] or counts["partial"] else 0
    except Exception as exc:
        LOGGER.error("%s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
