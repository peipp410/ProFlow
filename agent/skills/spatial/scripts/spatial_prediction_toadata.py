"""Predict spot proteins with ProFlow and write one AnnData object per slide."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from proflow_inference import DEFAULT_CHECKPOINT, load_checkpoint, load_coordinates, predict_image


DEFAULT_ROOT = Path("/home/peijiazheng/agent/server/skills/spatial")


def resolve_device(requested: str) -> torch.device:
    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise RuntimeError("--device cuda was requested, but no CUDA device is available")
    device = torch.device("cuda" if requested == "auto" and available else requested if requested != "auto" else "cpu")
    if device.type == "cuda":
        print(f"Using CUDA device {torch.cuda.current_device()}: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("No CUDA GPU is available; using CPU for inference.")
    return device


def get_shape_fingerprint(cluster_data, bins=50):
    x = cluster_data[:, 0] - np.mean(cluster_data[:, 0])
    y = cluster_data[:, 1] - np.mean(cluster_data[:, 1])
    heatmap, _, _ = np.histogram2d(x, y, bins=bins)
    return heatmap.flatten()


def prediction_to_adata(
    df_res: pd.DataFrame,
    dbscan_eps: float = 1500.0,
    dbscan_min_samples: int = 100,
    similarity_threshold: float = 0.5,
):
    """Apply the existing spatial duplicate-slice filter and build AnnData."""
    if df_res.empty:
        raise ValueError("Cannot convert an empty prediction table to AnnData")
    coords = df_res[["x", "y"]].to_numpy()
    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(coords)
    df = df_res.copy()
    df["cluster"] = db.labels_
    labels = [label for label in sorted(set(db.labels_)) if label != -1]

    if len(labels) < 2:
        final_df = df[df["cluster"] != -1].copy()
    else:
        ordered = df[df["cluster"] != -1]["cluster"].value_counts().index.tolist()
        first, second = ordered[:2]
        correlation = np.corrcoef(
            get_shape_fingerprint(df[df["cluster"] == first][["x", "y"]].to_numpy()),
            get_shape_fingerprint(df[df["cluster"] == second][["x", "y"]].to_numpy()),
        )[0, 1]
        final_df = df[df["cluster"] == first].copy() if correlation > similarity_threshold else df[df["cluster"] != -1].copy()

    final_df = final_df.drop(columns=["cluster"])
    expression = final_df.drop(columns=["x", "y"])
    adata = ad.AnnData(expression.to_numpy(dtype=np.float32))
    adata.var_names = list(expression.columns)
    adata.obs_names = [f"spot_{i}" for i in range(len(final_df))]
    adata.obsm["spatial"] = final_df[["x", "y"]].to_numpy(dtype=np.float32)
    return adata


def main():
    parser = argparse.ArgumentParser(
        prog="spatial_prediction_toadata.py",
        description="Predict protein expression with ProFlow and save AnnData results.",
    )
    parser.add_argument("--image_dir", default=str(DEFAULT_ROOT / "data/images"))
    parser.add_argument("--coord_dir", default=str(DEFAULT_ROOT / "data/coords"))
    parser.add_argument("--coord_prefix", default="valid_centers_")
    parser.add_argument("--coord_ext", default=".txt")
    parser.add_argument("--output_dir", default=str(DEFAULT_ROOT / "results"))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch_size", type=int, default=2560)
    parser.add_argument("--image_glob", default="*.tif")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dbscan_eps", type=float, default=1500.0)
    parser.add_argument("--dbscan_min_samples", type=int, default=100)
    parser.add_argument("--similarity_threshold", type=float, default=0.5)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch_size must be positive")
    if args.dbscan_min_samples < 1:
        parser.error("--dbscan_min_samples must be positive")

    image_dir = Path(args.image_dir).expanduser()
    coord_dir = Path(args.coord_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    if not image_dir.is_dir():
        parser.error(f"Image directory does not exist: {image_dir}")
    if not coord_dir.is_dir():
        parser.error(f"Coordinate directory does not exist: {coord_dir}")
    images = sorted(path for path in image_dir.glob(args.image_glob) if path.is_file() and path.suffix.lower() in {".tif", ".tiff"})
    print(f"Found {len(images)} TIFF images.")
    if not images:
        return

    device = resolve_device(args.device)
    model, preprocessing, checkpoint_path = load_checkpoint(args.checkpoint, device)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(images, desc="Predicting"):
        coord_path = coord_dir / f"{args.coord_prefix}{image_path.stem}{args.coord_ext}"
        if not coord_path.is_file():
            print(f"Skipping {image_path.name}: coordinate file not found ({coord_path})")
            continue
        coordinates = load_coordinates(coord_path)
        if coordinates.empty:
            print(f"Skipping {image_path.name}: no coordinates parsed")
            continue

        result = predict_image(model, image_path, coordinates, preprocessing, args.batch_size, device)
        adata = prediction_to_adata(
            result,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
            similarity_threshold=args.similarity_threshold,
        )
        output_path = output_dir / f"{image_path.stem}.h5ad"
        if output_path.resolve() in {image_path.resolve(), coord_path.resolve(), checkpoint_path}:
            raise ValueError(f"Output path would overwrite an input: {output_path}")
        temporary_path = output_path.with_name(output_path.name + ".tmp")
        try:
            # The temporary filename ends in ".tmp"; call the H5AD writer
            # explicitly because AnnData cannot infer the format from it.
            adata.write_h5ad(temporary_path)
            temporary_path.replace(output_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        print(f"Saved {output_path} ({adata.n_obs} spots, {adata.n_vars} proteins)")


if __name__ == "__main__":
    main()
