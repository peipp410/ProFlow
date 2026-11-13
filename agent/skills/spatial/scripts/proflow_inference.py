"""Shared inference implementation for the bundled ProFlow CRC checkpoint.

The architecture and preprocessing mirror
``tcga_coad_infer/inference_new/inference_tcga.py``.  Predictions are returned
on the checkpoint's trained log1p output scale without inverse transformation,
clipping, or rounding.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import tifffile


ARCHITECTURE = "crc_codex.shared_mlp.cached_features.v1"
DEFAULT_CHECKPOINT = "ckp/pred_flow_kl_mlp_new_checkpoint.pt"


def integrate_flow(flow_model, z_init, steps, cond_vec=None):
    if steps < 1:
        raise ValueError("flow_steps must be positive")
    dt = 1.0 / steps
    z = z_init.clone()
    for step in range(steps):
        t = torch.full((z.size(0), 1), step / steps, device=z.device, dtype=z.dtype)
        k1 = flow_model(z, t, cond_vec)
        k2 = flow_model(z + 0.5 * dt * k1, t + 0.5 * dt, cond_vec)
        k3 = flow_model(z + 0.5 * dt * k2, t + 0.5 * dt, cond_vec)
        k4 = flow_model(z + dt * k3, t + dt, cond_vec)
        z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return z


class ConditionalFlowNet(nn.Module):
    """Conditional velocity MLP with the checkpoint's original parameter names."""

    def __init__(self, feature_dim, cond_dim, hidden_dim):
        super().__init__()
        self.cond_dim = cond_dim
        self.net = nn.Sequential(
            nn.Linear(feature_dim + cond_dim + 1, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, z, t, cond_vec=None):
        inputs = torch.cat((z, cond_vec, t), dim=-1) if self.cond_dim else torch.cat((z, t), dim=-1)
        return self.net(inputs)


def build_conch_visual(config):
    """Construct the visual encoder from checkpoint architecture metadata."""
    from conch.open_clip_custom.coca_model import CoCa

    return CoCa(**deepcopy(config)).visual


class StaticHistFlowRNA(nn.Module):
    """Inference subset of the frozen HistFlowRNA branch stored in the checkpoint."""

    def __init__(self, hist_model, feature_dim, num_celltypes, hidden_dim):
        super().__init__()
        self.hist_model = hist_model
        self.register_buffer("class_embeddings", torch.zeros(num_celltypes, feature_dim))
        self.image_prob = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_celltypes, bias=False),
            nn.LogSoftmax(dim=1),
        )
        self.flow_model = ConditionalFlowNet(feature_dim, 0, hidden_dim)

    def forward(self, image, steps):
        image_latent, _ = self.hist_model(image)
        probabilities = self.image_prob(image_latent).exp()
        hist_features = probabilities @ self.class_embeddings
        pred_rna = integrate_flow(self.flow_model, hist_features, steps=steps)
        return pred_rna, probabilities


class ImageToProteinModel(nn.Module):
    """Checkpoint-defined image/RNA/protein model and shared protein head."""

    def __init__(self, checkpoint):
        super().__init__()
        config = checkpoint["model_config"]
        dim = int(config["spot_dim"])
        self.protein_names = list(checkpoint["protein_names"])
        self.num_proteins = int(config["num_proteins"])
        if len(self.protein_names) != self.num_proteins or len(set(self.protein_names)) != self.num_proteins:
            raise ValueError("Checkpoint protein names/count are inconsistent")

        protein_embeddings = checkpoint["protein_embeddings"].float()
        if tuple(protein_embeddings.shape) != (self.num_proteins, int(config["protein_dim"])):
            raise ValueError("Checkpoint protein embeddings are not aligned with protein names")
        self.register_buffer("protein_embeddings", protein_embeddings, persistent=False)
        self.steps = int(checkpoint["preprocessing"]["flow_steps"])

        self.hist_model = build_conch_visual(checkpoint["conch_config"])
        self.static_hist_flow = StaticHistFlowRNA(
            deepcopy(self.hist_model), dim, int(config["num_celltypes"]),
            int(config["rna_flow_hidden_dim"]),
        )
        self.rna_flow = ConditionalFlowNet(
            dim, int(config["num_celltypes"]), int(config["protein_flow_hidden_dim"])
        )
        self.image_ln = nn.LayerNorm(dim)
        self.protein_projection = nn.Linear(int(config["protein_dim"]), dim)
        self.trans_dim = nn.Linear(dim * 4, dim)
        self.prediction_layers_expression = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Dropout(float(config["head_dropout"])),
            nn.Linear(dim // 4, dim // 2),
            nn.GELU(),
            nn.Dropout(float(config["head_dropout"])),
            nn.Linear(dim // 2, 1),
        )

        trained_state = checkpoint["model_state_dict"]
        for module, prefix in (
            (self.hist_model, "hist_flow_rna.hist_model."),
            (self.rna_flow, "rna_flow."),
            (self.image_ln, "image_ln."),
            (self.protein_projection, "protein_projection."),
            (self.trans_dim, "trans_dim."),
            (self.prediction_layers_expression, "prediction_layers_expression."),
        ):
            state = {key[len(prefix):]: value for key, value in trained_state.items() if key.startswith(prefix)}
            module.load_state_dict(state, strict=True)
        self.static_hist_flow.load_state_dict(checkpoint["static_hist_flow_state_dict"], strict=True)

    def forward(self, image):
        pred_rna, probabilities = self.static_hist_flow(image, self.steps)
        pred_protein = integrate_flow(self.rna_flow, pred_rna, self.steps, probabilities)

        image_latent, _ = self.hist_model(image)
        spot_embeddings = torch.cat(
            (
                self.image_ln(image_latent),
                F.layer_norm(pred_rna, (pred_rna.size(-1),)),
                F.layer_norm(pred_protein, (pred_protein.size(-1),)),
            ),
            dim=-1,
        )
        protein_projected = self.protein_projection(self.protein_embeddings)
        protein_projected = protein_projected.unsqueeze(0).expand(image.size(0), -1, -1)
        spot_embeddings = spot_embeddings.unsqueeze(1).expand(-1, self.num_proteins, -1)
        fused = self.trans_dim(torch.cat((protein_projected, spot_embeddings), dim=-1))
        return self.prediction_layers_expression(fused).squeeze(-1)


def load_checkpoint(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve(strict=True)
    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=True, mmap=True
    )
    if checkpoint.get("architecture") != ARCHITECTURE or checkpoint.get("format_version") != 1:
        raise ValueError(
            "Unsupported checkpoint; expected the bundled shared-MLP cached-feature format v1"
        )
    model = ImageToProteinModel(checkpoint).eval().to(device)
    preprocessing = checkpoint["preprocessing"]
    if "patch_size" not in preprocessing or "image_mean" not in preprocessing or "image_std" not in preprocessing:
        raise ValueError("Checkpoint is missing image preprocessing metadata")
    print(
        f"Loaded checkpoint epoch {checkpoint.get('epoch', 'unknown')}; "
        f"{len(model.protein_names)} proteins; patch size {preprocessing['patch_size']}; "
        f"flow steps {model.steps}"
    )
    return model, preprocessing, checkpoint_path


def load_coordinates(path):
    """Read x,y centers in source order; extra columns and short lines are ignored."""
    coordinates = []
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.strip().replace(",", " ").replace("\t", " ").split()
            if len(parts) < 2:
                continue
            try:
                x, y = int(float(parts[0])), int(float(parts[1]))
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"Invalid x/y coordinates at {path}:{line_number}") from exc
            coordinates.append((x, y))
    return pd.DataFrame(coordinates, columns=["col", "row"])


class WholeImageSpots(Dataset):
    """Extract boundary-safe, normalized patches from level-0 image coordinates."""

    def __init__(self, image_path, coordinates, preprocessing):
        self.coordinates = coordinates.reset_index(drop=True)
        self.patch_size = int(preprocessing["patch_size"])
        if self.patch_size < 1:
            raise ValueError("Checkpoint patch_size must be positive")
        self.mean = torch.tensor(preprocessing["image_mean"], dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(preprocessing["image_std"], dtype=torch.float32)[:, None, None]
        if self.mean.shape[0] != 3 or self.std.shape[0] != 3 or torch.any(self.std <= 0):
            raise ValueError("Checkpoint image mean/std must contain three channels and positive std values")

        self.slide = None
        self.image = None
        image_path = Path(image_path).expanduser()
        try:
            import openslide
        except ImportError:
            openslide = None
        if openslide is not None:
            try:
                self.slide = openslide.OpenSlide(str(image_path))
            except openslide.OpenSlideUnsupportedFormatError:
                self.slide = None

        if self.slide is not None:
            self.width, self.height = self.slide.dimensions
        else:
            if image_path.suffix.lower() in (".tif", ".tiff", ".svs"):
                try:
                    self.image = tifffile.memmap(image_path, page=0, mode="r")
                except (ValueError, OSError):
                    with tifffile.TiffFile(image_path) as handle:
                        self.image = handle.pages[0].asarray()
                if self.image.ndim == 3 and self.image.shape[0] in (3, 4) and self.image.shape[-1] not in (3, 4):
                    self.image = np.moveaxis(self.image[:3], 0, -1)
            else:
                Image.MAX_IMAGE_PIXELS = None
                with Image.open(image_path) as handle:
                    self.image = np.asarray(handle.convert("RGB"))

            if self.image.dtype != np.uint8:
                raise ValueError("H&E must be 8-bit RGB, as in training; convert higher-bit-depth images explicitly")
            if self.image.ndim == 2:
                self.image = np.repeat(self.image[..., None], 3, axis=-1)
            if self.image.ndim != 3 or self.image.shape[-1] not in (3, 4):
                raise ValueError(f"Expected an RGB image; got shape {self.image.shape}")
            self.image = self.image[..., :3]
            self.height, self.width = self.image.shape[:2]

        if min(self.height, self.width) < self.patch_size:
            raise ValueError("Image is smaller than the checkpoint patch size")
        x = self.coordinates["col"].to_numpy(dtype=np.int64)
        y = self.coordinates["row"].to_numpy(dtype=np.int64)
        invalid = (x < 0) | (x >= self.width) | (y < 0) | (y >= self.height)
        if invalid.any():
            raise ValueError(
                f"{int(invalid.sum())} coordinates are outside the image; "
                "use level-0 pixels from this exact image"
            )

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, index):
        spot = self.coordinates.iloc[index]
        left = min(max(int(spot["col"]) - self.patch_size // 2, 0), self.width - self.patch_size)
        top = min(max(int(spot["row"]) - self.patch_size // 2, 0), self.height - self.patch_size)
        if self.slide is not None:
            patch = np.asarray(
                self.slide.read_region((left, top), 0, (self.patch_size, self.patch_size)).convert("RGB")
            )
        else:
            patch = self.image[top : top + self.patch_size, left : left + self.patch_size]
        image = torch.from_numpy(np.array(patch, copy=True)).permute(2, 0, 1).float().div_(255)
        return (image - self.mean) / self.std, index

    def close(self):
        if self.slide is not None:
            self.slide.close()
            self.slide = None
        self.image = None


def predict_image(model, image_path, coordinates, preprocessing, batch_size, device):
    """Return x,y and direct model outputs in original spot order."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    dataset = WholeImageSpots(image_path, coordinates, preprocessing)
    predictions = np.empty((len(coordinates), len(model.protein_names)), dtype=np.float32)
    try:
        loader = DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        with torch.inference_mode():
            for images, indices in loader:
                output = model(images.to(device, non_blocking=device.type == "cuda"))
                if output.ndim != 2 or output.shape != (len(indices), len(model.protein_names)):
                    raise ValueError("Prediction shape does not match batch size and checkpoint protein order")
                if not torch.isfinite(output).all():
                    raise FloatingPointError("Nonfinite predictions; final output was not written")
                predictions[indices.numpy()] = output.cpu().numpy()
    finally:
        dataset.close()

    result = pd.DataFrame(predictions, columns=model.protein_names)
    result.insert(0, "y", coordinates["row"].to_numpy(dtype=np.int64))
    result.insert(0, "x", coordinates["col"].to_numpy(dtype=np.int64))
    return result
