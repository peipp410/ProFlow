"""Predict protein expression with the simulation-free flow-matching model.

The model and preprocessing in this file mirror ``train.py``.  Inference
starts from the RNA scGPT CLS embedding, integrates the learned velocity
field with RK4, and decodes the terminal embedding for every protein.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate


class ConditionalFlowNet(nn.Module):
    """The conditional velocity network used by ``train.py``."""

    def __init__(self, prot_dim: int, rna_dim: int, cond_dim: int, hidden_dim: int = 1024):
        super().__init__()
        del rna_dim  # retained for API compatibility with train.py
        self.net = nn.Sequential(
            nn.Linear(prot_dim + cond_dim + 1, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, prot_dim),
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_t, cond_vec, t], dim=-1))


class _ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, expansion: int = 2, dropout: float = 0.05):
        super().__init__()
        expanded_dim = hidden_dim * expansion
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, expanded_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(expanded_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.residual_scale = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dropout1(self.activation(self.fc1(self.norm(x))))
        x = self.dropout2(self.fc2(x))
        return residual + self.residual_scale * x


class PredictModelWithCLassesDeepResidual(nn.Module):
    """Exact shared ESM2-conditioned prediction head used by ``train.py``."""

    def __init__(
        self,
        spot_dim: int,
        protein_dim: int,
        num_classes: int,
        num_proteins: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.05,
        expansion: int = 2,
        protein_names: List[str] | None = None,
    ):
        super().__init__()
        self.num_proteins = num_proteins
        self.protein_names = list(protein_names) if protein_names is not None else None
        self.protein_projection = nn.Linear(protein_dim, spot_dim)
        context_dim = spot_dim + num_classes
        self.input_projection = nn.Sequential(
            nn.Linear(context_dim + spot_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.residual_blocks = nn.ModuleList(
            [_ResidualMLPBlock(hidden_dim, expansion, dropout) for _ in range(num_layers)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, cell_embeddings_protein: torch.Tensor, cond_label: torch.Tensor, protein_emb: torch.Tensor
    ) -> torch.Tensor:
        if cell_embeddings_protein.ndim != 2 or cond_label.ndim != 2:
            raise ValueError("cell_embeddings_protein and cond_label must be 2-D")
        if protein_emb.ndim == 2:
            protein_emb = protein_emb.unsqueeze(0).expand(cell_embeddings_protein.size(0), -1, -1)
        elif protein_emb.ndim != 3:
            raise ValueError("protein_emb must have shape (P,D) or (B,P,D)")
        protein_projected = self.protein_projection(protein_emb)
        context = torch.cat((cell_embeddings_protein, cond_label), dim=-1)
        context = context.unsqueeze(1).expand(-1, protein_projected.size(1), -1)
        features = self.input_projection(torch.cat((protein_projected, context), dim=-1))
        for block in self.residual_blocks:
            features = block(features)
        return self.prediction_head(self.output_norm(features)).squeeze(-1)


def build_scgpt_model(model_dir: Union[str, Path], state_path: Union[str, Path]) -> TransformerModel:
    from scgpt.model import TransformerModel
    from scgpt.tokenizer import GeneVocab
    from scgpt.utils import load_pretrained

    model_dir = Path(model_dir)
    vocab = GeneVocab.from_file(model_dir / "vocab.json")
    for token in ("<pad>", "<cls>", "<eoc>"):
        if token not in vocab:
            vocab.append_token(token)
    vocab.set_default_index(vocab["<pad>"])
    with open(model_dir / "args.json", "r", encoding="utf-8") as handle:
        config = json.load(handle)
    model = TransformerModel(
        ntoken=len(vocab), d_model=config["embsize"], nhead=config["nheads"],
        d_hid=config["d_hid"], nlayers=config["nlayers"],
        nlayers_cls=config["n_layers_cls"], n_cls=1, vocab=vocab,
        dropout=config["dropout"], pad_token=config["pad_token"],
        pad_value=config["pad_value"], do_mvc=True, do_dab=False,
        use_batch_labels=False, domain_spec_batchnorm=False,
        explicit_zero_prob=False, use_fast_transformer=False,
        fast_transformer_backend="flash", pre_norm=False,
    )
    load_pretrained(model, torch.load(state_path, map_location="cpu"), verbose=False)
    return model


def read_protein_names(path: Union[str, Path]) -> List[str]:
    path = Path(path)
    if path.suffix.lower() in {".pkl", ".pickle"}:
        with open(path, "rb") as handle:
            return [str(x) for x in pickle.load(handle)]
    if path.suffix.lower() in {".csv", ".tsv"}:
        frame = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
        for col in ("protein", "protein_name", "name"):
            if col in frame.columns:
                return frame[col].astype(str).tolist()
        return frame.iloc[:, 0].astype(str).tolist()
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_state_dict(obj: object) -> Dict[str, torch.Tensor]:
    if isinstance(obj, Mapping):
        for key in ("state_dict", "model_state_dict"):
            if key in obj and isinstance(obj[key], Mapping):
                return dict(obj[key])
        if all(isinstance(key, str) for key in obj):
            return dict(obj)
    raise ValueError("Unsupported checkpoint format")


def rk4_flow(flow_model: nn.Module, z0: torch.Tensor, cond: torch.Tensor, steps: int) -> torch.Tensor:
    if steps < 1:
        raise ValueError("steps must be positive")
    dt = 1.0 / steps
    z = z0
    for step in range(steps):
        t = torch.full((z.size(0), 1), step / steps, device=z.device, dtype=z.dtype)
        k1 = flow_model(z, t, cond)
        k2 = flow_model(z + 0.5 * dt * k1, t + 0.5 * dt, cond)
        k3 = flow_model(z + 0.5 * dt * k2, t + 0.5 * dt, cond)
        k4 = flow_model(z + dt * k3, t + dt, cond)
        z = z + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return z


def infer_dimensions(flow_state: Dict[str, torch.Tensor], pred_state: Dict[str, torch.Tensor]):
    flow_in = flow_state["net.0.weight"].shape[1]
    flow_dim = flow_state["net.4.weight"].shape[0]
    flow_classes = flow_in - flow_dim - 1
    pred_spot = pred_state["protein_projection.weight"].shape[0]
    pred_protein_dim = pred_state["protein_projection.weight"].shape[1]
    pred_hidden = pred_state["input_projection.0.weight"].shape[0]
    pred_classes = pred_state["input_projection.0.weight"].shape[1] - 2 * pred_spot
    layers = sorted({int(k.split(".")[1]) for k in pred_state if k.startswith("residual_blocks.") and k.endswith(".norm.weight")})
    num_layers = len(layers)
    expansion = pred_state["residual_blocks.0.fc1.weight"].shape[0] // pred_hidden
    if flow_dim != pred_spot or flow_classes != pred_classes:
        raise ValueError(f"Checkpoint dimensions disagree: flow=({flow_dim},{flow_classes}), predictor=({pred_spot},{pred_classes})")
    return flow_dim, pred_protein_dim, pred_classes, pred_hidden, num_layers, expansion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rna_path", default="data/rna_only.parquet")
    parser.add_argument("--sample_group_path", default="data/new_data/sample_rna_filtered.csv")
    parser.add_argument("--protein_names_path", default="data/protein_name.pkl")
    parser.add_argument("--protein_emb_path", default="data/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2_new.pt")
    parser.add_argument("--scgpt_model_dir", default="ckp/scgpt_model")
    parser.add_argument("--rna_scgpt_state", default="ckp/scgpt_model/rna_state_epoch_5.pt")
    parser.add_argument("--flow_ckpt", default="=ckp/flow_model_new.pt")
    parser.add_argument("--pred_ckpt", default="=ckp/pred_model_new.pt")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default="auto",
        help="Inference device. 'auto' selects an available CUDA GPU, otherwise CPU.",
    )
    parser.add_argument("--zero_based_groups", action="store_true", help="Treat sample groups as 0-based; training data is 1-based by default.")
    parser.add_argument("--out_csv", default="results/bulk_protein_prediction.csv")
    args = parser.parse_args()

    # Match the seed used by train.py.  TestDataset performs stochastic gene
    # sampling when an RNA profile is longer than the tokenizer limit.
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Resolve the device at runtime so the same command works on GPU and CPU hosts.
    cuda_available = torch.cuda.is_available()
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not cuda_available:
            raise RuntimeError("--device cuda was requested, but no CUDA device is available")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if cuda_available else "cpu")
    if device.type == "cuda":
        print(f"Using CUDA device {torch.cuda.current_device()}: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("No CUDA GPU is available; using CPU for inference.")
    protein_names = read_protein_names(args.protein_names_path)
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from dataset import TestDataset

    test_rna_dataset = TestDataset(
        rna_path=args.rna_path, sample_group_path=args.sample_group_path,
        scgpt_model_path=args.scgpt_model_dir, protein_emb_path=args.protein_emb_path,
        protein_names=protein_names,
    )
    flow_state = resolve_state_dict(torch.load(args.flow_ckpt, map_location="cpu"))
    pred_state = resolve_state_dict(torch.load(args.pred_ckpt, map_location="cpu"))
    spot_dim, protein_dim, num_classes, hidden_dim, num_layers, expansion = infer_dimensions(flow_state, pred_state)
    if len(protein_names) != test_rna_dataset.protein_embedding.shape[0]:
        raise ValueError("Protein name count does not match embeddings found in protein_emb_path")

    rna_model = build_scgpt_model(args.scgpt_model_dir, args.rna_scgpt_state).to(device)
    flow_model = ConditionalFlowNet(spot_dim, spot_dim, num_classes).to(device)
    pred_model = PredictModelWithCLassesDeepResidual(
        spot_dim, protein_dim, num_classes, len(protein_names), hidden_dim=hidden_dim,
        num_layers=num_layers, expansion=expansion, protein_names=protein_names,
    ).to(device)
    flow_model.load_state_dict(flow_state, strict=True)
    pred_model.load_state_dict(pred_state, strict=True)
    flow_model.eval()
    pred_model.eval()

    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        return default_collate(batch) if batch else None

    loader = DataLoader(test_rna_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    sample_ids: List[str] = []
    predictions: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            input_gene_ids = batch["input_gene_ids"].to(device)
            expressions = batch["rna_expressions"].to(device)
            padding = batch["src_key_padding_mask"].to(device)
            z0 = rna_model.transformer_encoder(
                rna_model.encoder(input_gene_ids) + rna_model.value_encoder(expressions),
                src_key_padding_mask=padding,
            )[:, 0, :]
            raw_labels = batch["sample_group"].to(device).long()
            labels = raw_labels if args.zero_based_groups else raw_labels - 1
            if labels.numel() and (labels.min() < 0 or labels.max() >= num_classes):
                raise ValueError(f"Sample groups must map to [0, {num_classes - 1}], got {labels.min().item()}..{labels.max().item()}")
            cond = F.one_hot(labels, num_classes=num_classes).float()
            z_terminal = rk4_flow(flow_model, z0, cond, args.steps)
            predictions.append(pred_model(z_terminal, cond, batch["protein_emb"].to(device)).cpu())
            sample_ids.extend(str(x) for x in batch["barcode"])

    if not predictions:
        raise RuntimeError("No valid RNA samples were available for prediction")
    output = pd.DataFrame(torch.cat(predictions).numpy(), index=sample_ids, columns=protein_names)
    output.index.name = "sample"
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path)
    print(f"Saved {output.shape[0]} samples x {output.shape[1]} proteins to {out_path}")


if __name__ == "__main__":
    main()
