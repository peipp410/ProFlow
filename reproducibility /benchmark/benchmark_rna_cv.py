from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr, t
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dataset import BulkDataset  # the same as bulk training dataset
from scgpt.model import TransformerModel  
from scgpt.tokenizer import GeneVocab 
from scgpt.utils import load_pretrained 


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_scgpt_model(
    model_dir: str, checkpoint_path: str, device: torch.device
) -> nn.Module:
    model_dir = Path(model_dir)
    with open(model_dir / "args.json", "r") as handle:
        configs = json.load(handle)
    vocab = GeneVocab.from_file(model_dir / "vocab.json")
    for token in ("<pad>", "<cls>", "<eoc>"):
        if token not in vocab:
            vocab.append_token(token)
    vocab.set_default_index(vocab["<pad>"])
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=configs["embsize"],
        nhead=configs["nheads"],
        d_hid=configs["d_hid"],
        nlayers=configs["nlayers"],
        nlayers_cls=configs["n_layers_cls"],
        n_cls=1,
        vocab=vocab,
        dropout=configs["dropout"],
        pad_token=configs["pad_token"],
        pad_value=configs["pad_value"],
        do_mvc=True,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        explicit_zero_prob=False,
        use_fast_transformer=False,
        fast_transformer_backend="flash",
        pre_norm=False,
    )
    load_pretrained(
        model, torch.load(checkpoint_path, map_location=device), verbose=False
    )
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def encode_dataset(
    dataset: BulkDataset,
    rna_model: nn.Module,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    z_a_all = []
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        gene_ids = dataset.input_gene_ids[start:stop].to(device)
        rna_values = dataset.rna_expressions[start:stop].to(device)
        rna_mask = dataset.src_key_padding_mask[start:stop].to(device)
        z_a = rna_model.transformer_encoder(
            rna_model.encoder(gene_ids) + rna_model.value_encoder(rna_values),
            src_key_padding_mask=rna_mask,
        )[:, 0, :]
        z_a_all.append(z_a.cpu())
    return torch.cat(z_a_all)


def one_hot_labels(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    return F.one_hot(labels.long(), num_classes=n_classes).float()


def load_aligned_sample_groups(
    sample_group_path: str, barcodes: Sequence[str]
) -> np.ndarray:
    """Load cancer labels and align them to the RNA sample order."""
    group_table = pd.read_csv(sample_group_path, index_col=0)
    if group_table.shape[1] == 0:
        raise ValueError(f"No cancer-label column found in {sample_group_path}")

    barcode_index = pd.Index([str(barcode) for barcode in barcodes])
    group_table.index = group_table.index.astype(str)
    missing_barcodes = barcode_index.difference(group_table.index)
    if len(missing_barcodes) > 0:
        preview = ", ".join(missing_barcodes[:5])
        raise ValueError(
            f"{len(missing_barcodes)} RNA barcodes have no cancer label; "
            f"examples: {preview}"
        )

    groups = group_table.iloc[:, 0].reindex(barcode_index)
    if groups.isna().any():
        raise ValueError("Cancer labels contain missing values after barcode alignment")
    if groups.dtype == "object":
        groups = groups.astype("category").cat.codes
    return groups.to_numpy()


def load_aligned_protein_missing_mask(
    protein_path: str, barcodes: Sequence[str], protein_names: Sequence[str]
) -> np.ndarray:
    """Build a missing-value mask for the final ESM-2-filtered panel."""
    protein_table = pd.read_csv(protein_path, index_col=0)
    sample_index = pd.Index([str(barcode) for barcode in barcodes])
    protein_table.columns = protein_table.columns.astype(str)
    protein_table.index = protein_table.index.astype(str)
    protein_table = protein_table.loc[~protein_table.index.duplicated(keep="first")]

    missing_samples = sample_index.difference(protein_table.columns)
    if len(missing_samples) > 0:
        preview = ", ".join(missing_samples[:5])
        raise ValueError(
            f"{len(missing_samples)} RNA barcodes have no protein measurement; "
            f"examples: {preview}"
        )
    missing_proteins = pd.Index(protein_names).difference(protein_table.index)
    if len(missing_proteins) > 0:
        preview = ", ".join(missing_proteins[:5])
        raise ValueError(
            f"{len(missing_proteins)} final proteins are absent from the protein file; "
            f"examples: {preview}"
        )

    return (
        protein_table.loc[pd.Index(protein_names), sample_index]
        .isna()
        .T.to_numpy(dtype=bool)
    )


def mse_compute(
    pred: torch.Tensor, target: torch.Tensor, missing: torch.Tensor
) -> torch.Tensor:
    observed = (~missing).to(dtype=pred.dtype)
    return ((pred - target).pow(2) * observed).sum() / observed.sum().clamp_min(1.0)


class DeterministicProteinMLP(nn.Module):
    """Shared simple MLP predictor conditioned on RNA and ESM-2 features."""

    def __init__(self, rna_dim: int, n_classes: int, esm_dim: int):
        super().__init__()
        self.protein_projection = nn.Linear(esm_dim, rna_dim)
        input_dim = rna_dim + n_classes + esm_dim
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 4, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(
        self, z_a: torch.Tensor, cond: torch.Tensor, protein_emb: torch.Tensor
    ) -> torch.Tensor:
        # protein_features = self.protein_projection(protein_emb)
        context = torch.cat((z_a, cond), dim=-1).unsqueeze(1)
        context = context.expand(-1, protein_emb.size(0), -1)
        protein_emb = protein_emb.unsqueeze(0).expand(z_a.size(0), -1, -1)
        return self.head(torch.cat((context, protein_emb), dim=-1)).squeeze(-1)


def batch_ranges(
    indices: Sequence[int], batch_size: int, shuffle: bool, generator: torch.Generator
) -> Iterable[torch.Tensor]:
    order = torch.as_tensor(indices, dtype=torch.long)
    if shuffle:
        order = order[torch.randperm(order.numel(), generator=generator)]
    for start in range(0, order.numel(), batch_size):
        yield order[start : start + batch_size]


def metric_distribution(values: Sequence[float]) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {key: np.nan for key in ("mean", "max", "min", "median", "iqr")}
    q1, q3 = np.percentile(values, [25, 75])
    return {
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "iqr": float(q3 - q1),
    }


def train_latent_model(
    model: nn.Module,
    z_a: torch.Tensor,
    labels: torch.Tensor,
    target: torch.Tensor,
    missing: torch.Tensor,
    protein_emb: torch.Tensor,
    train_idx: Sequence[int],
    n_classes: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    test_idx: Sequence[int],
    protein_names: Sequence[str],
    method: str,
    fold: int,
) -> Tuple[nn.Module, List[Dict[str, object]]]:
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    z_a, labels, target, missing, protein_emb = [
        x.to(device) for x in (z_a, labels, target, missing, protein_emb)
    ]
    metric_history = []
    for epoch in range(epochs):
        running, n_batches = 0.0, 0
        for batch_idx in batch_ranges(train_idx, batch_size, True, generator):
            batch_idx = batch_idx.to(device)
            cond = one_hot_labels(labels[batch_idx], n_classes)
            prediction = model(z_a[batch_idx], cond, protein_emb)
            loss = mse_compute(prediction, target[batch_idx], missing[batch_idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu())
            n_batches += 1
        if epoch == epochs - 1 or (epoch + 1) % max(1, epochs // 5) == 0:
            print(
                f"      {model.__class__.__name__} "
                f"epoch {epoch + 1}/{epochs}: "
                f"loss={running / max(n_batches, 1):.5f}"
            )
        epoch_number = epoch + 1
        if epoch_number % 10 == 0:
            model.eval()
            train_prediction = predict_latent_model(
                model,
                z_a,
                labels,
                protein_emb,
                train_idx,
                n_classes,
                device,
                batch_size,
            )
            train_indices = torch.as_tensor(train_idx, dtype=torch.long, device=device)
            metric_history.append(
                report_protein_metrics(
                    train_prediction,
                    target[train_indices].detach().cpu().numpy(),
                    missing[train_indices].detach().cpu().numpy(),
                    protein_names,
                    method,
                    fold,
                    "Train",
                    epoch_number,
                    epochs,
                )
            )
            if epoch_number % 50 == 0:
                test_prediction = predict_latent_model(
                    model,
                    z_a,
                    labels,
                    protein_emb,
                    test_idx,
                    n_classes,
                    device,
                    batch_size,
                )
                test_indices = torch.as_tensor(
                    test_idx, dtype=torch.long, device=device
                )
                metric_history.append(
                    report_protein_metrics(
                        test_prediction,
                        target[test_indices].detach().cpu().numpy(),
                        missing[test_indices].detach().cpu().numpy(),
                        protein_names,
                        method,
                        fold,
                        "Test",
                        epoch_number,
                        epochs,
                    )
                )
            model.train()
    return model.eval(), metric_history


@torch.no_grad()
def predict_latent_model(
    model: nn.Module,
    z_a: torch.Tensor,
    labels: torch.Tensor,
    protein_emb: torch.Tensor,
    indices: Sequence[int],
    n_classes: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    z_a, labels, protein_emb = [x.to(device) for x in (z_a, labels, protein_emb)]
    outputs = []
    for batch_idx in batch_ranges(
        indices, batch_size, False, torch.Generator().manual_seed(0)
    ):
        batch_idx = batch_idx.to(device)
        outputs.append(
            model(
                z_a[batch_idx],
                one_hot_labels(labels[batch_idx], n_classes),
                protein_emb,
            ).cpu()
        )
    return torch.cat(outputs).numpy()


def fit_raw_regression(
    x_raw: np.ndarray,
    target: np.ndarray,
    missing: np.ndarray,
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    alpha_ridge: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    x_train = np.nan_to_num(
        x_raw[train_idx].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    x_test = np.nan_to_num(
        x_raw[test_idx].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    y_train, missing_train = target[train_idx], missing[train_idx]
    predictions = {
        "Ridge_raw_RNA": np.zeros((len(test_idx), target.shape[1])),
    }
    train_predictions = {
        "Ridge_raw_RNA": np.zeros((len(train_idx), target.shape[1])),
    }
    for protein_idx in range(target.shape[1]):
        valid = ~missing_train[:, protein_idx] & np.isfinite(y_train[:, protein_idx])
        fallback = (
            float(np.nanmean(y_train[valid, protein_idx])) if valid.any() else 0.0
        )
        if valid.sum() < 2:
            predictions["Ridge_raw_RNA"][:, protein_idx] = fallback
            train_predictions["Ridge_raw_RNA"][:, protein_idx] = fallback
            continue
        ridge = Ridge(alpha=alpha_ridge).fit(
            x_train[valid], y_train[valid, protein_idx]
        )
        predictions["Ridge_raw_RNA"][:, protein_idx] = ridge.predict(x_test)
        train_predictions["Ridge_raw_RNA"][:, protein_idx] = ridge.predict(x_train)
    return train_predictions, predictions


def correlation_records(
    predictions: np.ndarray,
    targets: np.ndarray,
    missing: np.ndarray,
    protein_names: Sequence[str],
    method: str,
    fold: int,
) -> List[Dict[str, object]]:
    records = []
    for j, protein in enumerate(protein_names):
        valid = (
            ~missing[:, j] & np.isfinite(predictions[:, j]) & np.isfinite(targets[:, j])
        )
        pred, true = predictions[valid, j].astype(float), targets[valid, j].astype(
            float
        )
        if len(true) < 3 or np.all(pred == pred[0]) or np.all(true == true[0]):
            pcc = pcc_p = scc = scc_p = np.nan
        else:
            pcc, pcc_p = pearsonr(pred, true)
            scc, scc_p = spearmanr(pred, true)
        mse = float(np.mean((pred - true) ** 2)) if len(true) else np.nan
        records.append(
            {
                "method": method,
                "fold": fold,
                "protein": protein,
                "pearson_r": pcc,
                "pearson_p": pcc_p,
                "spearman_rho": scc,
                "spearman_p": scc_p,
                "mse": mse,
            }
        )
    return records


def report_protein_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    missing: np.ndarray,
    protein_names: Sequence[str],
    method: str,
    fold: int,
    split: str,
    epoch: Optional[int] = None,
    epochs: Optional[int] = None,
) -> Dict[str, object]:
    records = correlation_records(
        predictions, targets, missing, protein_names, method, fold
    )
    pcc = metric_distribution([record["pearson_r"] for record in records])
    scc = metric_distribution([record["spearman_rho"] for record in records])
    mse_values = np.asarray([record["mse"] for record in records], dtype=float)
    mse_values = mse_values[np.isfinite(mse_values)]
    mse_mean = float(np.mean(mse_values)) if len(mse_values) else np.nan
    epoch_label = f"[Epoch {epoch}/{epochs}]" if epoch is not None else "[Fit]"
    print(
        f"[Fold {fold}][{method}][{split}]{epoch_label} "
        f"MSE={mse_mean:.4f} "
        f"| PCC(mean/max/min/median/IQR)="
        f"({pcc['mean']:.4f}/{pcc['max']:.4f}/{pcc['min']:.4f}/"
        f"{pcc['median']:.4f}/{pcc['iqr']:.4f}) "
        f"| SCC(mean/max/min/median/IQR)="
        f"({scc['mean']:.4f}/{scc['max']:.4f}/{scc['min']:.4f}/"
        f"{scc['median']:.4f}/{scc['iqr']:.4f})"
    )
    row: Dict[str, object] = {
        "method": method,
        "fold": fold,
        "split": split.lower(),
        "epoch": epoch,
        "mse_mean": mse_mean,
    }
    for prefix, summary in (("pearson_r", pcc), ("spearman_rho", scc)):
        for statistic, value in summary.items():
            row[f"{prefix}_{statistic}"] = value
    return row


def summarize_records(
    records: pd.DataFrame,
    output_dir: Path,
    pooled_predictions: Dict[str, np.ndarray],
    targets: np.ndarray,
    missing: np.ndarray,
    protein_names: Sequence[str],
) -> None:
    records.to_csv(output_dir / "fold_protein_metrics.csv", index=False)
    metrics = ["pearson_r", "spearman_rho", "mse"]
    protein_rows = []
    for (method, protein), group in records.groupby(["method", "protein"], sort=True):
        row = {"method": method, "protein": protein}
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            mean = float(np.mean(values)) if len(values) else np.nan
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            row[f"{metric}_mean"], row[f"{metric}_sd"] = mean, sd
            if len(values) > 1:
                half = t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values))
                row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = (
                    mean - half,
                    mean + half,
                )
            else:
                row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = np.nan, np.nan
        protein_rows.append(row)
    pd.DataFrame(protein_rows).to_csv(
        output_dir / "protein_cv_summary.csv", index=False
    )

    fold_rows = []
    for (method, fold), group in records.groupby(["method", "fold"], sort=True):
        row = {"method": method, "fold": fold}
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else np.nan
            row[f"{metric}_median"] = (
                float(np.median(values)) if len(values) else np.nan
            )
            row[f"{metric}_q1"] = (
                float(np.percentile(values, 25)) if len(values) else np.nan
            )
            row[f"{metric}_q3"] = (
                float(np.percentile(values, 75)) if len(values) else np.nan
            )
            row[f"{metric}_iqr"] = (
                row[f"{metric}_q3"] - row[f"{metric}_q1"] if len(values) else np.nan
            )
            row[f"{metric}_min"] = float(np.min(values)) if len(values) else np.nan
            row[f"{metric}_max"] = float(np.max(values)) if len(values) else np.nan
        fold_rows.append(row)
    fold_summary = pd.DataFrame(fold_rows)
    fold_summary.to_csv(output_dir / "fold_model_metrics.csv", index=False)
    model_rows = []
    for method, group in fold_summary.groupby("method", sort=True):
        row = {"method": method}
        for column in [c for c in fold_summary.columns if c not in {"method", "fold"}]:
            values = group[column].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            row[f"{column}_mean"] = float(np.mean(values)) if len(values) else np.nan
            row[f"{column}_sd"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            )
        model_rows.append(row)
    pd.DataFrame(model_rows).to_csv(output_dir / "model_cv_summary.csv", index=False)

    pooled_rows = []
    for method, prediction in pooled_predictions.items():
        pooled_rows.extend(
            correlation_records(
                prediction, targets, missing, protein_names, method, fold=0
            )
        )
    pooled_df = pd.DataFrame(pooled_rows).drop(columns=["fold"])
    pooled_df.to_csv(output_dir / "pooled_protein_metrics.csv", index=False)

    pooled_model_rows = []
    for method, group in pooled_df.groupby("method", sort=True):
        row = {"method": method}
        for metric in ["pearson_r", "spearman_rho", "mse"]:
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else np.nan
            row[f"{metric}_median"] = (
                float(np.median(values)) if len(values) else np.nan
            )
            row[f"{metric}_q1"] = (
                float(np.percentile(values, 25)) if len(values) else np.nan
            )
            row[f"{metric}_q3"] = (
                float(np.percentile(values, 75)) if len(values) else np.nan
            )
            row[f"{metric}_iqr"] = (
                row[f"{metric}_q3"] - row[f"{metric}_q1"] if len(values) else np.nan
            )
        pooled_model_rows.append(row)
    pd.DataFrame(pooled_model_rows).to_csv(
        output_dir / "pooled_model_metrics.csv", index=False
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--rna-path", required=True)
    parser.add_argument("--protein-path", required=True)
    parser.add_argument("--sample-group-path", required=True)
    parser.add_argument("--scgpt-model-path", required=True)
    parser.add_argument("--rna-checkpoint", required=True)
    parser.add_argument("--protein-emb-path", required=True)
    parser.add_argument("--output-dir", default="benchmark/results")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--baseline-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ridge-alpha", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(
        args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(
            f"CUDA device {device.index}: {props.name} | "
            f"total memory={props.total_memory / 2**20:.0f} MiB"
        )
    run_config = vars(args).copy()
    run_config["device"] = str(device)
    with open(output_dir / "run_config.json", "w") as handle:
        json.dump(run_config, handle, indent=2)
    dataset = BulkDataset(
        args.rna_path,
        args.protein_path,
        args.sample_group_path,
        args.scgpt_model_path,
        args.protein_emb_path,
    )
    protein_names = list(dataset.protein_names)
    n_proteins = len(protein_names)
    target = dataset.protein_exp.to_numpy(dtype=np.float32)
    # Rebuild the mask from the raw protein table so it is guaranteed to use
    # exactly the same ESM-2-filtered columns as ``target``.
    missing = load_aligned_protein_missing_mask(
        args.protein_path, dataset.barcode_tsv, protein_names
    )
    protein_emb = torch.as_tensor(dataset.protein_embedding, dtype=torch.float32)
    raw_rna = pd.read_parquet(args.rna_path).T.reindex(dataset.barcode_tsv)
    if raw_rna.isnull().all(axis=1).any():
        raise ValueError("RNA and protein sample identifiers are not aligned")
    raw_rna_values = (
        raw_rna.apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    raw_labels = load_aligned_sample_groups(args.sample_group_path, dataset.barcode_tsv)
    _, encoded_labels = np.unique(raw_labels, return_inverse=True)
    labels = torch.as_tensor(encoded_labels, dtype=torch.long)
    n_classes = int(labels.max().item() + 1)
    dimension_errors = []
    if target.shape != missing.shape:
        dimension_errors.append(
            f"target shape {target.shape} != missing-mask shape {missing.shape}"
        )
    if target.shape[1] != n_proteins:
        dimension_errors.append(
            f"target protein count {target.shape[1]} != embedding count {n_proteins}"
        )
    if len(dataset) != len(raw_rna_values):
        dimension_errors.append(
            f"dataset samples {len(dataset)} != raw RNA samples {len(raw_rna_values)}"
        )
    if len(dataset) != len(labels):
        dimension_errors.append(
            f"dataset samples {len(dataset)} != cancer labels {len(labels)}"
        )
    if dimension_errors:
        raise ValueError("Dimension check failed: " + "; ".join(dimension_errors))
    print(
        f"Samples: {len(dataset)} | Cancer classes: {n_classes} | Proteins: {n_proteins}"
    )
    rna_model = build_scgpt_model(args.scgpt_model_path, args.rna_checkpoint, device)
    z_a = encode_dataset(dataset, rna_model, args.encode_batch_size, device)
    target_t, missing_t = torch.as_tensor(target), torch.as_tensor(missing)
    splitter = KFold(
        n_splits=args.n_splits, shuffle=True, random_state=args.seed
    )
    fold_splits = list(splitter.split(np.arange(len(dataset))))
    all_records, split_records, metric_history = [], [], []
    pooled_predictions = {
        "Ridge_raw_RNA": np.full_like(target, np.nan, dtype=np.float64),
        "Deterministic_MLP": np.full_like(target, np.nan, dtype=np.float64),
    }
    for fold, (train_idx, test_idx) in enumerate(fold_splits, start=1):
        fold_seed = args.seed + fold - 1
        set_seed(fold_seed)
        print(
            f"\n===== Fold {fold}/{args.n_splits}: "
            f"train={len(train_idx)}, test={len(test_idx)} ====="
        )
        split_records += [
            {
                "sample_index": int(i),
                "sample": dataset.barcode_tsv[i],
                "fold": fold,
                "split": "train",
            }
            for i in train_idx
        ]
        split_records += [
            {
                "sample_index": int(i),
                "sample": dataset.barcode_tsv[i],
                "fold": fold,
                "split": "test",
            }
            for i in test_idx
        ]
        raw_train_predictions, raw_predictions = fit_raw_regression(
            raw_rna_values,
            target,
            missing,
            train_idx,
            test_idx,
            args.ridge_alpha,
        )
        for method, prediction in raw_predictions.items():
            metric_history.append(
                report_protein_metrics(
                    raw_train_predictions[method],
                    target[train_idx],
                    missing[train_idx],
                    protein_names,
                    method,
                    fold,
                    "Train",
                )
            )
            metric_history.append(
                report_protein_metrics(
                    prediction,
                    target[test_idx],
                    missing[test_idx],
                    protein_names,
                    method,
                    fold,
                    "Test",
                )
            )
            pooled_predictions[method][test_idx] = prediction
            all_records += correlation_records(
                prediction,
                target[test_idx],
                missing[test_idx],
                protein_names,
                method,
                fold,
            )
        mlp, mlp_metric_history = train_latent_model(
            DeterministicProteinMLP(512, n_classes, protein_emb.size(-1)),
            z_a,
            labels,
            target_t,
            missing_t,
            protein_emb,
            train_idx,
            n_classes,
            device,
            args.baseline_epochs,
            args.batch_size,
            args.lr,
            args.weight_decay,
            fold_seed,
            test_idx,
            protein_names,
            "Deterministic_MLP",
            fold,
        )
        metric_history.extend(mlp_metric_history)
        mlp_prediction = predict_latent_model(
            mlp, z_a, labels, protein_emb, test_idx, n_classes, device, args.batch_size
        )
        pooled_predictions["Deterministic_MLP"][test_idx] = mlp_prediction
        all_records += correlation_records(
            mlp_prediction,
            target[test_idx],
            missing[test_idx],
            protein_names,
            "Deterministic_MLP",
            fold,
        )
        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)
        predictions = {
            **raw_predictions,
            "Deterministic_MLP": mlp_prediction,
        }
        for method, prediction in predictions.items():
            pd.DataFrame(
                prediction,
                index=np.asarray(dataset.barcode_tsv)[test_idx],
                columns=protein_names,
            ).to_csv(fold_dir / f"{method}_predictions.csv")
        pd.DataFrame(metric_history).to_csv(
            output_dir / "epoch_metric_summary.csv", index=False
        )
    records_df = pd.DataFrame(all_records)
    summarize_records(
        records_df, output_dir, pooled_predictions, target, missing, protein_names
    )
    pd.DataFrame(split_records).to_csv(output_dir / "cv_splits.csv", index=False)
    pd.DataFrame({"protein": protein_names}).to_csv(
        output_dir / "protein_panel.csv", index=False
    )
    print(f"\nBenchmark complete. Results written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
