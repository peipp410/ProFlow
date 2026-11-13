"""Train an image-only protein-expression model with a shared prediction head."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from accelerate import Accelerator
from scgpt.model import TransformerModel
from scgpt.tokenizer import GeneVocab
from scgpt.utils import load_pretrained
from conch.open_clip_custom import create_model_from_pretrained
from pretrained_model import ConditionalFlowNet, HistFlowRNA, load_pretrained_model


class SpatialDataset(Dataset):
    """Load image patches and aligned protein expression/embedding targets."""

    PROTEIN_NAMES = [
        "PDCD1", "VSIR", "CD274", "LAG3", "FCGR3A", "GZMB", "CD163", "CD4",
        "MS4A1", "CD8A", "CD3E", "SDC1", "HLA-DRA", "ITGAX", "CD68", "PTPRCRA",
        "PCNA", "PTPRCRO", "MKI67", "CTNNB1", "PECAM1", "PTEN", "KRT19", "VIM",
        "ACTA2", "PTPRC", "CDH1",
    ]

    def __init__(self, barcodes, image_path, spatial_pos_path, protein_exp_path,
                 protein_emb_path, augment=False, patch_size=256):
        with tifffile.TiffFile(image_path) as tif:
            self.image = tif.pages[0].asarray()
        self.positions = pd.read_parquet(spatial_pos_path)
        self.protein_expression = pd.read_parquet(protein_exp_path)
        if self.protein_expression.shape[1] != len(self.PROTEIN_NAMES):
            raise ValueError(f"Expected {len(self.PROTEIN_NAMES)} protein columns; got {self.protein_expression.shape[1]}")
        self.protein_expression.columns = self.PROTEIN_NAMES
        embeddings = torch.load(protein_emb_path, map_location="cpu", weights_only=False)
        names = [name for name in self.protein_expression.columns if name in embeddings]
        if not names:
            raise ValueError("No protein names from the expression table occur in the embedding file.")
        self.protein_names = names
        self.protein_embeddings = torch.stack([torch.as_tensor(embeddings[name]).float() for name in names])
        self.protein_expression = self.protein_expression[names]
        self.barcodes = list(barcodes)
        self.patch_size = patch_size
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ]) if augment else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])

    def __len__(self):
        return len(self.barcodes)

    def __getitem__(self, index):
        barcode = self.barcodes[index]
        row = self.positions.loc[barcode]
        y, x = int(row["row"]), int(row["col"])
        size = self.patch_size
        height, width = self.image.shape[:2]
        y0 = min(max(y - size // 2, 0), height - size)
        x0 = min(max(x - size // 2, 0), width - size)
        patch = self.image[y0:y0 + size, x0:x0 + size]
        if patch.shape[:2] != (size, size):
            raise IndexError(f"Patch for {barcode} is outside image bounds: {patch.shape}")
        if patch.ndim == 2:
            patch = np.repeat(patch[..., None], 3, axis=2)
        if patch.shape[2] > 3:
            patch = patch[..., :3]
        expression = self.protein_expression.loc[barcode].to_numpy(dtype=np.float32)
        return self.transform(Image.fromarray(patch)), torch.from_numpy(expression), self.protein_embeddings


def integrate_flow(flow, z, steps, condition):
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((z.size(0), 1), i / steps, device=z.device, dtype=z.dtype)
        k1 = flow(z, t, condition)
        k2 = flow(z + dt * k1 / 2, t + dt / 2, condition)
        k3 = flow(z + dt * k2 / 2, t + dt / 2, condition)
        k4 = flow(z + dt * k3, t + dt, condition)
        z = z + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return z


class ImageToProteinModel(nn.Module):
    def __init__(self, hist_flow_rna, rna_flow, spot_dim, protein_dim):
        super().__init__()
        self.hist_flow_rna = hist_flow_rna
        self.rna_flow = rna_flow
        self.protein_projection = nn.Linear(protein_dim, spot_dim)
        self.trans_dim = nn.Linear(spot_dim * 2, spot_dim)
        # Shared weights across all proteins; protein embeddings distinguish targets.
        self.shared_prediction_head = nn.Sequential(
            nn.Linear(spot_dim, spot_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(spot_dim // 2, 1),
        )

    def forward(self, image, protein_embeddings, steps):
        image_latent, pred_rna_emb, _, celltype_prob = self.hist_flow_rna.encode_image_to_rna_emb(image, steps=steps)
        predicted_protein_embedding = integrate_flow(self.rna_flow, pred_rna_emb, steps, celltype_prob)
        protein_features = self.protein_projection(protein_embeddings)
        image_features = image_latent.unsqueeze(1).expand(-1, protein_features.size(1), -1)
        combined = self.trans_dim(torch.cat((protein_features, image_features), dim=-1))
        return predicted_protein_embedding, self.shared_prediction_head(combined).squeeze(-1)


class RBF(nn.Module):
    def __init__(self, bandwidth=1.0, kernels=5, multiplier=2.0):
        super().__init__()
        self.register_buffer("multipliers", multiplier ** (torch.arange(kernels) - kernels // 2))
        self.bandwidth = nn.Parameter(torch.tensor(float(bandwidth)))

    def forward(self, x):
        distances = torch.cdist(x, x).square()
        scale = (self.bandwidth * self.multipliers.to(x.device)).clamp_min(1e-8)
        return torch.exp((-distances.unsqueeze(0) / scale[:, None, None]).clamp(-32, 0)).sum(0)


def mmd_loss(x, y, kernel):
    joined = torch.cat((x, y))
    gram = kernel(joined)
    nx, ny = len(x), len(y)
    xx, yy, xy = gram[:nx, :nx], gram[nx:, nx:], gram[:nx, nx:]
    return ((xx.sum() - xx.diagonal().sum()) / max(nx * (nx - 1), 1)
            + (yy.sum() - yy.diagonal().sum()) / max(ny * (ny - 1), 1)
            - 2 * xy.mean())


def build_scgpt_model(model_dir):
    model_dir = Path(model_dir)
    vocab = GeneVocab.from_file(model_dir / "vocab.json")
    for token in ("<pad>", "<cls>", "<eoc>"):
        if token not in vocab:
            vocab.append_token(token)
    vocab.set_default_index(vocab["<pad>"])
    config = json.loads((model_dir / "args.json").read_text())
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
    load_pretrained(model, torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=False), verbose=False)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", required=True, help="Parquet with row/col and split columns, indexed by barcode")
    parser.add_argument("--image", required=True)
    parser.add_argument("--train-proteins", required=True)
    parser.add_argument("--test-proteins", required=True)
    parser.add_argument("--scgpt-model-dir", required=True)
    parser.add_argument("--protein-embeddings", required=True)
    parser.add_argument("--coach-checkpoint", required=True)
    parser.add_argument("--reference-embeddings", required=True)
    parser.add_argument("--hist-flow-checkpoint", required=True)
    parser.add_argument("--output", default="results/pretrain_protein_pcc_onlyhe.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--flow-steps", type=int, default=60)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def correlations(predictions, targets):
    results = []
    for p, t in zip(predictions.T, targets.T):
        if np.std(p) == 0 or np.std(t) == 0:
            results.append((0.0, 0.0))
        else:
            results.append((pearsonr(p, t).statistic, spearmanr(p, t).statistic))
    return np.asarray(results)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    accelerator = Accelerator()
    positions = pd.read_parquet(args.positions)
    train_barcodes = positions.index[positions["split"] == "train"].tolist()
    test_barcodes = positions.index[positions["split"] == "test"].tolist()
    common = dict(image_path=args.image, spatial_pos_path=args.positions,
                  protein_emb_path=args.protein_embeddings)
    train_data = SpatialDataset(train_barcodes, protein_exp_path=args.train_proteins,
                                augment=True, **common)
    test_data = SpatialDataset(test_barcodes, protein_exp_path=args.test_proteins,
                               augment=False, **common)
    if train_data.protein_names != test_data.protein_names:
        raise ValueError("Training and test protein tables resolve to different protein columns/order.")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    rna_model = build_scgpt_model(args.scgpt_model_dir)
    conch, _ = create_model_from_pretrained("conch_ViT-B-16", args.coach_checkpoint, force_image_size=256)
    class_embeddings = pd.read_csv(args.reference_embeddings, index_col=0).values
    image_flow = ConditionalFlowNet(source_dim=512, to_dim=512, cond_dim=0, hidden_dim=1024)
    hist_flow = HistFlowRNA(rna_model=rna_model, hist_model=conch.visual,
                            class_embeddings=class_embeddings, flow_model=image_flow)
    hist_flow = load_pretrained_model(hist_flow, args.hist_flow_checkpoint, device="cpu")
    for parameter in hist_flow.parameters():
        parameter.requires_grad_(False)
    protein_flow = ConditionalFlowNet(512, 512, 40)
    model = ImageToProteinModel(hist_flow, protein_flow, 512,
                                int(train_data.protein_embeddings.shape[-1]))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    kernel = RBF()
    model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for images, targets, protein_embeddings in train_loader:
            optimizer.zero_grad(set_to_none=True)
            _, predictions = model(images, protein_embeddings, args.flow_steps)
            loss = F.mse_loss(predictions, targets) + mmd_loss(predictions.detach(), targets.detach(), kernel)
            accelerator.backward(loss)
            optimizer.step()
            total += loss.detach().item()
        accelerator.print(f"Epoch {epoch + 1}/{args.epochs} train loss: {total / max(len(train_loader), 1):.5f}")
        if (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            model.eval()
            preds, targets_all = [], []
            with torch.no_grad():
                for images, targets, protein_embeddings in test_loader:
                    _, output = model(images, protein_embeddings, args.flow_steps)
                    preds.append(output)
                    targets_all.append(targets)
            preds = accelerator.gather_for_metrics(torch.cat(preds)).cpu().numpy()
            targets_all = accelerator.gather_for_metrics(torch.cat(targets_all)).cpu().numpy()
            stats = correlations(preds, targets_all)
            accelerator.print(f"Epoch {epoch + 1}: mean Pearson={stats[:, 0].mean():.4f}, mean Spearman={stats[:, 1].mean():.4f}")
            if epoch + 1 == args.epochs and accelerator.is_main_process:
                output = Path(args.output)
                output.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"protein": train_data.protein_names,
                              "PCC": stats[:, 0], "Spearman": stats[:, 1]}).to_csv(output, index=False)


if __name__ == "__main__":
    main()
