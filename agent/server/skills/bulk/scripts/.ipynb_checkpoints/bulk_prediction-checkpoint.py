import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from scgpt.model import TransformerModel
from scgpt.tokenizer import GeneVocab
from scgpt.utils import load_pretrained
from torch.utils.data._utils.collate import default_collate


def read_protein_names(path: Union[str, Path]) -> List[str]:
    path = Path(path)
    if path.suffix.lower() in (".pkl", ".pickle"):
        import pickle

        with open(path, "rb") as f:
            names = pickle.load(f)
        return [str(x) for x in names]

    if path.suffix.lower() in (".csv", ".tsv"):
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        if df.shape[1] == 1:
            return df.iloc[:, 0].astype(str).to_list()
        for col in ("protein", "protein_name", "name"):
            if col in df.columns:
                return df[col].astype(str).to_list()
        return df.iloc[:, 0].astype(str).to_list()

    names = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                names.append(s)
    return names


def build_scgpt_model(scgpt_model_dir: Union[str, Path], pretrained_state_path: Union[str, Path]) -> TransformerModel:
    scgpt_model_dir = Path(scgpt_model_dir)
    vocab_file = scgpt_model_dir / "vocab.json"
    model_config_file = scgpt_model_dir / "args.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in ("<pad>", "<cls>", "<eoc>"):
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab["<pad>"])

    with open(model_config_file, "r") as f:
        model_configs = json.load(f)

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=model_configs["embsize"],
        nhead=model_configs["nheads"],
        d_hid=model_configs["d_hid"],
        nlayers=model_configs["nlayers"],
        nlayers_cls=model_configs["n_layers_cls"],
        n_cls=1,
        vocab=vocab,
        dropout=model_configs["dropout"],
        pad_token=model_configs["pad_token"],
        pad_value=model_configs["pad_value"],
        do_mvc=True,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        explicit_zero_prob=False,
        use_fast_transformer=False,
        fast_transformer_backend="flash",
        pre_norm=False,
    )
    state = torch.load(pretrained_state_path, map_location="cpu")
    load_pretrained(model, state, verbose=False)
    return model


class ConditionalFlowNet(nn.Module):
    def __init__(self, prot_dim: int, rna_dim: int, cond_dim: int, hidden_dim: int = 1024):
        super().__init__()
        input_dim = prot_dim + cond_dim + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, prot_dim),
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        ht = torch.cat([z_t, cond_vec, t], dim=-1)
        return self.net(ht)


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels, num_classes=num_classes).float()


def rk4_flow(flow_model: nn.Module, z0: torch.Tensor, cond: torch.Tensor, steps: int) -> torch.Tensor:
    dt = 1.0 / steps
    z = z0
    for step in range(steps):
        t_val = step / steps
        t_tensor = torch.full((z.size(0), 1), t_val, device=z.device, dtype=z.dtype)
        k1 = flow_model(z, t_tensor, cond)
        k2 = flow_model(z + 0.5 * dt * k1, t_tensor + 0.5 * dt, cond)
        k3 = flow_model(z + 0.5 * dt * k2, t_tensor + 0.5 * dt, cond)
        k4 = flow_model(z + dt * k3, t_tensor + dt, cond)
        z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return z


def resolve_state_dict(obj: object) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
        if all(isinstance(k, str) for k in obj.keys()):
            return obj
    raise ValueError("Unsupported checkpoint format for model weights")


def main():
    parser = argparse.ArgumentParser(prog="bulk_prediction.py")
    parser.add_argument("--rna_path", default="/home/peijiazheng/agent/server/skills/bulk/data/test_all_rna.parquet")
    parser.add_argument("--sample_group_path", default="/home/peijiazheng/agent/server/skills/bulk/data/test_all_samples.csv")
    parser.add_argument("--protein_names_path", default="/home/peijiazheng/agent/server/skills/bulk/data/protein_name.pkl")
    parser.add_argument("--protein_emb_path", default="/home/peijiazheng/agent/server/skills/bulk/data/ESM2_embedding.pt")

    parser.add_argument("--scgpt_model_dir", default="/home/peijiazheng/agent/server/skills/bulk/ckp/scgpt_model/")
    parser.add_argument("--rna_scgpt_state", default="/home/peijiazheng/agent/server/skills/bulk/ckp/scgpt_model/best_model.pt")

    parser.add_argument("--flow_ckpt", default="/home/peijiazheng/agent/server/skills/bulk/ckp/flow_model.pt")
    parser.add_argument("--pred_ckpt", default="/home/peijiazheng/agent/server/skills/bulk/ckp/pred_model.pt")

    parser.add_argument("--num_classes", type=int, default=32)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_csv", default="/home/peijiazheng/agent/server/skills/bulk/results/bulk_protein_prediction.csv")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    protein_names = read_protein_names(args.protein_names_path)

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from dataset import TestDataset
    from encoders import PredictModelWithCLasses

    dataset = TestDataset(
        rna_path=args.rna_path,
        sample_group_path=args.sample_group_path,
        scgpt_model_path=args.scgpt_model_dir,
        protein_emb_path=args.protein_emb_path,
        protein_names=protein_names,
    )

    scgpt_model = build_scgpt_model(args.scgpt_model_dir, args.rna_scgpt_state).to(device)
    scgpt_model.eval()
    for p in scgpt_model.parameters():
        p.requires_grad = False

    flow_model = ConditionalFlowNet(512, 512, args.num_classes).to(device)
    pred_model = PredictModelWithCLasses(
        512, int(dataset.protein_embedding.shape[1]), args.num_classes, len(protein_names), protein_names=protein_names
    ).to(device)

    flow_state = resolve_state_dict(torch.load(args.flow_ckpt, map_location="cpu"))
    pred_state = resolve_state_dict(torch.load(args.pred_ckpt, map_location="cpu"))
    flow_model.load_state_dict(flow_state, strict=True)
    pred_model.load_state_dict(pred_state, strict=True)
    flow_model.eval()
    pred_model.eval()

    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        return default_collate(batch)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    all_ids: List[str] = []
    all_preds: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            input_gene_ids = batch["input_gene_ids"].to(device)
            rna_expressions = batch["rna_expressions"].to(device)
            src_key_padding_mask = batch["src_key_padding_mask"].to(device)

            gene_emb = scgpt_model.encoder(input_gene_ids)
            expr_emb = scgpt_model.value_encoder(rna_expressions)
            z0 = scgpt_model.transformer_encoder(gene_emb + expr_emb, src_key_padding_mask=src_key_padding_mask)[:, 0, :]

            labels_raw = batch["sample_group"].to(device).long()
            labels = labels_raw if int(labels_raw.min().item()) == 0 else (labels_raw - 1)
            cond_label = one_hot(labels, num_classes=args.num_classes).to(device)
            zT = rk4_flow(flow_model, z0, cond_label, steps=args.steps)

            protein_emb_batch = batch["protein_emb"].to(device)
            preds = pred_model(zT, cond_label, protein_emb_batch)

            all_ids.extend([str(x) for x in batch["barcode"]])
            all_preds.append(preds.cpu())

    pred_mat = torch.cat(all_preds, dim=0).numpy()
    out_df = pd.DataFrame(pred_mat, index=all_ids, columns=protein_names)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path)
    print(str(out_path))


if __name__ == "__main__":
    main()
