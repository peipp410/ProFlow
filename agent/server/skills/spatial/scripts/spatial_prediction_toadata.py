import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
from torch.utils.data import DataLoader
from tqdm import tqdm

import anndata as ad

from conch.open_clip_custom import create_model_from_pretrained
from dataset import InferenceProteinDataset
from pretrained_model import HistFlowRNA, ConditionalFlowNet, load_pretrained_model
from scgpt.model import TransformerModel
from scgpt.tokenizer import GeneVocab
from scgpt.utils import load_pretrained


def integrate_flow(flow_model, z_init, steps=60, cond_vec=None):
    dt = 1.0 / steps
    z = z_init.clone()
    for step in range(steps):
        t_val = step / steps
        t_tensor = torch.full((z.size(0), 1), t_val, device=z.device)
        k1 = flow_model(z, t_tensor, cond_vec)
        k2 = flow_model(z + 0.5 * dt * k1, t_tensor + 0.5 * dt, cond_vec)
        k3 = flow_model(z + 0.5 * dt * k2, t_tensor + 0.5 * dt, cond_vec)
        k4 = flow_model(z + dt * k3, t_tensor + dt, cond_vec)
        z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return z


class ImageToProteinModel(nn.Module):
    def __init__(self, hist_flow_rna, rna_flow, spot_dim, protein_dim, num_proteins):
        super().__init__()
        self.hist_flow_rna = hist_flow_rna
        self.rna_flow = rna_flow
        self.num_proteins = num_proteins
        self.protein_projection = nn.Linear(protein_dim, spot_dim)
        self.trans_dim = nn.Linear(spot_dim * 4, spot_dim)
        self.prediction_layers_expression = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(spot_dim, spot_dim // 2),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(spot_dim // 2, 1),
                )
                for _ in range(num_proteins)
            ]
        )

    def forward(self, image, protein_emb, steps=60):
        image_latent, pred_rna_emb, _, hist_celltype_prob = self.hist_flow_rna.encode_image_to_rna_emb(image, steps=steps)
        pred_prot_emb = integrate_flow(self.rna_flow, pred_rna_emb, steps=steps, cond_vec=hist_celltype_prob)
        spot_embeddings = torch.cat((image_latent, pred_rna_emb, pred_prot_emb), dim=-1)
        protein_projected = self.protein_projection(protein_emb)
        spot_embeddings_expanded = spot_embeddings.unsqueeze(1).repeat(1, self.num_proteins, 1)
        x = torch.cat((protein_projected, spot_embeddings_expanded), dim=-1)
        x = self.trans_dim(x)
        expr_preds = []
        for i in range(self.num_proteins):
            expr_preds.append(self.prediction_layers_expression[i](x[:, i, :]))
        return torch.cat(expr_preds, dim=1)


def parse_target_proteins(raw: str) -> List[str]:
    p = Path(raw)
    if p.exists():
        if p.suffix.lower() in [".csv", ".tsv"]:
            sep = "\t" if p.suffix.lower() == ".tsv" else ","
            df = pd.read_csv(p, sep=sep)
            if len(df.columns) == 1:
                return df.iloc[:, 0].astype(str).tolist()
            for c in ["protein", "protein_name", "name"]:
                if c in df.columns:
                    return df[c].astype(str).tolist()
            return df.iloc[:, 0].astype(str).tolist()
        with open(p, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return [x.strip() for x in raw.split(",") if x.strip()]


def build_scgpt_model(scgpt_model_dir: str, scgpt_model_state: Optional[str] = None):
    model_dir = Path(scgpt_model_dir)
    vocab_file = model_dir / "vocab.json"
    model_config_file = model_dir / "args.json"
    model_file = Path(scgpt_model_state) if scgpt_model_state else model_dir / "best_model.pt"

    vocab = GeneVocab.from_file(vocab_file)
    vocab.set_default_index(vocab["<pad>"])
    with open(model_config_file, "r", encoding="utf-8") as f:
        model_configs = json.load(f)

    scgpt_model = TransformerModel(
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
    load_pretrained(scgpt_model, torch.load(model_file, map_location="cpu"), verbose=False)
    return scgpt_model


def resolve_state_dict(ckpt_obj: Dict) -> Dict:
    if isinstance(ckpt_obj, dict):
        if "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
            return ckpt_obj["model_state_dict"]
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        return ckpt_obj
    raise ValueError("checkpoint format is invalid")


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        return {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def run_inference_for_one_image(
    model: nn.Module,
    image_path: str,
    coord_path: str,
    target_proteins: List[str],
    protein_emb_path: str,
    patch_size: int,
    batch_size: int,
    num_workers: int,
    steps: int,
    device: torch.device,
):
    dataset = InferenceProteinDataset(
        image_path=image_path,
        coord_path=coord_path,
        protein_names=target_proteins,
        protein_emb_path=protein_emb_path,
        patch_size=patch_size,
    )
    if len(dataset) == 0:
        return None

    effective_bs = min(batch_size, len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=effective_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    results_list = []
    with torch.no_grad():
        for images, prot_embs, xs, ys in loader:
            images = images.to(device)
            prot_embs = prot_embs.to(device)
            preds = model(images, prot_embs, steps=steps)
            preds_np = preds.cpu().numpy()
            xs_np = xs.numpy()
            ys_np = ys.numpy()

            for i in range(len(images)):
                row = {"x": xs_np[i], "y": ys_np[i]}
                for j, prot_name in enumerate(dataset.get_protein_names()):
                    row[prot_name] = preds_np[i, j]
                results_list.append(row)
    if not results_list:
        return None
    return pd.DataFrame(results_list)


def get_shape_fingerprint(cluster_data, bins=50):
    x = cluster_data[:, 0]
    y = cluster_data[:, 1]
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    heatmap, _, _ = np.histogram2d(x_centered, y_centered, bins=bins)
    return heatmap.flatten()


def prediction_to_adata(
    df_res: pd.DataFrame,
    dbscan_eps: float = 1500.0,
    dbscan_min_samples: int = 100,
    similarity_threshold: float = 0.5,
):
    coords = df_res.iloc[:, 0:2].values
    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(coords)
    df = df_res.copy()
    df["cluster"] = db.labels_
    unique_labels = [lbl for lbl in set(db.labels_) if lbl != -1]

    if len(unique_labels) < 2:
        final_df = df[df["cluster"] != -1].drop(columns=["cluster"])
    else:
        sorted_labels = df[df["cluster"] != -1]["cluster"].value_counts().index.tolist()
        label_a = sorted_labels[0]
        label_b = sorted_labels[1]

        coords_a = df[df["cluster"] == label_a].iloc[:, 0:2].values
        coords_b = df[df["cluster"] == label_b].iloc[:, 0:2].values

        vec_a = get_shape_fingerprint(coords_a)
        vec_b = get_shape_fingerprint(coords_b)

        similarity = np.corrcoef(vec_a, vec_b)[0, 1]

        if similarity > similarity_threshold:
            final_df = df[df["cluster"] == label_a].copy()
        else:
            final_df = df[df["cluster"] != -1].copy()

        final_df = final_df.drop(columns=["cluster"])

    expression_data = final_df.iloc[:, 2:]
    adata = ad.AnnData(expression_data)
    spatial_coords = final_df.iloc[:, 0:2].values
    adata.obsm["spatial"] = spatial_coords
    return adata


def main():
    parser = argparse.ArgumentParser(prog="spatial_prediction_toadata.py")
    parser.add_argument("--image_dir", default="/home/peijiazheng/agent/server/skills/spatial/data/images")
    parser.add_argument("--coord_dir", default="/home/peijiazheng/agent/server/skills/spatial/data/coords")
    parser.add_argument("--coord_prefix", default="valid_centers_")
    parser.add_argument("--coord_ext", default=".txt")
    parser.add_argument("--output_dir", default="/home/peijiazheng/agent/server/skills/spatial/results")

    parser.add_argument("--protein_emb_path", default="data/ESM2_embedding.pt")

    parser.add_argument("--scgpt_model_dir", default="/data/pjz/proteomics_back/data_model/scgpt_model")
    parser.add_argument("--scgpt_model_state", default=None)
    parser.add_argument("--conch_checkpoint_path", default="/data/pjz/proteomics_back/data_model/coach_model/pytorch_model.bin")
    parser.add_argument("--conch_model_cfg", default="conch_ViT-B-16")
    parser.add_argument("--reference_embedding_path", default="/home/peijiazheng/agent/server/skills/spatial/ckp/reference_embedding.csv")

    parser.add_argument("--hist_flow_rna_ckpt", default="/data/pjz/proteomics_back/data_model/tcga_coad/HistFlowRNA_final.pt")
    parser.add_argument("--final_model_ckpt", default="/data/pjz/proteomics_back/data_model/tcga_coad/image_to_protein_final_15.pt")

    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--image_glob", default="*.tif")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--dbscan_eps", type=float, default=1500.0, help="DBSCAN eps parameter for spatial region clustering")
    parser.add_argument("--dbscan_min_samples", type=int, default=100, help="DBSCAN min_samples parameter")
    parser.add_argument("--similarity_threshold", type=float, default=0.5, help="Shape similarity threshold for duplicate slice detection")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    target_proteins = ['PECAM1', 'PTPRC', 'CD68', 'CD4', 'FOXP3', 'CD8A', 'PTPRCRO',
                       'MS4A1', 'CD274', 'CD3E', 'CD163', 'CDH1', 'MKI67', 'KRT19', 'ACTA2']

    spatial_rna_model = build_scgpt_model(args.scgpt_model_dir, args.scgpt_model_state)

    path_model_full, _ = create_model_from_pretrained(
        args.conch_model_cfg,
        args.conch_checkpoint_path,
        force_image_size=256,
    )
    path_model_visual = path_model_full.visual

    class_embeddings = pd.read_csv(args.reference_embedding_path, index_col=0).values

    flow_model_rna = ConditionalFlowNet(source_dim=512, to_dim=512, cond_dim=0, hidden_dim=1024)
    hist_flow_rna = HistFlowRNA(
        rna_model=spatial_rna_model,
        hist_model=path_model_visual,
        class_embeddings=class_embeddings,
        flow_model=flow_model_rna,
    )
    hist_flow_rna = load_pretrained_model(hist_flow_rna, args.hist_flow_rna_ckpt, device="cpu")

    flow_model_prot = ConditionalFlowNet(512, 512, 40)
    model = ImageToProteinModel(hist_flow_rna, flow_model_prot, 512, 5120, len(target_proteins))

    state_obj = torch.load(args.final_model_ckpt, map_location="cpu")
    state_dict = strip_module_prefix(resolve_state_dict(state_obj))
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    tif_files = sorted(glob.glob(os.path.join(args.image_dir, args.image_glob)))
    for tif_path in tqdm(tif_files, desc="Predicting"):
        filename = os.path.basename(tif_path)
        basename = os.path.splitext(filename)[0]
        coord_filename = f"{args.coord_prefix}{basename}{args.coord_ext}"
        coord_path = os.path.join(args.coord_dir, coord_filename)
        if not os.path.exists(coord_path):
            continue

        df_res = run_inference_for_one_image(
            model=model,
            image_path=tif_path,
            coord_path=coord_path,
            target_proteins=target_proteins,
            protein_emb_path=args.protein_emb_path,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            steps=args.steps,
            device=device,
        )
        if df_res is None:
            continue

        adata = prediction_to_adata(
            df_res,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
            similarity_threshold=args.similarity_threshold,
        )

        save_path = os.path.join(args.output_dir, f"{basename}.h5ad")
        adata.write(save_path)
        print(f"Saved: {save_path} ({adata.n_obs} spots, {adata.n_vars} proteins)")


if __name__ == "__main__":
    main()
