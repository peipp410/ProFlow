import argparse


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Cleaned bulk-proteomics K-fold training; original behavior is retained by default.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    inputs = parser.add_argument_group("Input and output paths")
    inputs.add_argument(
        '--rna-path',
        default='train_data/rna_common.parquet',
        help='RNA parquet, features by samples',
    )
    inputs.add_argument(
        '--protein-path',
        default='train_data/protein_common.csv.gz',
        help='Protein CSV, features by samples',
    )
    inputs.add_argument(
        '--sample-group-path',
        default='train_data/sample_filtered_correct.csv',
        help='Sample group CSV',
    )
    inputs.add_argument(
        '--sample-column',
        default='sample',
        help='Column used to export test sample names',
    )
    inputs.add_argument(
        '--scgpt-model-path',
        default='ckp/scgpt_model',
        help='scGPT vocabulary/config directory',
    )
    inputs.add_argument(
        '--vocab-name',
        default='vocab.json',
        help='Vocabulary filename or absolute path',
    )
    inputs.add_argument(
        '--model-config-name',
        default='args.json',
        help='scGPT model configuration filename or absolute path',
    )
    inputs.add_argument(
        '--rna-checkpoint',
        default='ckp/scgpt_model/rna_state_epoch_5.pt',
        help='Pretrained RNA weights',
    )
    inputs.add_argument(
        '--protein-checkpoint',
        default='ckp/scgpt_model/protein_state_all.pt',
        help='Pretrained protein weights',
    )
    inputs.add_argument(
        '--protein-emb-path',
        default='/data/pjz/proteomics_back/train_brca/pickle/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2_new.pt',
        help='ESM-2 embedding dictionary',
    )
    inputs.add_argument(
        '--result-dir',
        default='final_result_new',
        help='Output directory, relative to the current working directory',
    )

    training = parser.add_argument_group("Training and evaluation")
    training.add_argument(
        '--device',
        default='auto',
        help='auto selects cuda when CUDA is available, otherwise cpu; or supply cpu/cuda:N',
    )
    training.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Global seed; each fold uses seed + fold - 1',
    )
    training.add_argument("--num-folds", type=int, default=5, help="Number of KFold splits")
    training.add_argument("--batch-size", type=int, default=512, help="Training batch size")
    training.add_argument("--eval-batch-size", type=int, default=512, help="Test batch size")
    training.add_argument("--epochs", type=int, default=500, help="Training epochs per fold")
    training.add_argument(
        '--steps',
        type=int,
        default=60,
        help='RK4 steps in training and evaluation',
    )
    training.add_argument(
        '--eval-every',
        type=int,
        default=5,
        help='Evaluation interval; also evaluate the last epoch',
    )
    training.add_argument(
        '--learning-rate',
        '--lr',
        type=float,
        default=0.0002,
        help='AdamW learning rate',
    )
    training.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    training.add_argument(
        '--lambda-pred',
        type=float,
        default=1.0,
        help='Weight of the original MSE + MMD prediction loss',
    )
    training.add_argument(
        '--start-from',
        choices=('rna', 'noise'),
        default='rna',
        help='Initial flow state',
    )
    training.add_argument(
        '--drop-last',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Drop the last incomplete training batch',
    )
    training.add_argument(
        '--train-shuffle',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Shuffle training batches',
    )
    training.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes")

    model = parser.add_argument_group("Flow and prediction model")
    model.add_argument(
        '--latent-dim',
        type=int,
        default=512,
        help='Latent dimension; must match the pretrained scGPT models',
    )
    model.add_argument(
        '--flow-hidden-dim',
        type=int,
        default=1024,
        help='Flow MLP hidden dimension',
    )
    model.add_argument(
        '--pred-hidden-dim',
        type=int,
        default=512,
        help='Residual prediction head hidden dimension',
    )
    model.add_argument("--pred-num-layers", type=int, default=2, help="Residual block count")
    model.add_argument("--pred-dropout", type=float, default=0.05, help="Prediction head dropout")
    model.add_argument("--pred-expansion", type=int, default=2, help="Residual block expansion")
    model.add_argument(
        '--residual-scale',
        type=float,
        default=0.1,
        help='Residual branch multiplier',
    )

    data = parser.add_argument_group("Existing preprocessing settings")
    data.add_argument(
        '--protein-missing-rate',
        type=float,
        default=0.1,
        help='Keep proteins with missing rate strictly below this value, before splitting',
    )
    data.add_argument(
        '--zero-row-threshold',
        type=float,
        default=1e-09,
        help='Original all-zero RNA detection threshold',
    )
    data.add_argument(
        '--max-seq-length',
        type=int,
        default=1200,
        help='Maximum scGPT token sequence length',
    )
    data.add_argument("--n-bins", type=int, default=51, help="Per-sample expression binning count")
    data.add_argument(
        '--sample-tokens',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Randomly sample overlong token sequences; otherwise truncate',
    )

    loss = parser.add_argument_group("Existing MMD settings")
    loss.add_argument("--mmd-bandwidth", type=float, default=1.0, help="RBF bandwidth")
    loss.add_argument("--mmd-n-kernels", type=int, default=5, help="Number of RBF kernels")
    loss.add_argument(
        '--mmd-mul-factor',
        type=float,
        default=2.0,
        help='RBF bandwidth multiplier base',
    )
    loss.add_argument(
        '--mmd-learnable-bandwidth',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Keep original parameter declaration; the kernel remains outside the optimizer',
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()


import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from encoders import PredictModelWithCLasses
from losses import RBF, MMDLoss
from utils import calculate_correlations, summarize_correlations


def build_scgpt_model(model_dir, model_file, vocab_name="vocab.json", model_config_name="args.json"):
    from scgpt.model import TransformerModel
    from scgpt.tokenizer import GeneVocab
    from scgpt.utils import load_pretrained

    model_dir = Path(model_dir)
    vocab_file = model_dir / vocab_name
    model_config_file = model_dir / model_config_name

    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab["<pad>"])

    with open(model_config_file, "r") as f:
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
    load_pretrained(scgpt_model, torch.load(model_file), verbose=False)
    return scgpt_model


class ConditionalFlowNet(nn.Module):
    def __init__(self, prot_dim, cond_dim, hidden_dim=1024):
        super().__init__()
        input_dim = prot_dim + cond_dim + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, prot_dim),
        )

    def forward(self, z_t, t, cond_vec):
        """
        z_t: (B, prot_dim)
        t: (B, 1)
        cond_vec: (B, cond_dim)
        """
        ht = torch.cat([z_t, cond_vec, t], dim=-1)
        return self.net(ht)


def one_hot(labels, num_classes):
    return F.one_hot(labels, num_classes=num_classes).float()


def integrate_rk4(flow_model, z0, cond_vec, steps=60):
    """Integrate the learned velocity field from t=0 to t=1 for inference."""
    dt = 1.0 / steps
    z = z0.clone()

    for step in range(steps):
        t_val = step / steps
        t_tensor = torch.full((z.size(0), 1), t_val, device=z.device)
        k1 = flow_model(z, t_tensor, cond_vec)
        k2 = flow_model(z + 0.5 * dt * k1, t_tensor + 0.5 * dt, cond_vec)
        k3 = flow_model(z + 0.5 * dt * k2, t_tensor + 0.5 * dt, cond_vec)
        k4 = flow_model(z + dt * k3, t_tensor + dt, cond_vec)
        z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return z


def train_flow_matching_with_pred(
    rna_model,
    protein_model,
    flow_model,
    model_pred,
    optimizer,
    loader,
    eval_loader,
    num_cancer_types,
    start_from="rna",
    epochs=100,
    steps=60,
    lambda_pred=1.0,
    device="cuda",
    protein_names=None,
    fold=None,
    num_folds=5,
    result_dir="final_result_new",
    eval_every=5,
    mmd_bandwidth=1.0,
    mmd_n_kernels=5,
    mmd_mul_factor=2.0,
    mmd_learnable_bandwidth=True,
):
    rbf_kernel = RBF(
        bandwidth=mmd_bandwidth,
        learnable_bandwidth=mmd_learnable_bandwidth,
        n_kernels=mmd_n_kernels,
        mul_factor=mmd_mul_factor,
    )
    mmd_loss_fn = MMDLoss(rbf_kernel)
    eval_metrics = None

    for epoch in range(1, epochs + 1):
        flow_model.train()
        model_pred.train()

        total_loss = 0.0
        flow_loss_acc = 0.0
        pred_loss_acc = 0.0

        all_train_predictions = []
        all_train_targets = []
        all_train_missing_masks = []

        for batch in loader:
            for k in batch:
                batch[k] = batch[k].to(device)

            with torch.no_grad():
                gene_emb = rna_model.encoder(batch["input_gene_ids"])
                expr_emb = rna_model.value_encoder(batch["rna_expressions"])
                zA = rna_model.transformer_encoder(
                    gene_emb + expr_emb,
                    src_key_padding_mask=batch["src_key_padding_mask"],
                )[:, 0, :]

                gene_emb_p = protein_model.encoder(batch["input_protein_ids"])
                expr_emb_p = protein_model.value_encoder(batch["protein_expressions"])
                zB = protein_model.transformer_encoder(
                    gene_emb_p + expr_emb_p,
                    src_key_padding_mask=batch["src_key_padding_mask_protein"],
                )[:, 0, :]

            labels = batch["sample_group"] - 1
            cond_label = one_hot(labels, num_classes=num_cancer_types).to(device)

            cond_vec = cond_label

            if start_from.lower() == "rna":
                z0 = zA
            elif start_from.lower() == "noise":
                z0 = torch.randn_like(zB)
            else:
                raise ValueError("start_from should be 'rna' or 'noise'")

            z1 = zB
            t_tensor = torch.rand(z0.size(0), 1, device=device)
            z_t = (1.0 - t_tensor) * z0 + t_tensor * z1
            u_true = z1 - z0
            v_t = flow_model(z_t, t_tensor, cond_vec)
            loss_flow = F.mse_loss(v_t, u_true)

            with torch.no_grad():
                z = integrate_rk4(flow_model, z0, cond_vec, steps=steps)

            protein_emb = batch["protein_emb"]
            protein_expressions_ori = batch["protein_exp_ori"]

            expression_predictions = model_pred(z, cond_label, protein_emb)
            loss_pred = F.mse_loss(
                expression_predictions, protein_expressions_ori
            ) + mmd_loss_fn(expression_predictions, protein_expressions_ori)

            loss = loss_flow + lambda_pred * loss_pred

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            flow_loss_acc += loss_flow.item()
            pred_loss_acc += loss_pred.item()

            all_train_predictions.append(expression_predictions.detach().cpu())
            all_train_targets.append(protein_expressions_ori.detach().cpu())
            all_train_missing_masks.append(batch["protein_missing_mask"].detach().cpu())

        all_train_predictions = torch.cat(all_train_predictions, dim=0)
        all_train_targets = torch.cat(all_train_targets, dim=0)
        all_train_missing_masks = torch.cat(all_train_missing_masks, dim=0)
        correlation_records = calculate_correlations(
            all_train_predictions,
            all_train_targets,
            feature_names=protein_names,
            missing_mask=all_train_missing_masks,
        )
        correlation_summary = summarize_correlations(correlation_records)
        pcc_summary = correlation_summary["pearson_r"]
        scc_summary = correlation_summary["spearman_rho"]

        fold_prefix = f"[Fold {fold}/{num_folds}]" if fold is not None else ""
        print(
            f"{fold_prefix}[Train][Epoch {epoch}] Total={total_loss/len(loader):.4f} "
            f"| Flow={flow_loss_acc/len(loader):.4f} "
            f"| Pred={pred_loss_acc/len(loader):.4f} "
            f"| PCC(mean/max/min/median/IQR)="
            f"({pcc_summary['mean']:.4f}/{pcc_summary['max']:.4f}/"
            f"{pcc_summary['min']:.4f}/{pcc_summary['median']:.4f}/"
            f"{pcc_summary['iqr']:.4f}) "
            f"| SCC(mean/max/min/median/IQR)="
            f"({scc_summary['mean']:.4f}/{scc_summary['max']:.4f}/"
            f"{scc_summary['min']:.4f}/{scc_summary['median']:.4f}/"
            f"{scc_summary['iqr']:.4f})"
        )

        if epoch == epochs:
            os.makedirs(result_dir, exist_ok=True)
            pd.DataFrame(correlation_records).to_csv(
                os.path.join(result_dir, "train_protein_correlations.csv"), index=False
            )

        if (eval_loader is not None) and (epoch % eval_every == 0 or epoch == epochs):
            flag = epoch == epochs
            eval_metrics = evaluate_flow_matching_with_pred(
                rna_model,
                protein_model,
                flow_model,
                model_pred,
                eval_loader,
                num_cancer_types,
                start_from,
                device,
                steps,
                protein_names,
                save=flag,
                fold=fold,
                num_folds=num_folds,
                result_dir=result_dir,
            )

    return {
        "total_loss": total_loss / len(loader),
        "flow_loss": flow_loss_acc / len(loader),
        "pred_loss": pred_loss_acc / len(loader),
        "pearson_mean": pcc_summary["mean"],
        "spearman_mean": scc_summary["mean"],
    }, eval_metrics


def evaluate_flow_matching_with_pred(
    rna_model,
    protein_model,
    flow_model,
    model_pred,
    loader,
    num_cancer_types,
    start_from="rna",
    device="cuda",
    steps=80,
    protein_names=None,
    save=False,
    fold=None,
    num_folds=5,
    result_dir="final_result_new",
):
    flow_model.eval()
    model_pred.eval()
    total_pred_loss = 0.0
    total_emb_mse = 0.0
    all_preds = []
    all_tgts = []
    all_missing_masks = []

    with torch.no_grad():
        for batch in loader:
            for k in batch:
                batch[k] = batch[k].to(device)

            gene_emb = rna_model.encoder(batch["input_gene_ids"])
            expr_emb = rna_model.value_encoder(batch["rna_expressions"])
            zA = rna_model.transformer_encoder(
                gene_emb + expr_emb, src_key_padding_mask=batch["src_key_padding_mask"]
            )[:, 0, :]

            gene_emb_p = protein_model.encoder(batch["input_protein_ids"])
            expr_emb_p = protein_model.value_encoder(batch["protein_expressions"])
            zB = protein_model.transformer_encoder(
                gene_emb_p + expr_emb_p,
                src_key_padding_mask=batch["src_key_padding_mask_protein"],
            )[:, 0, :]

            cond_label = one_hot(batch["sample_group"] - 1, num_cancer_types).to(device)

            cond_vec = cond_label

            if start_from == "rna":
                z = zA.clone()
            else:
                z = torch.randn_like(zB)

            dt = 1.0 / steps
            for step in range(steps):
                t_val = step / steps
                t_tensor = torch.full((z.size(0), 1), t_val, device=device)
                k1 = flow_model(z, t_tensor, cond_vec)
                k2 = flow_model(z + 0.5 * dt * k1, t_tensor + 0.5 * dt, cond_vec)
                k3 = flow_model(z + 0.5 * dt * k2, t_tensor + 0.5 * dt, cond_vec)
                k4 = flow_model(z + dt * k3, t_tensor + dt, cond_vec)
                z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            total_emb_mse += F.mse_loss(z, zB).item()
            protein_emb = batch["protein_emb"]
            protein_exp_ori = batch["protein_exp_ori"]
            preds = model_pred(z, cond_label, protein_emb)

            total_pred_loss += F.mse_loss(preds, protein_exp_ori).item()
            all_preds.append(preds.cpu())
            all_tgts.append(protein_exp_ori.cpu())
            all_missing_masks.append(batch["protein_missing_mask"].cpu())

    all_preds = torch.cat(all_preds)
    all_tgts = torch.cat(all_tgts)
    all_missing_masks = torch.cat(all_missing_masks)
    correlation_records = calculate_correlations(
        all_preds,
        all_tgts,
        feature_names=protein_names,
        missing_mask=all_missing_masks,
    )
    correlation_summary = summarize_correlations(correlation_records)
    pcc_summary = correlation_summary["pearson_r"]
    scc_summary = correlation_summary["spearman_rho"]

    fold_prefix = f"[Fold {fold}/{num_folds}]" if fold is not None else ""
    print(
        f"{fold_prefix}[Test] EmbMSE={total_emb_mse/len(loader):.4f} "
        f"| PredMSE={total_pred_loss/len(loader):.4f} "
        f"| PCC(mean/max/min/median/IQR)="
        f"({pcc_summary['mean']:.4f}/{pcc_summary['max']:.4f}/"
        f"{pcc_summary['min']:.4f}/{pcc_summary['median']:.4f}/"
        f"{pcc_summary['iqr']:.4f}) "
        f"| SCC(mean/max/min/median/IQR)="
        f"({scc_summary['mean']:.4f}/{scc_summary['max']:.4f}/"
        f"{scc_summary['min']:.4f}/{scc_summary['median']:.4f}/"
        f"{scc_summary['iqr']:.4f})"
    )
    if save:
        os.makedirs(result_dir, exist_ok=True)
        all_test_predictions = pd.DataFrame(
            all_preds.cpu().numpy(), columns=protein_names
        )
        all_test_targets = pd.DataFrame(all_tgts.cpu().numpy(), columns=protein_names)
        all_test_predictions.to_csv(os.path.join(result_dir, "tcga_test_prediction.csv"))
        all_test_targets.to_csv(os.path.join(result_dir, "tcga_test_target.csv"))
        pd.DataFrame(correlation_records).to_csv(
            os.path.join(result_dir, "tcga_test_protein_correlations.csv"),
            index=False,
        )

    return {
        "emb_mse": total_emb_mse / len(loader),
        "pred_mse": total_pred_loss / len(loader),
        "pearson_mean": pcc_summary["mean"],
        "spearman_mean": scc_summary["mean"],
    }


def main(args):
    from accelerate.utils import set_seed
    from dataset import BulkDataset

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    set_seed(seed)

    dataset = BulkDataset(
        rna_path=args.rna_path,
        protein_path=args.protein_path,
        sample_group_path=args.sample_group_path,
        scgpt_model_path=args.scgpt_model_path,
        protein_emb_path=args.protein_emb_path,
        protein_missing_rate=args.protein_missing_rate,
        zero_row_threshold=args.zero_row_threshold,
        max_seq_length=args.max_seq_length,
        n_bins=args.n_bins,
        sample_tokens=args.sample_tokens,
        vocab_name=args.vocab_name,
        model_config_name=args.model_config_name,
    )
    sample_df = pd.read_csv(args.sample_group_path)
    sample_names = sample_df[args.sample_column].values
    protein_names = dataset.get_protein_names()

    rna_model = build_scgpt_model(
        args.scgpt_model_path, args.rna_checkpoint, args.vocab_name, args.model_config_name
    ).to(device)
    protein_model = build_scgpt_model(
        args.scgpt_model_path, args.protein_checkpoint, args.vocab_name, args.model_config_name
    ).to(device)
    for p in rna_model.parameters():
        p.requires_grad = False
    for p in protein_model.parameters():
        p.requires_grad = False

    print(dataset.sample_group.max().item())
    num_folds = args.num_folds
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    fold_results = []

    for fold, (train_indices, test_indices) in enumerate(
        kfold.split(np.arange(len(dataset))), start=1
    ):
        set_seed(seed + fold - 1)
        fold_result_dir = os.path.join(args.result_dir, f"fold_{fold}")
        os.makedirs(fold_result_dir, exist_ok=True)
        train_dataset = Subset(dataset, train_indices.tolist())
        test_dataset = Subset(dataset, test_indices.tolist())
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=args.train_shuffle,
            drop_last=args.drop_last, num_workers=args.num_workers,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.eval_batch_size, shuffle=False,
            num_workers=args.num_workers,
        )
        test_sample_names = sample_names[test_indices]
        sample_path = os.path.join(fold_result_dir, "test_samples.csv")
        pd.DataFrame({"Sample": test_sample_names}).to_csv(sample_path, index=False)
        print(
            f"\n[Fold {fold}/{num_folds}] "
            f"Train size: {len(train_dataset)} | Test size: {len(test_dataset)}"
        )

        flow_model = ConditionalFlowNet(
            args.latent_dim, dataset.sample_group.max().item(),
            hidden_dim=args.flow_hidden_dim,
        ).to(device)
        pred_model = PredictModelWithCLasses(
            args.latent_dim,
            dataset.protein_embedding.shape[1],
            dataset.sample_group.max().item(),
            hidden_dim=args.pred_hidden_dim,
            num_layers=args.pred_num_layers,
            dropout=args.pred_dropout,
            expansion=args.pred_expansion,
            residual_scale=args.residual_scale,
        ).to(device)
        optimizer_joint = AdamW(
            list(flow_model.parameters()) + list(pred_model.parameters()),
            lr=args.learning_rate, weight_decay=args.weight_decay,
        )
        train_metrics, test_metrics = train_flow_matching_with_pred(
            rna_model, protein_model, flow_model, pred_model, optimizer_joint,
            train_loader, test_loader,
            num_cancer_types=dataset.sample_group.max().item(),
            start_from=args.start_from, epochs=args.epochs, steps=args.steps,
            lambda_pred=args.lambda_pred, device=device, protein_names=protein_names,
            fold=fold, num_folds=num_folds, result_dir=fold_result_dir,
            eval_every=args.eval_every, mmd_bandwidth=args.mmd_bandwidth,
            mmd_n_kernels=args.mmd_n_kernels, mmd_mul_factor=args.mmd_mul_factor,
            mmd_learnable_bandwidth=args.mmd_learnable_bandwidth,
        )
        fold_results.append(
            {
                "fold": fold,
                "train_size": len(train_dataset),
                "test_size": len(test_dataset),
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
        )
        print(
            f"[Fold {fold}/{num_folds}][Final] "
            f"Train PredLoss={train_metrics['pred_loss']:.4f} "
            f"PCC={train_metrics['pearson_mean']:.4f} "
            f"SCC={train_metrics['spearman_mean']:.4f} | "
            f"Test PredMSE={test_metrics['pred_mse']:.4f} "
            f"PCC={test_metrics['pearson_mean']:.4f} "
            f"SCC={test_metrics['spearman_mean']:.4f}"
        )
        pd.DataFrame(fold_results).to_csv(
            os.path.join(args.result_dir, "five_fold_metrics.csv"), index=False
        )


if __name__ == "__main__":
    main(args)
