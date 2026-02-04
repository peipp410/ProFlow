import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from torch.utils.data import DataLoader, random_split
from scgpt.model import TransformerModel
from scgpt.utils import load_pretrained
from scgpt.tokenizer import GeneVocab
from torch.optim import AdamW
from attn_lora import MHAttenLoRA
from pathlib import Path
from accelerate import Accelerator  # Import Accelerator
from accelerate.utils import set_seed
import os
import pickle

from dataset import BulkDataset, TestDataset
from losses import MMDLoss, RBF, cosine_similarity_loss
from encoders import PredictModel, PredictModelWithCLasses
from utils import calculate_correlations


seed = 42
torch.manual_seed(seed)     
np.random.seed(seed)      
torch.cuda.manual_seed(seed)        
torch.cuda.manual_seed_all(seed)    
torch.backends.cudnn.deterministic = True  
torch.backends.cudnn.benchmark = False 


def build_scgpt_model(mode='rna'):
    model_dir = "/home/peijiazheng/proteomics/data_model/scgpt_model"
    model_dir = Path(model_dir)
    vocab_file = model_dir / "vocab.json"
    model_config_file = model_dir / "args.json"
    # model_file = model_dir / "best_model.pt"
    if mode == 'rna':
        model_file = '/mnt/vdd/pjz/rna_state_epoch_5.pt'
        # model_file = os.path.join(model_dir, 'best_model.pt')
    elif mode == 'protein':
        model_file = '/mnt/vdd/pjz/protein_state_all.pt'
        # model_file = os.path.join(model_dir, 'best_model.pt')
    else:
        raise ValueError("mode should be 'rna' or 'protein'")

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
    def __init__(self, prot_dim, rna_dim, cond_dim, hidden_dim=1024):
        super().__init__()
        input_dim = prot_dim + cond_dim + 1  # +1 for t
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, prot_dim)
        )

    def forward(self, z_t, t, cond_vec):
        """
        z_t: (B, prot_dim)
        t: (B, 1)
        cond_vec: (B, rna_dim + cond_dim)
        """
        ht = torch.cat([z_t, cond_vec, t], dim=-1)
        return self.net(ht)


def one_hot(labels, num_classes):
    return F.one_hot(labels, num_classes=num_classes).float()


def train_flow_matching_with_pred(
    rna_model, protein_model, flow_model, model_pred, optimizer,
    loader, eval_loader, num_cancer_types, start_from="rna",
    epochs=100, steps=60, lambda_final=1.0, lambda_pred=1.0, device="cuda", protein_names=None
):
    rbf_kernel = RBF(bandwidth=1.0, learnable_bandwidth=True)
    mmd_loss_fn = MMDLoss(rbf_kernel)

    for epoch in range(1, epochs+1):
        flow_model.train()
        model_pred.train()

        total_loss = 0.0
        flow_loss_acc = 0.0
        final_loss_acc = 0.0
        pred_loss_acc = 0.0

        all_train_predictions = []
        all_train_targets = []

        for batch in loader:
            for k in batch:
                batch[k] = batch[k].to(device)

            # RNA embedding
            with torch.no_grad():
                gene_emb = rna_model.encoder(batch["input_gene_ids"])
                expr_emb = rna_model.value_encoder(batch["rna_expressions"])
                zA = rna_model.transformer_encoder(
                    gene_emb + expr_emb,
                    src_key_padding_mask=batch["src_key_padding_mask"]
                )[:, 0, :]

                gene_emb_p = protein_model.encoder(batch["input_protein_ids"])
                expr_emb_p = protein_model.value_encoder(batch["protein_expressions"])
                zB = protein_model.transformer_encoder(
                    gene_emb_p + expr_emb_p,
                    src_key_padding_mask=batch["src_key_padding_mask_protein"]
                )[:, 0, :]

            labels = batch["sample_group"] - 1
            cond_label = one_hot(labels, num_classes=num_cancer_types).to(device)
            # cond_vec = torch.cat([zB, cond_label], dim=-1)
            cond_vec= cond_label

            if start_from.lower() == "rna":
                start_state = zA.clone()
                u_true = zB - zA
            elif start_from.lower() == "noise":
                start_state = torch.randn_like(zB)
                u_true = zB - start_state
            else:
                raise ValueError("start_from should be 'rna' or 'noise'")

            dt = 1.0 / steps
            z = start_state.clone()
            loss_flow_total = 0.0
            for step in range(steps):
                t_val = step / steps
                t_tensor = torch.full((z.size(0), 1), t_val, device=device)
                k1 = flow_model(z, t_tensor, cond_vec)
                k2 = flow_model(z + 0.5 * dt * k1, t_tensor + 0.5*dt, cond_vec)
                k3 = flow_model(z + 0.5 * dt * k2, t_tensor + 0.5*dt, cond_vec)
                k4 = flow_model(z + dt * k3,       t_tensor + dt,      cond_vec)
                z = z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                loss_flow_total += ((k1 - u_true)**2).mean()

            loss_final = cosine_similarity_loss(z, zB) + mmd_loss_fn(z, zB)
            protein_emb = batch["protein_emb"]
            protein_expressions_ori = batch["protein_exp_ori"]
            # print(z.shape, cond_label.shape, protein_emb.shape)
            expression_predictions = model_pred(z, cond_label, protein_emb)
            loss_pred = F.mse_loss(expression_predictions, protein_expressions_ori) + \
                        mmd_loss_fn(expression_predictions.detach(), protein_expressions_ori.detach())

            loss = (loss_flow_total / steps) + lambda_final * loss_final + lambda_pred * loss_pred

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            flow_loss_acc += (loss_flow_total / steps).item()
            final_loss_acc += loss_final.item()
            pred_loss_acc += loss_pred.item()

            all_train_predictions.append(expression_predictions.detach().cpu())
            all_train_targets.append(protein_expressions_ori.detach().cpu())

        all_train_predictions = torch.cat(all_train_predictions, dim=0)
        all_train_targets = torch.cat(all_train_targets, dim=0)
        correlations = calculate_correlations(all_train_predictions, all_train_targets)
        pcc_vals, scc_vals = zip(*correlations)
        avg_pcc = np.mean(pcc_vals); max_pcc = np.max(pcc_vals); min_pcc = np.min(pcc_vals)
        avg_scc = np.mean(scc_vals); max_scc = np.max(scc_vals); min_scc = np.min(scc_vals)

        print(f"[Epoch {epoch}] Total={total_loss/len(loader):.4f} "
              f"| Flow={flow_loss_acc/len(loader):.4f} "
              f"| Final={final_loss_acc/len(loader):.4f} "
              f"| Pred={pred_loss_acc/len(loader):.4f} "
              f"| PCC(avg/max/min)=({avg_pcc:.4f}/{max_pcc:.4f}/{min_pcc:.4f}) "
              f"| SCC(avg/max/min)=({avg_scc:.4f}/{max_scc:.4f}/{min_scc:.4f})")

        if (eval_loader is not None) and (epoch % 5 == 0):
            flag = (epoch == epochs)
            evaluate_flow_matching_with_pred(
                rna_model, protein_model, flow_model, model_pred,
                eval_loader, num_cancer_types, start_from, device, "rk4", steps, protein_names, save=flag
            )

    all_train_predictions = pd.DataFrame(all_train_predictions.cpu().numpy(), columns=protein_names)
    all_train_targets = pd.DataFrame(all_train_targets.cpu().numpy(), columns=protein_names)
    # all_train_predictions.to_csv(f'result/train_prediction.csv')
    # all_train_targets.to_csv(f'result/train_target.csv')


def evaluate_flow_matching_with_pred(
    rna_model, protein_model, flow_model, model_pred,
    loader, num_cancer_types, start_from="rna",
    device="cuda", method="rk4", steps=80, protein_names=None, save=False
):
    rbf_kernel = RBF(bandwidth=1.0, learnable_bandwidth=True)
    mmd_loss_fn = MMDLoss(rbf_kernel)

    flow_model.eval()
    model_pred.eval()
    total_pred_loss = 0.0
    total_emb_mse = 0.0
    all_preds = []
    all_tgts = []

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
                src_key_padding_mask=batch["src_key_padding_mask_protein"]
            )[:, 0, :]

            cond_label = one_hot(batch["sample_group"] - 1, num_cancer_types).to(device)
            # cond_vec = torch.cat([zB, cond_label], dim=-1)
            cond_vec= cond_label

            if start_from == "rna":
                z = zA.clone()
            else:
                z = torch.randn_like(zB)

            dt = 1.0 / steps
            for step in range(steps):
                t_val = step / steps
                t_tensor = torch.full((z.size(0), 1), t_val, device=device)
                k1 = flow_model(z, t_tensor, cond_vec)
                k2 = flow_model(z + 0.5 * dt * k1, t_tensor + 0.5*dt, cond_vec)
                k3 = flow_model(z + 0.5 * dt * k2, t_tensor + 0.5*dt, cond_vec)
                k4 = flow_model(z + dt * k3,       t_tensor + dt,      cond_vec)
                z = z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            total_emb_mse += F.mse_loss(z, zB).item()
            protein_emb = batch["protein_emb"]
            protein_exp_ori = batch["protein_exp_ori"]
            preds = model_pred(z, cond_label, protein_emb)

            total_pred_loss += F.mse_loss(preds, protein_exp_ori).item()
            all_preds.append(preds.cpu())
            all_tgts.append(protein_exp_ori.cpu())

    all_preds = torch.cat(all_preds)
    all_tgts = torch.cat(all_tgts)
    corrs = calculate_correlations(all_preds, all_tgts)
    pcc_vals, scc_vals = zip(*corrs)
    avg_pcc = np.mean(pcc_vals); max_pcc = np.max(pcc_vals); min_pcc = np.min(pcc_vals)
    avg_scc = np.mean(scc_vals); max_scc = np.max(scc_vals); min_scc = np.min(scc_vals)

    print(f"[Eval] EmbMSE={total_emb_mse/len(loader):.4f} "
          f"| PredMSE={total_pred_loss/len(loader):.4f} "
          f"| PCC(avg/max/min)=({avg_pcc:.4f}/{max_pcc:.4f}/{min_pcc:.4f}) "
          f"| SCC(avg/max/min)=({avg_scc:.4f}/{max_scc:.4f}/{min_scc:.4f})")
    if save:
        all_test_predictions = pd.DataFrame(all_preds.cpu().numpy(), columns=protein_names)
        all_test_targets = pd.DataFrame(all_tgts.cpu().numpy(), columns=protein_names)
        all_test_predictions.to_csv(f'final_result/tcga_test_prediction.csv')
        all_test_targets.to_csv(f'final_result/tcga_test_target.csv')


def test_flow_matching_with_pred(
    rna_model, protein_model, flow_model, model_pred,
    loader, num_cancer_types, start_from="rna",
    device="cuda", method="rk4", steps=80, protein_names=None, save=False
):
    rbf_kernel = RBF(bandwidth=1.0, learnable_bandwidth=True)
    mmd_loss_fn = MMDLoss(rbf_kernel)

    flow_model.eval()
    model_pred.eval()
    total_pred_loss = 0.0
    total_emb_mse = 0.0
    all_preds = []
    all_tgts = []

    with torch.no_grad():
        for batch in loader:
            for k in batch:
                if k != 'barcode':
                    batch[k] = batch[k].to(device)

            gene_emb = rna_model.encoder(batch["input_gene_ids"])
            expr_emb = rna_model.value_encoder(batch["rna_expressions"])
            zA = rna_model.transformer_encoder(
                gene_emb + expr_emb, src_key_padding_mask=batch["src_key_padding_mask"]
            )[:, 0, :]

            cond_label = one_hot(batch["sample_group"] - 1, num_cancer_types).to(device)
            # cond_vec = torch.cat([zB, cond_label], dim=-1)
            cond_vec= cond_label

            if start_from == "rna":
                z = zA.clone()
            else:
                z = torch.randn_like(zB)

            dt = 1.0 / steps
            for step in range(steps):
                t_val = step / steps
                t_tensor = torch.full((z.size(0), 1), t_val, device=device)
                k1 = flow_model(z, t_tensor, cond_vec)
                k2 = flow_model(z + 0.5 * dt * k1, t_tensor + 0.5*dt, cond_vec)
                k3 = flow_model(z + 0.5 * dt * k2, t_tensor + 0.5*dt, cond_vec)
                k4 = flow_model(z + dt * k3,       t_tensor + dt,      cond_vec)
                z = z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            protein_emb = batch["protein_emb"]
            preds = model_pred(z, cond_label, protein_emb)

            all_preds.append(preds.cpu())

    all_preds = torch.cat(all_preds)

    if save:
        all_test_predictions = pd.DataFrame(all_preds.cpu().numpy(), columns=protein_names)
        all_test_predictions.to_csv(f'final_result/tcga_test_prediction_from_rna.csv')
                            


if __name__ == '__main__':

    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    dataset = BulkDataset(
        rna_path='/home/peijiazheng/proteomics/crc_codex/pretrain/bulk_data/new_data/rna_common.parquet',
        protein_path='/home/peijiazheng/proteomics/crc_codex/pretrain/bulk_data/new_data/protein_common.csv.gz',
        sample_group_path='/home/peijiazheng/proteomics/crc_codex/pretrain/bulk_data/new_data/sample_filtered.csv',
        scgpt_model_path='/home/peijiazheng/proteomics/data_model/scgpt_model',
        protein_emb_path='/home/peijiazheng/proteomics/train_brca/pickle/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt'
    )  # XA: (N, in_dimA), cancer_labels: (N,)


    # train_size = int(0.7 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    # train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, drop_last=True)
    # test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    sample_df = pd.read_csv('/home/peijiazheng/proteomics/crc_codex/pretrain/bulk_data/new_data/sample_filtered.csv')
    sample_names = sample_df['sample'].values

    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    test_indices = test_dataset.indices

    test_sample_names = sample_names[test_indices]
    save_path = '/home/peijiazheng/proteomics/cross_modal_ot/final_result/test_samples.csv'
    pd.DataFrame({'Sample': test_sample_names}).to_csv(save_path, index=False)
    print(f"Saving to : {save_path}")
    print(f"Test size: {len(test_sample_names)}")

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    protein_names = dataset.get_protein_names()
    # with open("data/protein_name.pkl", "wb") as f:
    #     pickle.dump(protein_names, f)
    #     print('save protein done')

    test_rna_dataset = TestDataset(
        rna_path='/home/peijiazheng/proteomics/crc_codex/pretrain/bulk_data/new_data/rna_only.parquet',
        sample_group_path='/home/peijiazheng/proteomics/crc_codex/pretrain/bulk_data/new_data/sample_rna_filtered.csv',
        scgpt_model_path='/home/peijiazheng/proteomics/data_model/scgpt_model',
        protein_emb_path='/home/peijiazheng/proteomics/train_brca/pickle/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt',
        protein_names=protein_names
    )
    test_rna_loader = DataLoader(test_rna_dataset, batch_size=512, shuffle=False)

    rna_model = build_scgpt_model(mode='rna').to(device)
    protein_model = build_scgpt_model(mode='protein').to(device)
    for p in rna_model.parameters(): p.requires_grad = False
    for p in protein_model.parameters(): p.requires_grad = False

    flow_model = ConditionalFlowNet(512, 512, dataset.sample_group.max().item()).to(device)
    pred_model = PredictModelWithCLasses(512, 5120, dataset.sample_group.max().item(), 383, protein_names=protein_names).to(device)
    optimizer_joint = AdamW(list(flow_model.parameters()) + list(pred_model.parameters()),
                            lr=2e-4, weight_decay=1e-4)

    # flow_model = nn.DataParallel(flow_model)
    # pred_model = nn.DataParallel(pred_model)

    print(dataset.sample_group.max().item())

    train_flow_matching_with_pred(
        rna_model, protein_model, flow_model, pred_model, optimizer_joint,
        train_loader, test_loader, num_cancer_types=dataset.sample_group.max().item(),
        start_from='rna', epochs=100, steps=60, device=device, protein_names=protein_names
    )

    test_flow_matching_with_pred(
        rna_model, protein_model, flow_model, pred_model, 
        test_rna_loader, num_cancer_types=dataset.sample_group.max().item(),
        start_from='rna', steps=60, device=device, protein_names=protein_names, save=True
        )


    # save_dir = "/mnt/vdd/pjz/flow/bulk"

    # torch.save(flow_model.state_dict(), os.path.join(save_dir, "flow_model.pt"))
    # torch.save(pred_model.state_dict(), os.path.join(save_dir, "pred_model.pt"))
