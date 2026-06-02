import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from lora_pytorch import LoRA
from attn_lora import MHAttenLoRA
from itertools import cycle

from scgpt.model import TransformerModel
from scgpt.utils import load_pretrained
from scgpt.tokenizer import GeneVocab
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

import numpy as np
import pandas as pd
from pathlib import Path
import json
from get_Data import build_loaders_spatial

from accelerate import Accelerator  # Import Accelerator
from accelerate import FullyShardedDataParallelPlugin

seed = 42
torch.manual_seed(seed)     
np.random.seed(seed)      
torch.cuda.manual_seed(seed)        
torch.cuda.manual_seed_all(seed)    
torch.backends.cudnn.deterministic = True  
torch.backends.cudnn.benchmark = False 

accelerator = Accelerator(
    gradient_accumulation_steps=1,  # Let Accelerate figure out the correct value
    log_with="tensorboard",
    project_dir="./logs"   #  directory for saving logs
)


class ConditionalFlowNet(nn.Module):
    def __init__(self, source_dim, to_dim, cond_dim, hidden_dim=1024):
        super().__init__()
        self.cond_dim = cond_dim
        input_dim = source_dim + cond_dim + 1  # +1 for t
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, to_dim)
        )

    def forward(self, z_t, t, cond_vec=None):
        """
        z_t: (B, prot_dim)
        t: (B, 1)
        cond_vec: (B, rna_dim + cond_dim)
        """
        if self.cond_dim > 0:
            ht = torch.cat([z_t, cond_vec, t], dim=-1)
        else:
            ht = torch.cat([z_t, t], dim=-1)
        return self.net(ht)
    

class HistFlowRNA(nn.Module):
    def __init__(self, rna_model, hist_model, class_embeddings, flow_model,
                 feature_dim=512, lambda_focal=1.0,
                 lambda_cos=1.0, lambda_mmd=1.0, mmd_kernel=None):
        super().__init__()
        self.rna_model = rna_model
        self.hist_model = hist_model
        self.flow_model = flow_model
        self.lambda_focal = lambda_focal
        self.lambda_cos = lambda_cos
        self.lambda_mmd = lambda_mmd

        # MMD loss kernel
        self.mmd_loss_fn = mmd_kernel

        class_embeddings_tensor = torch.from_numpy(class_embeddings).float()
        self.n_ct = class_embeddings_tensor.shape[0]
        self.register_buffer('class_embeddings', class_embeddings_tensor)

        self.image_prob = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, self.n_ct, bias=False),
            nn.LogSoftmax(dim=1)
        )
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.image_prob.apply(init_weights)

    # @staticmethod
    # def masked_focal_loss(hist_celltype_prob, celltype_mask, gamma=2, alpha=0.25):
    #     hist_celltype_prob = torch.clamp(hist_celltype_prob, min=1e-7, max=1.0 - 1e-7)
    #     ce_loss = -torch.log(1 - hist_celltype_prob) * (1 - celltype_mask)
    #     p_t = (1 - hist_celltype_prob) * (1 - celltype_mask)
    #     focal_weight = alpha * (1 - p_t) ** gamma
    #     loss = (focal_weight * ce_loss).sum()
    #     return loss

    @staticmethod
    def kldiv_loss(hist_celltype_prob, celltype_mask):
        criterion = nn.KLDivLoss(reduction='batchmean')
        loss = criterion(hist_celltype_prob, celltype_mask)
        return loss

    def forward(self, image, input_gene_ids, expressions, src_key_padding_mask,
                celltype_mask=None, steps=60, cond_vec=None):
        """
        Flow Matching inference:
          1. image -> image_latent -> hist_celltype_prob -> hist_features
          2. input_gene_ids + expressions -> cell_embeddings
          3. hist_features matches pred_cell_emb
        Returns:
          pred_cell_emb, hist_features, cell_embeddings, hist_celltype_prob
        """
        cell_embeddings = self.rna_model._encode(
            input_gene_ids,
            expressions,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None
        )
        cell_embeddings = cell_embeddings[:, 0, :]

        # Hist -> image_latent
        image_latent, _ = self.hist_model(image)

        # Patch-level -> cell type probability
        hist_celltype_prob_log = self.image_prob(image_latent)  # [B, n_ct]
        hist_celltype_prob = torch.exp(hist_celltype_prob_log)
        hist_features = torch.matmul(hist_celltype_prob, self.class_embeddings)  # [B, feat_dim]

        # Flow Matching: hist_features -> pred_cell_emb
        dt = 1.0 / steps
        z = hist_features.clone()
        for step in range(steps):
            t_val = step / steps
            t_tensor = torch.full((z.size(0), 1), t_val, device=z.device)
            k1 = self.flow_model(z, t_tensor, cond_vec)
            k2 = self.flow_model(z + 0.5 * dt * k1, t_tensor + 0.5*dt, cond_vec)
            k3 = self.flow_model(z + 0.5 * dt * k2, t_tensor + 0.5*dt, cond_vec)
            k4 = self.flow_model(z + dt * k3, t_tensor + dt, cond_vec)
            z = z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        pred_cell_emb = z

        return pred_cell_emb, hist_features, cell_embeddings, hist_celltype_prob_log

    def compute_loss(self, pred_cell_emb, hist_features, cell_embeddings, hist_celltype_prob, celltype_mask):
        # 1. Flow MSE
        loss_flow_mse = F.mse_loss(pred_cell_emb, cell_embeddings)

        # 2. Cosine Loss
        cosine_sim = F.cosine_similarity(pred_cell_emb, cell_embeddings, dim=-1)
        loss_cosine = (1 - cosine_sim).mean()

        # 3. MMD Loss
        if self.mmd_loss_fn is not None:
            loss_mmd = self.mmd_loss_fn(pred_cell_emb, cell_embeddings)
        else:
            loss_mmd = torch.tensor(0.0, device=pred_cell_emb.device)

        # 4. Masked focal loss
        loss_focal = self.kldiv_loss(hist_celltype_prob, celltype_mask) * self.lambda_focal

        # Combine
        total_loss = loss_flow_mse + self.lambda_cos * loss_cosine + self.lambda_mmd * loss_mmd + loss_focal ###

        return total_loss, {
            "loss_mse": loss_flow_mse.item(),
            "loss_cosine": loss_cosine.item(),
            "loss_mmd": loss_mmd.item() if not isinstance(loss_mmd, float) else 0.0,
            "loss_focal": loss_focal.item()
        }
    

def train_and_validate(model, train_loader, test_loader,
                       optimizer,
                       num_epochs=5,
                       eval_every=5,
                       device="cuda"):

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss_vals = {
            "total": 0.0,
            "mse": 0.0,
            "cosine": 0.0,
            "mmd": 0.0,
            "focal": 0.0
        }
        num_batches = len(train_loader)

        # ======== Train Loop ========
        for batch in train_loader:
            image = batch["image"].to(device)
            input_gene_ids = batch["input_gene_ids"].to(device)
            expressions = batch["expressions"].to(device)
            src_key_padding_mask = batch["src_key_padding_mask"].to(device)
            celltype_mask = batch["spot_celltype_mask"].to(device)

            # Forward 
            pred_cell_emb, hist_features, cell_embeddings, hist_celltype_prob = accelerator.unwrap_model(model)(
                image, input_gene_ids, expressions, src_key_padding_mask, celltype_mask, steps=60
            )

            # Compute loss
            total_loss, loss_dict = accelerator.unwrap_model(model).compute_loss(
                pred_cell_emb, hist_features, cell_embeddings, hist_celltype_prob, celltype_mask
            )

            optimizer.zero_grad()
            accelerator.backward(total_loss)
            optimizer.step()

            # Accumulate
            epoch_loss_vals["total"] += total_loss.item()
            epoch_loss_vals["mse"] += loss_dict["loss_mse"]
            epoch_loss_vals["cosine"] += loss_dict["loss_cosine"]
            epoch_loss_vals["mmd"] += loss_dict["loss_mmd"]
            epoch_loss_vals["focal"] += loss_dict["loss_focal"]

        # Epoch 
        avg_losses = {k: v / num_batches for k, v in epoch_loss_vals.items()}
        train_losses.append(avg_losses)

        accelerator.print(
            f"[Train] Epoch {epoch+1}/{num_epochs} | total: {avg_losses['total']:.4f} "
            f"mse: {avg_losses['mse']:.4f} cosine: {avg_losses['cosine']:.4f} "
            f"mmd: {avg_losses['mmd']:.4f} focal: {avg_losses['focal']:.4f}"
        )

        # ======== Validation ========
        if (epoch + 1) % eval_every == 0:
            model.eval()
            val_epoch_loss_vals = {
                "total": 0.0,
                "mse": 0.0,
                "cosine": 0.0,
                "mmd": 0.0,
                "focal": 0.0
            }
            num_val_batches = len(test_loader)
            all_cell_embs = []
            all_celltypes = []
            all_indices = []

            with torch.no_grad():
                for batch in test_loader:
                    image = batch["image"].to(device)
                    input_gene_ids = batch["input_gene_ids"].to(device)
                    expressions = batch["expressions"].to(device)
                    src_key_padding_mask = batch["src_key_padding_mask"].to(device)
                    celltype_mask = batch["spot_celltype_mask"].to(device)

                    pred_cell_emb, hist_features, cell_embeddings, hist_celltype_prob = accelerator.unwrap_model(model)(
                        image, input_gene_ids, expressions, src_key_padding_mask, celltype_mask, steps=60
                    )

                    total_loss, loss_dict = accelerator.unwrap_model(model).compute_loss(
                        pred_cell_emb, hist_features, cell_embeddings, hist_celltype_prob, celltype_mask
                    )

                    val_epoch_loss_vals["total"] += total_loss.item()
                    val_epoch_loss_vals["mse"] += loss_dict["loss_mse"]
                    val_epoch_loss_vals["cosine"] += loss_dict["loss_cosine"]
                    val_epoch_loss_vals["mmd"] += loss_dict["loss_mmd"]
                    val_epoch_loss_vals["focal"] += loss_dict["loss_focal"]

                    all_cell_embs.append(pred_cell_emb.detach())
                    all_celltypes.append(torch.exp(hist_celltype_prob.detach()))
                    all_indices.append(batch["actual_idx"].detach())

            avg_val_losses = {k: v / num_val_batches for k, v in val_epoch_loss_vals.items()}
            val_losses.append(avg_val_losses)

            all_cell_embs = accelerator.gather_for_metrics(all_cell_embs)
            all_celltypes = accelerator.gather_for_metrics(all_celltypes)
            all_indices = accelerator.gather_for_metrics(all_indices)
            all_cell_embs = torch.cat(all_cell_embs, dim=0)
            all_celltypes = torch.cat(all_celltypes, dim=0)
            all_indices = torch.cat(all_indices, dim=0)
            accelerator.print(all_celltypes.shape)

            accelerator.print(
                f"[Val] Epoch {epoch+1} | total: {avg_val_losses['total']:.4f} "
                f"mse: {avg_val_losses['mse']:.4f} cosine: {avg_val_losses['cosine']:.4f} "
                f"mmd: {avg_val_losses['mmd']:.4f} focal: {avg_val_losses['focal']:.4f}"
            )

            if (epoch + 1) == num_epochs and accelerator.is_main_process:
                save_dir = 'result'
                all_cell_embs = pd.DataFrame(all_cell_embs.cpu().numpy())
                all_celltypes = pd.DataFrame(all_celltypes.cpu().numpy()
                )
                all_cell_embs.to_csv(f'result/all_cell_embs.csv')
                all_celltypes.to_csv(f'result/all_celltypes.csv')
                torch.save(all_indices.cpu().numpy(), 'result/all_indices_scale.pkl')

            # # save
            # accelerator.save(
            #     accelerator.unwrap_model(model).state_dict(),
            #     f'/macroverse-nas/pjz/crc_codex/pretrain/temp/HistFlowRNA_epoch_{epoch+1}.pth'
            # )

    return train_losses, val_losses


class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None, learnable_bandwidth=False):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        if bandwidth is None:
            self.bandwidth = None
        else:
            self.bandwidth = nn.Parameter(torch.tensor(bandwidth, dtype=torch.float)) if learnable_bandwidth else torch.tensor(bandwidth, dtype=torch.float) # Make bandwidth learnable
        self.learnable_bandwidth = learnable_bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            # Median heuristic
            n_samples = L2_distances.shape[0]
            # Calculate median instead of mean
            bandwidth = torch.median(L2_distances)
            return bandwidth
        return self.bandwidth

    def forward(self, X):
        self.bandwidth_multipliers = self.bandwidth_multipliers.to(X.device)
        L2_distances = torch.cdist(X, X) ** 2
        bandwidth = self.get_bandwidth(L2_distances)
        scaled_distances = -L2_distances[None, ...] / (bandwidth * self.bandwidth_multipliers)[:, None, None]
        scaled_distances = torch.clamp(scaled_distances, min=-32, max=0)
        return torch.exp(scaled_distances).sum(dim=0)


class LinearKernel(nn.Module):
    """
    Linear Kernel: K(x, y) = x^T y
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, X):
        # Compute the Gram matrix using the linear kernel
        return torch.mm(X, X.T)


class MMDLoss(nn.Module):
    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        n_x = X.size(0)
        n_y = Y.size(0)
        Z = torch.cat([X, Y], dim=0)
        K = self.kernel(Z)

        K_XX = K[:n_x, :n_x]
        K_YY = K[n_x:, n_x:]
        K_XY = K[:n_x, n_x:]

        # Unbiased MMD estimator
        sum_XX = K_XX.sum() - torch.diag(K_XX).sum()
        sum_YY = K_YY.sum() - torch.diag(K_YY).sum()
        sum_XY = K_XY.sum()

        mmd = sum_XX / (n_x * (n_x - 1) + 1e-8) + sum_YY / (n_y * (n_y - 1) + 1e-8) - 2 * sum_XY / (n_x * n_y + 1e-8) # Adding small constant for numerical stability

        return mmd


def build_scgpt_model():
    model_dir = "/macroverse-nas/pjz/proteomics/data_model/scgpt_model"
    model_dir = Path(model_dir)
    vocab_file = model_dir / "vocab.json"
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
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


if __name__ == '__main__':

    scgpt_model_path = "/macroverse-nas/pjz/proteomics/data_model/scgpt_model"
    coach_checkpoint_path = "/macroverse-nas/pjz/proteomics/data_model/coach_model/pytorch_model.bin"
    reference_embedding_path = "/macroverse-nas/pjz/crc_codex/pretrain/data/reference_embedding.csv"

    batch_size = 1024
    num_epochs = 50
    eval_every = 5
    lr = 5e-4
    lambda_focal = 10.0
    lambda_cos = 1.0
    lambda_mmd = 1.0
    steps = 60   # flow matching steps

    spatial_rna_model = build_scgpt_model()

    model_cfg = 'conch_ViT-B-16'
    path_model_full, preprocess = create_model_from_pretrained(
        model_cfg,
        coach_checkpoint_path,
        force_image_size=256
    )
    path_model_visual = path_model_full.visual

    for p in spatial_rna_model.parameters():
        p.requires_grad = False

    for p in path_model_visual.parameters():
        p.requires_grad = False

    class_embeddings = pd.read_csv(reference_embedding_path, index_col=0).values

    rbf_kernel = RBF(bandwidth=1.0, learnable_bandwidth=True)
    mmd_loss_fn = MMDLoss(rbf_kernel)

    flow_model = ConditionalFlowNet(
        source_dim=512,
        to_dim=512,
        cond_dim=0,     
        hidden_dim=1024
    )

    hist_flow_rna = HistFlowRNA(
        rna_model=spatial_rna_model,
        hist_model=path_model_visual,
        class_embeddings=class_embeddings,
        flow_model=flow_model,
        feature_dim=512,
        lambda_focal=lambda_focal,
        lambda_cos=lambda_cos,
        lambda_mmd=lambda_mmd,
        mmd_kernel=mmd_loss_fn
    )

    optimizer = torch.optim.Adam(
        hist_flow_rna.parameters(), lr=lr, betas=(0.9, 0.999)
    )

    # ----------- DataLoader -----------
    train_loader, test_loader = build_loaders_spatial(batch_size=batch_size)
    print('[INFO] Load data done.')

    hist_flow_rna, optimizer, train_loader, test_loader = accelerator.prepare(
        hist_flow_rna, optimizer, train_loader, test_loader
    )
    print('[INFO] Prepare done.')

    train_losses, val_losses = train_and_validate(
        model=hist_flow_rna,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        num_epochs=num_epochs,
        eval_every=eval_every,
        device=accelerator.device
    )

    accelerator.wait_for_everyone()

    # unwrapped_model = accelerator.unwrap_model(hist_flow_rna)
    # accelerator.save(
    #     unwrapped_model.state_dict(),
    #     "/macroverse-nas/pjz/crc_codex/pretrain/pickle/HistFlowRNA_final_kl_new.pt"
    # )
    # print('[INFO] Training complete. Model saved.')