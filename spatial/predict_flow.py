import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torch.optim as optim
from lora_pytorch import LoRA
from attn_lora import MHAttenLoRA
from itertools import cycle
from pretrained_model import HistFlowRNA, ConditionalFlowNet, load_pretrained_model
from dataset import ProteinExpressionDataset
import os
from tqdm import tqdm

from scgpt.model import TransformerModel
from scgpt.utils import load_pretrained
from scgpt.tokenizer import GeneVocab
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

import numpy as np
import pandas as pd
from pathlib import Path
import json
from scipy.stats import pearsonr, spearmanr

from accelerate import Accelerator  # Import Accelerator
from accelerate import FullyShardedDataParallelPlugin
from accelerate.utils import DistributedDataParallelKwargs

torch.autograd.set_detect_anomaly(True)
seed = 42
torch.manual_seed(seed)     
np.random.seed(seed)      
torch.cuda.manual_seed(seed)        
torch.cuda.manual_seed_all(seed)    
torch.backends.cudnn.deterministic = True  
torch.backends.cudnn.benchmark = False 

kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(
    gradient_accumulation_steps=1,  # Let Accelerate figure out the correct value
    log_with="tensorboard",
    project_dir="./logs",
    kwargs_handlers=[kwargs]
)


def my_collate(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch)


def integrate_flow(flow_model, z_init, steps=60, cond_vec=None):
    """ RK4 integration """
    dt = 1.0 / steps
    z = z_init.clone()
    for step in range(steps):
        t_val = step / steps
        t_tensor = torch.full((z.size(0), 1), t_val, device=z.device)
        k1 = flow_model(z, t_tensor, cond_vec)
        k2 = flow_model(z + 0.5 * dt * k1, t_tensor + 0.5*dt, cond_vec)
        k3 = flow_model(z + 0.5 * dt * k2, t_tensor + 0.5*dt, cond_vec)
        k4 = flow_model(z + dt * k3,       t_tensor + dt,      cond_vec)
        z = z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return z


class ImageToProteinModel(nn.Module):
    def __init__(self, hist_flow_rna, rna_flow, spot_dim, protein_dim, num_proteins, protein_names=None):
        super(ImageToProteinModel, self).__init__()
        self.hist_flow_rna = hist_flow_rna
        self.rna_flow = rna_flow
        self.num_proteins = num_proteins
        self.image_ln = nn.LayerNorm(spot_dim)

        self.protein_projection = nn.Linear(protein_dim, spot_dim)  # protein_emb -> spot_dim
        self.trans_dim = nn.Linear(spot_dim * 4, spot_dim)  # hist + rna + prot + protein_proj

        self.prediction_layers_expression = nn.ModuleList([
            nn.Sequential(
                nn.Linear(spot_dim, spot_dim // 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(spot_dim // 2, 1)
            ) for _ in range(num_proteins)
        ])

        if protein_names is not None:
            for i, name in enumerate(protein_names):
                self.prediction_layers_expression[i].name = name

    def forward(self, image, protein_emb, hist_features, hist_celltype_prob,
                pred_rna_emb, pred_prot_emb, steps=60):

        # === 1. Dynamic: image_latent via trainable hist_model (with LoRA) ===
        image_latent, _ = self.hist_flow_rna.hist_model(image)

        # === 2. Static: normalize cached features (no trainable affine params) ===
        image_latent = self.image_ln(image_latent)
        pred_rna_emb = F.layer_norm(pred_rna_emb, (pred_rna_emb.size(-1),))
        pred_prot_emb_norm = F.layer_norm(pred_prot_emb, (pred_prot_emb.size(-1),))

        # === 3. Integration ===
        spot_embeddings = torch.cat(
            (image_latent, pred_rna_emb, pred_prot_emb_norm), dim=-1
        )  # [B, spot_dim*3]
        protein_projected = self.protein_projection(protein_emb)  # [B, num_proteins, spot_dim]
        spot_embeddings = spot_embeddings.unsqueeze(1).repeat(1, self.num_proteins, 1)
        x = torch.cat((protein_projected, spot_embeddings), dim=-1)  # [B, num_proteins, spot_dim*4]
        x = self.trans_dim(x)  # [B, num_proteins, spot_dim]

        # === 4. Prediction ===
        expr_preds = []
        for i in range(self.num_proteins):
            pred = self.prediction_layers_expression[i](x[:, i, :])  # (B, 1)
            expr_preds.append(pred)
        expr_preds = torch.cat(expr_preds, dim=1)  # (B, num_proteins)

        return pred_prot_emb, expr_preds


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


def cosine_similarity_loss(x, y):
    
    cosine_sim = F.cosine_similarity(x, y, dim=-1)
    return 1 - cosine_sim.mean()


class ZILNLoss(nn.Module):
    def __init__(self, classification_weight=1.0, regression_weight=1.0, mmd_weight=1.0):
        super(ZILNLoss, self).__init__()
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        self.mmd_weight = mmd_weight
        self.rbf_kernel = RBF(bandwidth=1.0, learnable_bandwidth=True)
        self.mmd = MMDLoss(self.rbf_kernel)

    def forward(self, predicted_expression_values, target_expression):
        """
        Args:
            predicted_presence_logits (torch.Tensor): Model predicted protein expression logits, shape (batch_size, num_proteins)
            predicted_expression_values (torch.Tensor): Model predicted protein expression values, shape (batch_size, num_proteins)
            target_expression (torch.Tensor): Ground truth protein expression values, shape (batch_size, num_proteins)

        Returns:
            loss (torch.Tensor): Loss value
        """
        # 1. Construct the ground truth for expression status (0/1)
        target_presence = (target_expression > 0).float()

        regression_loss = F.mse_loss(predicted_expression_values, target_expression)
        # weights = torch.where(target_expression <= 0.5, torch.tensor(3.0), torch.tensor(1.0))
        # regression_loss = torch.mean(weights * F.mse_loss(predicted_expression_values, target_expression, reduction="none"))

        mmd_loss = self.mmd(predicted_expression_values.detach(), target_expression.detach())

        loss = self.regression_weight * regression_loss + \
               self.mmd_weight * mmd_loss 

        return loss, regression_loss, mmd_loss
    

def calculate_accuracy(predictions, targets, threshold=0):
    targets = (targets > 0)
    gene_accuracies = (
        ((predictions > threshold).float() == targets).float().mean(dim=0).cpu().numpy()
    )
    return np.mean(gene_accuracies), np.max(gene_accuracies), np.min(gene_accuracies)


# def calculate_correlations(predictions, targets):

#     if not isinstance(predictions, np.ndarray):
#         predictions = predictions.cpu().numpy()
#     if not isinstance(targets, np.ndarray):
#         targets = targets.cpu().numpy()

#     num_proteins = predictions.shape[1]
#     correlations = []

#     for j in range(num_proteins):
#         pred_j = predictions[:, j]
#         target_j = targets[:, j]

#         mask = target_j != 0
#         if mask.sum() < 2:
#             correlations.append((0.0, 0.0))
#             continue

#         pred_valid = pred_j[mask]
#         target_valid = target_j[mask]

#         try:
#             pearson_corr, _ = pearsonr(pred_valid, target_valid)
#         except Exception:
#             pearson_corr = 0.0
#         try:
#             spearman_corr, _ = spearmanr(pred_valid, target_valid)
#         except Exception:
#             spearman_corr = 0.0

#         correlations.append((pearson_corr, spearman_corr))

#     return correlations

def calculate_correlations(predictions, targets):
    """Calculates Pearson and Spearman correlations for each gene.
    
    Returns 0 if either predictions or targets are all zeros for a given gene.
    """
    num_genes = predictions.shape[1]  # Assuming genes are in the second dimension
    correlations = []
    for j in range(num_genes):
        gene_predictions = predictions[:, j].cpu().numpy()
        gene_targets = targets[:, j].cpu().numpy()

        # Check if either gene_predictions or gene_targets are all zeros
        if np.all(gene_predictions == 0) or np.all(gene_targets == 0):
            correlations.append((0, 0))
        else:
            # Calculate Pearson and Spearman correlations
            pearson_corr, _ = pearsonr(gene_predictions, gene_targets)
            spearman_corr, _ = spearmanr(gene_predictions, gene_targets)
            correlations.append((pearson_corr, spearman_corr))
    return correlations


def calculate_metrics(preds, targets):
    """
    preds/targets: (N, Num_Proteins) torch tensors
    Returns: (mean_pcc, mean_scc, list_pcc, list_scc)
    """
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    num_proteins = preds.shape[1]
    pccs = []
    sccs = []

    for i in range(num_proteins):
        p = preds[:, i]
        t = targets[:, i]
        if np.std(p) < 1e-6 or np.std(t) < 1e-6:
            pcc, scc = 0.0, 0.0
        else:
            pcc, _ = pearsonr(p, t)
            scc, _ = spearmanr(p, t)
        pccs.append(pcc)
        sccs.append(scc)

    return np.mean(pccs), np.mean(sccs), pccs, sccs


def train_and_evaluate(model, protein_model, train_loader, test_loader, optimizer, scheduler, train_protein, num_epochs=20):
    """
    Trains and evaluates the AttentionFusionModel using Accelerate.

    Args:
        model: The AttentionFusionModel instance.
        train_dataset: The training dataset.
        test_dataset: The test dataset.
        num_epochs: The number of training epochs.
        batch_size: The batch size.
        learning_rate: The learning rate.
    """

    # # Initialize Accelerator
    # accelerator = Accelerator(mixed_precision="bf16")

    # Define loss function and optimizer
    rbf_kernel = RBF(bandwidth=1.0, learnable_bandwidth=True)
    mmd_loss_fn = MMDLoss(rbf_kernel)
    criterion = ZILNLoss(mmd_weight=1)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        train_loss = 0.0
        train_flow_loss = 0.0
        train_reg_loss = 0.0
        train_mmd_loss = 0.0
        all_train_predictions = []
        all_train_targets = []

        # Training loop
        for batch_idx, (
            image,
            protein_exp,
            protein_emb,
            input_protein_ids,
            protein_expressions,
            src_key_padding_mask_protein,
            hist_features,
            hist_celltype_prob,
            pred_rna_emb,
            pred_prot_emb,
        ) in enumerate(train_loader):

            optimizer.zero_grad()  # Zero the gradients
            protein_cell_embeddings = accelerator.unwrap_model(protein_model)._encode(
                input_protein_ids,
                protein_expressions,
                src_key_padding_mask=src_key_padding_mask_protein,
                batch_labels=None
            )
            protein_cell_embeddings = protein_cell_embeddings[:, 0, :]

            with accelerator.autocast():
                pred_prot_emb_out, expression_predictions = model(
                    image,
                    protein_emb,
                    hist_features,
                    hist_celltype_prob,
                    pred_rna_emb,
                    pred_prot_emb,
                )  # Forward pass
                loss_pred, regression_loss, mmd_loss = criterion(expression_predictions, protein_exp)  # Calculate the loss

            # loss_final is a monitoring metric only (pred_prot_emb is static/cached)
            with torch.no_grad():
                flow_metric = cosine_similarity_loss(pred_prot_emb_out, protein_cell_embeddings)

            loss = loss_pred
            accelerator.backward(loss)  # Backpropagation
            optimizer.step()  # Update the weights

            train_loss += loss.item()
            train_reg_loss += regression_loss.item()
            train_mmd_loss += mmd_loss.item()
            train_flow_loss += flow_metric.item()

            # expression_predictions[expression_predictions < 0] = 0
            all_train_predictions.append(expression_predictions.detach())
            all_train_targets.append(protein_exp.detach())

        avg_train_loss = train_loss / len(train_loader)
        avg_train_flow_loss = train_flow_loss / len(train_loader)
        avg_train_reg_loss = train_reg_loss / len(train_loader)
        avg_train_mmd_loss = train_mmd_loss / len(train_loader)

        # Calculate training correlations
        all_train_predictions = torch.cat(all_train_predictions, dim=0)
        all_train_targets = torch.cat(all_train_targets, dim=0)
        all_train_predictions = accelerator.gather_for_metrics(all_train_predictions)
        all_train_targets = accelerator.gather_for_metrics(all_train_targets)
        train_correlations = calculate_correlations(all_train_predictions, all_train_targets)
        train_pearson_corrs, train_spearman_corrs = zip(*train_correlations)

        avg_train_pearson = np.mean(train_pearson_corrs)
        avg_train_spearman = np.mean(train_spearman_corrs)
        max_train_pearson = np.max(train_pearson_corrs)
        min_train_pearson = np.min(train_pearson_corrs)
        max_train_spearman = np.max(train_spearman_corrs)
        min_train_spearman = np.min(train_spearman_corrs)

        avg_accuracy, max_accuracy, min_accuracy = calculate_accuracy(all_train_predictions, all_train_targets)

        # if (epoch + 1) % 5 == 0:
        #     all_train_predictions = pd.DataFrame(all_train_predictions.cpu().numpy(), columns=train_protein)
        #     all_train_targets = pd.DataFrame(all_train_targets.cpu().numpy(), columns=train_protein)
        #     all_train_predictions.to_csv(f'result/train_prediction_{epoch+1}.csv')
        #     all_train_targets.to_csv(f'result/train_target_{epoch+1}.csv')


        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Training Flow Loss: {avg_train_flow_loss:.4f}, Training Regression Loss: {avg_train_reg_loss:.4f}, Training MMD Loss: {avg_train_mmd_loss:.4f}, "
                  f"Avg Train Pearson Correlation: {avg_train_pearson:.4f}, "
                  f"Avg Train Spearman Correlation: {avg_train_spearman:.4f}, "
                  f"Max Train Pearson Correlation: {max_train_pearson:.4f}, "
                  f"Min Train Pearson Correlation: {min_train_pearson:.4f}, "
                  f"Max Train Spearman Correlation: {max_train_spearman:.4f}, "
                  f"Min Train Spearman Correlation: {min_train_spearman:.4f}, accuracy: {avg_accuracy}")

        # Evaluation every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()  # Set the model to evaluation mode
            results = []
            epoch_avg_pearsons = []
            epoch_avg_spearmans = []
            all_samples_list_pcc = []
            all_samples_list_scc = []

            with torch.no_grad():  # Disable gradient calculation during evaluation
                for sample_name, dataloader in test_loader.items():
                    test_loss = 0.0
                    all_test_predictions = []
                    all_test_targets = []
                    for (
                        image,
                        protein_exp,
                        protein_emb,
                        input_protein_ids,
                        protein_expressions,
                        src_key_padding_mask_protein,
                        hist_features,
                        hist_celltype_prob,
                        pred_rna_emb,
                        pred_prot_emb,
                    ) in dataloader:
                        with accelerator.autocast():
                            pred_prot_emb_out, expression_predictions = model(
                                image,
                                protein_emb,
                                hist_features,
                                hist_celltype_prob,
                                pred_rna_emb,
                                pred_prot_emb,
                            )  # Forward pass
                            protein_exp = protein_exp.to(expression_predictions.device)
                            loss, regression_loss, mmd_loss = criterion(expression_predictions, protein_exp)  # Loss计算
                        test_loss += loss.item()
                        all_test_predictions.append(expression_predictions.detach())
                        all_test_targets.append(protein_exp.detach())
                    all_test_predictions = torch.cat(all_test_predictions, dim=0)
                    all_test_targets = torch.cat(all_test_targets, dim=0)
                    # Gather metrics across devices if using distributed training
                    all_test_predictions = accelerator.gather_for_metrics(all_test_predictions)
                    all_test_targets = accelerator.gather_for_metrics(all_test_targets)

                    mean_pcc, mean_scc, list_pcc, list_scc = calculate_metrics(all_test_predictions, all_test_targets)
                    epoch_avg_pearsons.append(mean_pcc)
                    epoch_avg_spearmans.append(mean_scc)
                    results.append({
                        'sample_name': sample_name,
                        'avg_pearson': mean_pcc,
                        'max_pearson': np.max(list_pcc),
                        'min_pearson': np.min(list_pcc),
                        'avg_spearman': mean_scc,
                        'max_spearman': np.max(list_scc),
                        'min_spearman': np.min(list_scc)
                    })
                    all_samples_list_pcc.append(np.array(list_pcc, dtype=np.float32))
                    all_samples_list_scc.append(np.array(list_scc, dtype=np.float32))

                    if (epoch + 1) == num_epochs and accelerator.is_main_process:
                        save_dir = 'result/corr'
                        df_corr = pd.DataFrame({
                            'Pearson': list_pcc,
                            'Spearman': list_scc
                        }, index=train_protein)

                        csv_path = os.path.join(save_dir, f"{sample_name}.csv")
                        df_corr.to_csv(csv_path)
                        if sample_name == '19510_P37-S83_C40_US_SCAN_OR_001':
                            all_test_predictions = pd.DataFrame(all_test_predictions.cpu().numpy())
                            all_test_targets = pd.DataFrame(all_test_targets.cpu().numpy())
                            all_test_predictions.to_csv(os.path.join("result", f"{sample_name}_pred.csv"))
                            all_test_targets.to_csv(os.path.join("result", f"{sample_name}_target.csv"))

            if accelerator.is_main_process:
                print(f"\n" + "="*30 + f" Epoch {epoch+1} Test Summary " + "="*30)

                global_avg_p = np.mean(epoch_avg_pearsons)
                global_max_p = np.max(epoch_avg_pearsons)
                global_min_p = np.min(epoch_avg_pearsons)

                global_avg_s = np.mean(epoch_avg_spearmans)
                global_max_s = np.max(epoch_avg_spearmans)
                global_min_s = np.min(epoch_avg_spearmans)

                print(f"[GLOBAL ALL SAMPLES]")
                print(f"Pearson  -> Global Avg: {global_avg_p:.4f} | Best Sample Avg: {global_max_p:.4f} | Worst Sample Avg: {global_min_p:.4f}")
                print(f"Spearman -> Global Avg: {global_avg_s:.4f} | Best Sample Avg: {global_max_s:.4f} | Worst Sample Avg: {global_min_s:.4f}")
                print("-" * 20)

                print(f"{'Sample Name':<25} | {'Avg Pearson':<12} | {'Max Pearson':<12} | {'Min Pearson':<12} | {'Avg Spearman':<12}")
                print("-" * 20)
                for r in results:
                    print(f"{r['sample_name']:<25} | {r['avg_pearson']:<12.4f} | {r['max_pearson']:<12.4f} | {r['min_pearson']:<12.4f} | {r['avg_spearman']:<12.4f}")

                # ===== Per-protein PCC across all test samples =====
                all_samples_list_pcc = np.stack(all_samples_list_pcc, axis=0)  # (num_samples, num_proteins)

                # Mean Pearson per protein across all test samples
                protein_mean_pcc = np.mean(all_samples_list_pcc, axis=0)       # (num_proteins,)

                # Stats over protein dimension
                protein_mean_pcc_avg = float(np.mean(protein_mean_pcc))
                protein_mean_pcc_median = float(np.median(protein_mean_pcc))

                print("\n[Per-protein PCC across all test samples]")
                print(f"Protein-wise mean PCC -> Mean over proteins: {protein_mean_pcc_avg:.4f}, Median over proteins: {protein_mean_pcc_median:.4f}")

                # Print per-protein PCC using train_protein names
                print("\nProtein\t\tMean_PCC(over samples)")
                for i, pname in enumerate(train_protein):
                    print(f"{pname}\t\t{protein_mean_pcc[i]:.4f}")

                print("="*20 + "\n")


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

def freeze_model_parameters(model):
  for param in model.parameters():
    param.requires_grad = False


class LoRALinear(nn.Module):
    """LoRA adapter for nn.Linear layers: h = Wx + (alpha/r) * B A x.

    Freezes the original weight W and only trains low-rank matrices A and B,
    dramatically reducing trainable parameters.
    """
    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank

        # Original frozen linear layer
        self.linear = linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        out_features, in_features = linear.weight.shape
        # LoRA low-rank matrices
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))

        # Initialization: Kaiming uniform for A, zeros for B (so delta = 0 at start)
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        lora_update = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return self.linear(x) + lora_update * self.scaling


def apply_lora_to_linear_children(module: nn.Module, rank: int = 8,
                                   alpha: float = 16.0, dropout: float = 0.0):
    """Recursively replace all nn.Linear children of `module` with LoRALinear."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha,
                                              dropout=dropout))
        else:
            apply_lora_to_linear_children(child, rank=rank, alpha=alpha,
                                           dropout=dropout)


def enable_hist_model_lora(hist_model, lora_rank=8, lora_alpha=16.0, lora_dropout=0.0):
    """Enable LoRA fine-tuning on the last 2 ViT blocks of hist_model.

    Follows pred_imgonly_new.py lines 660-683:
      - Freeze entire hist_model first.
      - Apply LoRA to attention Linear layers in the last 2 blocks.
      - Full fine-tune non-attention submodules in the last 2 blocks.
      - Unfreeze attn_pool_contrast and ln_contrast.
    """
    # 1. Freeze entire hist_model
    freeze_model_parameters(hist_model)

    # 2. Get ViT blocks
    blocks = hist_model.trunk.blocks

    # 3. Only process last 2 blocks
    for i, block in enumerate(blocks):
        if i >= len(blocks) - 2:
            # Apply LoRA to attention Linear layers (qkv, proj)
            if hasattr(block, 'attn'):
                apply_lora_to_linear_children(
                    block.attn,
                    rank=lora_rank,
                    alpha=lora_alpha,
                    dropout=lora_dropout,
                )
            # Full fine-tuning for all other submodules in this block
            for name, child in block.named_children():
                if name != 'attn':
                    for param in child.parameters():
                        param.requires_grad = True

    # 4. Unfreeze attn_pool_contrast and ln_contrast
    for param in hist_model.attn_pool_contrast.parameters():
        param.requires_grad = True
    for param in hist_model.ln_contrast.parameters():
        param.requires_grad = True


def build_static_hist_flow_cache(
    dataset,
    cache_path,
    hist_flow_rna,
    rna_flow,
    batch_size=512,
    steps=60,
    device="cuda",
):
    """Pre-compute static features from frozen pretrained models and save to cache.

    Uses the dataset's image loading pipeline (center-crop, normalize) to ensure
    consistency with training. Outputs a single .pt dict:

        {
            "image_paths": list[str],
            "hist_features": [N, 512],
            "hist_celltype_prob": [N, n_celltypes],
            "pred_rna_emb": [N, 512],
            "pred_prot_emb": [N, 512],
        }

    Args:
        dataset: ProteinExpressionDataset instance (without static_feature_path).
        cache_path: Path to save the .pt cache file.
        hist_flow_rna: Frozen HistFlowRNA model.
        rna_flow: Frozen ConditionalFlowNet for RNA→protein transformation.
        batch_size: Batch size for the temporary DataLoader.
        steps: Number of RK4 integration steps.
        device: Device for inference.
    """
    if os.path.exists(cache_path):
        print(f"Static cache already exists at {cache_path}, skipping generation.")
        return

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)

    hist_flow_rna.eval()
    rna_flow.eval()

    hist_flow_rna = hist_flow_rna.to(device)
    rna_flow = rna_flow.to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
    )

    all_hist_features = []
    all_hist_celltype_prob = []
    all_pred_rna_emb = []
    all_pred_prot_emb = []

    with torch.no_grad():
        for batch in tqdm(loader):
            image = batch[0].to(device)

            # Get histology features via frozen HistFlowRNA
            image_latent, pred_rna_emb, hist_features, hist_celltype_prob = (
                hist_flow_rna.encode_image_to_rna_emb(image, steps=steps)
            )

            # Generate pred_prot_emb using frozen RNA→protein flow
            pred_prot_emb = integrate_flow(
                rna_flow,
                pred_rna_emb,
                steps=steps,
                cond_vec=hist_celltype_prob,
            )

            all_hist_features.append(hist_features.detach().cpu().float())
            all_hist_celltype_prob.append(hist_celltype_prob.detach().cpu().float())
            all_pred_rna_emb.append(pred_rna_emb.detach().cpu().float())
            all_pred_prot_emb.append(pred_prot_emb.detach().cpu().float())

    cache = {
        "image_paths": list(dataset.image_paths),
        "hist_features": torch.cat(all_hist_features, dim=0),
        "hist_celltype_prob": torch.cat(all_hist_celltype_prob, dim=0),
        "pred_rna_emb": torch.cat(all_pred_rna_emb, dim=0),
        "pred_prot_emb": torch.cat(all_pred_prot_emb, dim=0),
    }
    torch.save(cache, cache_path)
    print(f"Static cache saved to {cache_path} "
          f"({len(cache['image_paths'])} samples, "
          f"hist_features: {cache['hist_features'].shape}, "
          f"pred_rna_emb: {cache['pred_rna_emb'].shape}, "
          f"pred_prot_emb: {cache['pred_prot_emb'].shape})")


if __name__ == '__main__':

    batch_size = 2560  # 5120

    train_df = pd.read_csv('../predict_exp/data/train_norm.csv')
    val_df = pd.read_csv('../predict_exp/data/val_norm_filtered.csv')
    train_df.columns = ['new_filenames', 'PECAM1', 'PTPRC', 'CD68', 'CD4', 'FOXP3', 'CD8A', 'PTPRCRO', 'MS4A1', 'CD274', 'CD3E', 'CD163', 'CDH1', 'MKI67', 'KRT19', 'ACTA2']
    val_df.columns = ['new_filenames', 'PECAM1', 'PTPRC', 'CD68', 'CD4', 'FOXP3', 'CD8A', 'PTPRCRO', 'MS4A1', 'CD274', 'CD3E', 'CD163', 'CDH1', 'MKI67', 'KRT19', 'ACTA2']

    val_df['sample_name'] = val_df['new_filenames'].apply(lambda x: x.split('/')[1].split('__')[0])
    sample_dfs = {
        sample: group.drop(columns=['sample_name']).reset_index(drop=True)
        for sample, group in val_df.groupby('sample_name')
    }

    # === Step 1: Create raw datasets (no static features) for cache generation ===
    train_dataset_raw = ProteinExpressionDataset(
        train_df,
        scgpt_model_path='/macroverse-nas/pjz/proteomics/data_model/scgpt_model',
        protein_emb_path='/macroverse-nas/pjz/proteomics/train_brca/pickle/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2_new.pt',
    )
    train_protein = train_dataset_raw.get_protein_names()

    val_dataset_raw = {}
    for sample_name, sample_df in sample_dfs.items():
        val_dataset_raw[sample_name] = ProteinExpressionDataset(
            sample_df,
            scgpt_model_path='/macroverse-nas/pjz/proteomics/data_model/scgpt_model',
            protein_emb_path='/macroverse-nas/pjz/proteomics/train_brca/pickle/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2_new.pt',
        )

    # === Step 2: Load pretrained models ===
    scgpt_model_path = "/macroverse-nas/pjz/proteomics/data_model/scgpt_model"
    coach_checkpoint_path = "/macroverse-nas/pjz/proteomics/data_model/coach_model/pytorch_model.bin"
    reference_embedding_path = "/macroverse-nas/pjz/crc_codex/pretrain/data/reference_embedding.csv"

    spatial_rna_model = build_scgpt_model()
    protein_model = build_scgpt_model()

    model_cfg = 'conch_ViT-B-16'
    path_model_full, preprocess = create_model_from_pretrained(
        model_cfg,
        coach_checkpoint_path,
        force_image_size=256
    )
    path_model_visual = path_model_full.visual

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
        flow_model=flow_model
    )
    hist_flow_rna = load_pretrained_model(
        model=hist_flow_rna,
        pretrained_weights_path='/macroverse-nas/pjz/crc_codex/pretrain/pickle/HistFlowRNA_final_kl_new.pt',
        device='cpu',
    )

    # RNA→protein flow (frozen, used only for cache generation)
    flow_model_prot = ConditionalFlowNet(512, 512, 40)
    for p in flow_model_prot.parameters():
        p.requires_grad = False

    ## === Step 3: Generate static feature caches (BEFORE LoRA, from original pretrained weights) ===
    os.makedirs("/macroverse-nas/pjz/crc_codex/codex/cache", exist_ok=True)

    build_static_hist_flow_cache(
        dataset=train_dataset_raw,
        cache_path="/macroverse-nas/pjz/crc_codex/codex/cache/train_hist_flow_static.pt",
        hist_flow_rna=hist_flow_rna,
        rna_flow=flow_model_prot,
        batch_size=4096,
        steps=60,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    for sample_name, ds_raw in val_dataset_raw.items():
        build_static_hist_flow_cache(
            dataset=ds_raw,
            cache_path=f"/macroverse-nas/pjz/crc_codex/codex/cache/val_{sample_name}_hist_flow_static.pt",
            hist_flow_rna=hist_flow_rna,
            rna_flow=flow_model_prot,
            batch_size=5120,
            steps=60,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    # === Step 4: Re-create datasets with static feature caches ===
    train_dataset = ProteinExpressionDataset(
        train_df,
        scgpt_model_path='/macroverse-nas/pjz/proteomics/data_model/scgpt_model',
        protein_emb_path='/macroverse-nas/pjz/proteomics/train_brca/pickle/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2_new.pt',
        static_feature_path="/macroverse-nas/pjz/crc_codex/codex/cache/train_hist_flow_static.pt",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )

    val_dataset = {}
    val_loader = {}
    for sample_name, sample_df in sample_dfs.items():
        dataset = ProteinExpressionDataset(
            sample_df,
            scgpt_model_path='/macroverse-nas/pjz/proteomics/data_model/scgpt_model',
            protein_emb_path='/macroverse-nas/pjz/proteomics/train_brca/pickle/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2_new.pt',
            static_feature_path=f"/macroverse-nas/pjz/crc_codex/codex/cache/val_{sample_name}_hist_flow_static.pt",
        )
        dataloader = DataLoader(
            dataset,
            batch_size=4096,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            drop_last=False,
        )
        val_dataset[sample_name] = dataset
        val_loader[sample_name] = dataloader

    # === Step 5: Freeze hist_flow_rna, then enable LoRA on hist_model ===
    for p in hist_flow_rna.parameters():
        p.requires_grad = False

    enable_hist_model_lora(
        hist_flow_rna.hist_model,
        lora_rank=8,
        lora_alpha=16.0,
        lora_dropout=0.0,
    )

    # === Step 6: Construct model and optimizer (only trainable params) ===
    model = ImageToProteinModel(
        hist_flow_rna, flow_model_prot, 512, 5120, 15, train_protein
    )

    optimizer = optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=0.0015,
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # === Step 7: Accelerator prepare ===
    model, protein_model, optimizer, scheduler, train_loader = accelerator.prepare(
        model,
        protein_model,
        optimizer,
        scheduler,
        train_loader,
    )

    # Prepare validation loaders
    val_loader = {
        sample_name: accelerator.prepare(dataloader)
        for sample_name, dataloader in val_loader.items()
    }

    train_and_evaluate(
        model, protein_model, train_loader, val_loader,
        optimizer, scheduler, train_protein, num_epochs=30,
    )