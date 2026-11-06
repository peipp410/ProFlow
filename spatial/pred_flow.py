import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from pretrained_model import HistFlowRNA, ConditionalFlowNet, load_pretrained_model
from dataset import ProteinExpressionDataset
import os
import argparse

from scgpt.model import TransformerModel
from scgpt.utils import load_pretrained
from scgpt.tokenizer import GeneVocab
from conch.open_clip_custom import create_model_from_pretrained
from conch.open_clip_custom.factory import CFG_DIR

import numpy as np
import pandas as pd
from pathlib import Path
import json
from scipy.stats import pearsonr, spearmanr

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(
    gradient_accumulation_steps=1,  # Let Accelerate figure out the correct value
    log_with="tensorboard",
    project_dir="./logs",
    kwargs_handlers=[kwargs]
)


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



        self.prediction_layers_expression = nn.Sequential(
            nn.Linear(spot_dim, spot_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(spot_dim // 4, spot_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(spot_dim // 2, 1)
        )
        self.protein_names = list(protein_names) if protein_names is not None else None

    def forward(self, image, protein_emb, hist_features, hist_celltype_prob,
                pred_rna_emb, pred_prot_emb, steps=60):

        image_latent, _ = self.hist_flow_rna.hist_model(image)

        image_latent = self.image_ln(image_latent)
        pred_rna_emb = F.layer_norm(pred_rna_emb, (pred_rna_emb.size(-1),))
        pred_prot_emb_norm = F.layer_norm(pred_prot_emb, (pred_prot_emb.size(-1),))

        spot_embeddings = torch.cat(
            (image_latent, pred_rna_emb, pred_prot_emb_norm), dim=-1
        )  # [B, spot_dim*3]
        protein_projected = self.protein_projection(protein_emb)  # [B, num_proteins, spot_dim]
        spot_embeddings = spot_embeddings.unsqueeze(1).repeat(1, self.num_proteins, 1)
        x = torch.cat((protein_projected, spot_embeddings), dim=-1)  # [B, num_proteins, spot_dim*4]
        x = self.trans_dim(x)  # [B, num_proteins, spot_dim]

        expr_preds = self.prediction_layers_expression(x).squeeze(-1)

        return pred_prot_emb, expr_preds


class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None, learnable_bandwidth=False):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        if bandwidth is None:
            self.bandwidth = None
        else:
            self.bandwidth = nn.Parameter(torch.tensor(bandwidth, dtype=torch.float)) if learnable_bandwidth else torch.tensor(bandwidth, dtype=torch.float) # Make bandwidth learnable

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
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
        regression_loss = F.mse_loss(predicted_expression_values, target_expression)

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











def calculate_correlations(predictions, targets):
    """Calculates Pearson and Spearman correlations for each gene.
    
    Returns 0 if either predictions or targets are all zeros for a given gene.
    """
    num_genes = predictions.shape[1]  # Assuming genes are in the second dimension
    correlations = []
    for j in range(num_genes):
        gene_predictions = predictions[:, j].cpu().numpy()
        gene_targets = targets[:, j].cpu().numpy()

        if np.all(gene_predictions == 0) or np.all(gene_targets == 0):
            correlations.append((0, 0))
        else:
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


def train_and_evaluate(model, protein_model, train_loader, test_loader, optimizer, train_protein, num_epochs=20):
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


    criterion = ZILNLoss(mmd_weight=1)

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        train_loss = 0.0
        train_flow_loss = 0.0
        train_reg_loss = 0.0
        train_mmd_loss = 0.0
        all_train_predictions = []
        all_train_targets = []

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
        ) in train_loader:

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

            with torch.no_grad():
                flow_metric = cosine_similarity_loss(pred_prot_emb_out, protein_cell_embeddings)

            loss = loss_pred
            accelerator.backward(loss)  # Backpropagation
            optimizer.step()  # Update the weights

            train_loss += loss.item()
            train_reg_loss += regression_loss.item()
            train_mmd_loss += mmd_loss.item()
            train_flow_loss += flow_metric.item()

            all_train_predictions.append(expression_predictions.detach())
            all_train_targets.append(protein_exp.detach())

        avg_train_loss = train_loss / len(train_loader)
        avg_train_flow_loss = train_flow_loss / len(train_loader)
        avg_train_reg_loss = train_reg_loss / len(train_loader)
        avg_train_mmd_loss = train_mmd_loss / len(train_loader)

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



        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Training Flow Loss: {avg_train_flow_loss:.4f}, Training Regression Loss: {avg_train_reg_loss:.4f}, Training MMD Loss: {avg_train_mmd_loss:.4f}, "
                  f"Avg Train Pearson Correlation: {avg_train_pearson:.4f}, "
                  f"Avg Train Spearman Correlation: {avg_train_spearman:.4f}, "
                  f"Max Train Pearson Correlation: {max_train_pearson:.4f}, "
                  f"Min Train Pearson Correlation: {min_train_pearson:.4f}, "
                  f"Max Train Spearman Correlation: {max_train_spearman:.4f}, "
                  f"Min Train Spearman Correlation: {min_train_spearman:.4f}, accuracy: {avg_accuracy}")

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

                all_samples_list_pcc = np.stack(all_samples_list_pcc, axis=0)  # (num_samples, num_proteins)

                protein_mean_pcc = np.mean(all_samples_list_pcc, axis=0)       # (num_proteins,)

                protein_mean_pcc_avg = float(np.mean(protein_mean_pcc))
                protein_mean_pcc_median = float(np.median(protein_mean_pcc))

                print("\n[Per-protein PCC across all test samples]")
                print(f"Protein-wise mean PCC -> Mean over proteins: {protein_mean_pcc_avg:.4f}, Median over proteins: {protein_mean_pcc_median:.4f}")

                print("\nProtein\t\tMean_PCC(over samples)")
                for i, pname in enumerate(train_protein):
                    print(f"{pname}\t\t{protein_mean_pcc[i]:.4f}")

                print("="*20 + "\n")


def build_scgpt_model(model_dir):
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


def enable_hist_model_mlp(hist_model):
    """Enable full fine-tuning on the last 2 ViT MLP layers of hist_model,
    with attention layers frozen (no gradients).

      - Freeze entire hist_model first.
      - Keep attention layers frozen in the last 2 blocks.
      - Full fine-tune non-attention submodules (MLP, norms) in the last 2 blocks.
      - Unfreeze attn_pool_contrast and ln_contrast.
    """
    freeze_model_parameters(hist_model)

    blocks = hist_model.trunk.blocks

    for i, block in enumerate(blocks):
        if i >= len(blocks) - 2:
            for name, child in block.named_children():
                if name != 'attn':  # skip attention submodule (keep frozen)
                    for param in child.parameters():
                        param.requires_grad = True

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
    overwrite=False,
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
    if os.path.exists(cache_path) and not overwrite:
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
        for batch in loader:
            image = batch[0].to(device)

            image_latent, pred_rna_emb, hist_features, hist_celltype_prob = (
                hist_flow_rna.encode_image_to_rna_emb(image, steps=steps)
            )

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



def verify_static_cache(dataset, hist_flow_rna, rna_flow, steps=60, sample_count=8):
    """Check legacy cache samples against the weights to be exported for inference.

    This is a numerical spot check, not proof of the provenance of a legacy cache.
    Use --rebuild-static-cache when the weights that produced it are uncertain.
    """
    if len(dataset) == 0:
        raise ValueError("Cannot verify an empty dataset.")
    indices = np.unique(np.linspace(0, len(dataset) - 1, min(sample_count, len(dataset)), dtype=int))
    hist_device = next(hist_flow_rna.parameters()).device
    flow_device = next(rna_flow.parameters()).device
    hist_flow_rna.eval().to(accelerator.device)
    rna_flow.eval().to(accelerator.device)
    try:
        images = torch.stack([dataset[int(i)][0] for i in indices]).to(accelerator.device)
        with torch.no_grad():
            _, pred_rna, features, probabilities = hist_flow_rna.encode_image_to_rna_emb(images, steps=steps)
            pred_protein = integrate_flow(rna_flow, pred_rna, steps=steps, cond_vec=probabilities)
        actual = {
            "hist_features": features,
            "hist_celltype_prob": probabilities,
            "pred_rna_emb": pred_rna,
            "pred_prot_emb": pred_protein,
        }
        for name, value in actual.items():
            expected = dataset.static_features[name][indices].float().cpu()
            value = value.float().cpu()
            if value.shape != expected.shape or not torch.allclose(value, expected, rtol=1e-4, atol=1e-5):
                raise ValueError(
                    f"Static cache {name} does not match the frozen inference weights. "
                    "Run this training script with --rebuild-static-cache to regenerate "
                    "the caches with the same weights that will be saved in the checkpoint."
                )
    finally:
        hist_flow_rna.to(hist_device)
        rna_flow.to(flow_device)
    return {"sample_indices": indices.tolist(), "rtol": 1e-4, "atol": 1e-5}


def make_cached_dataset(
    dataframe,
    cache_path,
    hist_flow_rna,
    rna_flow,
    scgpt_model_path,
    protein_embedding_path,
    image_root,
    crop_size,
    rebuild=False,
    cache_batch_size=64,
    flow_steps=60,
):
    dataset = ProteinExpressionDataset(
        dataframe,
        scgpt_model_path=scgpt_model_path,
        protein_emb_path=protein_embedding_path,
        image_root=image_root,
        crop_size=crop_size,
        static_feature_path=None if rebuild else cache_path,
    )
    if rebuild:
        if accelerator.is_main_process:
            build_static_hist_flow_cache(
                dataset, cache_path, hist_flow_rna, rna_flow,
                batch_size=cache_batch_size,
                steps=flow_steps,
                device=accelerator.device,
                overwrite=True,
            )
        accelerator.wait_for_everyone()
        dataset.static_features = torch.load(cache_path, map_location="cpu", weights_only=False)
        dataset._validate_static_features()
    dataset.inference_cache_check = verify_static_cache(
        dataset, hist_flow_rna, rna_flow, steps=flow_steps
    )
    return dataset


def _checkpoint_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _checkpoint_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_checkpoint_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_checkpoint_cpu(item) for item in value)
    return value


def build_inference_metadata(hist_flow_rna, rna_flow, dataset, model_cfg):
    """Snapshot the ORIGINAL frozen feature branch before image fine-tuning."""
    with open(Path(CFG_DIR) / f"{model_cfg}.json", "r") as handle:
        conch_config = json.load(handle)
    conch_config.pop("custom_text", None)
    conch_config["vision_cfg"]["image_size"] = 256
    feature_dim = int(hist_flow_rna.class_embeddings.shape[1])
    if rna_flow.cond_dim != hist_flow_rna.n_ct:
        raise ValueError("RNA/protein flow condition size must match the cell-type probabilities.")
    normalize = dataset.non_augmentations.transforms[-1]
    return {
        "format_version": 1,
        "architecture": "crc_codex.shared_mlp.cached_features.v1",
        "protein_names": list(dataset.get_protein_names()),
        "protein_embeddings": torch.as_tensor(dataset.protein_emb, dtype=torch.float32).clone(),
        "conch_config": conch_config,
        "model_config": {
            "spot_dim": feature_dim,
            "protein_dim": int(dataset.protein_emb.shape[1]),
            "num_proteins": len(dataset.get_protein_names()),
            "num_celltypes": int(hist_flow_rna.n_ct),
            "rna_flow_hidden_dim": hist_flow_rna.flow_model.net[0].out_features,
            "protein_flow_hidden_dim": rna_flow.net[0].out_features,
            "head_dropout": 0.1,
        },
        "preprocessing": {
            "patch_size": 256,
            "image_mean": list(normalize.mean),
            "image_std": list(normalize.std),
            "coordinate_units": "pixels in the supplied image (level 0)",
            "output_transform": "log1p",
            "expression_scale": "normalized expression from train_norm.csv",
            "flow_steps": 60,
        },
        "static_hist_flow_state_dict": {
            key: _checkpoint_cpu(value)
            for key, value in hist_flow_rna.state_dict().items()
            if not key.startswith("rna_model.")
        },
    }


def save_training_checkpoint(model, optimizer, epoch, metadata, path):
    accelerator.wait_for_everyone()
    model_state = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = dict(metadata)
        checkpoint.update({
            "epoch": int(epoch),
            "model_state_dict": _checkpoint_cpu(model_state),
            "optimizer_state_dict": _checkpoint_cpu(optimizer.state_dict()),
        })
        temporary = path.with_name(path.name + ".tmp")
        try:
            torch.save(checkpoint, temporary)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        print(f"Saved training/inference checkpoint: {path}")
    accelerator.wait_for_everyone()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the shared-MLP protein predictor and export its checkpoint."
    )
    parser.add_argument("--train-data", required=True, help="Training CSV")
    parser.add_argument("--val-data", required=True, help="Validation CSV")
    parser.add_argument("--checkpoint", required=True, help="Output checkpoint")
    parser.add_argument("--train-cache", required=True, help="Training static-feature cache")
    parser.add_argument("--val-cache-dir", required=True, help="Validation cache directory")
    parser.add_argument("--scgpt-model-path", required=True)
    parser.add_argument("--protein-embedding-path", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--reference-embedding", required=True)
    parser.add_argument("--conch-checkpoint", required=True)
    parser.add_argument("--hist-flow-checkpoint", required=True)
    parser.add_argument("--conch-model-config", default="conch_ViT-B-16")
    parser.add_argument("--protein-columns", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--val-batch-size", type=int, default=2048)
    parser.add_argument("--cache-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--spot-dim", type=int, default=512)
    parser.add_argument("--flow-hidden-dim", type=int, default=1024)
    parser.add_argument("--flow-steps", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.0015)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-static-cache", action="store_true")
    args = parser.parse_args()
    positive = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "val_batch_size": args.val_batch_size,
        "cache_batch_size": args.cache_batch_size,
        "num_workers": args.num_workers,
        "crop_size": args.crop_size,
        "spot_dim": args.spot_dim,
        "flow_hidden_dim": args.flow_hidden_dim,
        "flow_steps": args.flow_steps,
        "learning_rate": args.learning_rate,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        parser.error(f"these arguments must be positive: {', '.join(invalid)}")
    return args


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    set_seed(args.seed)
    columns = ["new_filenames", *args.protein_columns]
    train_df = pd.read_csv(args.train_data)
    val_df = pd.read_csv(args.val_data)
    if train_df.shape[1] != len(columns) or val_df.shape[1] != len(columns):
        raise ValueError("CSV column count must equal one image column plus --protein-columns")
    train_df.columns = columns
    val_df.columns = columns
    val_df["sample_name"] = val_df["new_filenames"].map(
        lambda value: Path(str(value)).parts[1].split("__")[0]
        if len(Path(str(value)).parts) > 1 else Path(str(value)).stem.split("__")[0]
    )
    sample_dfs = {
        sample: group.drop(columns=["sample_name"]).reset_index(drop=True)
        for sample, group in val_df.groupby("sample_name")
    }

    spatial_rna_model = build_scgpt_model(args.scgpt_model_path)
    protein_model = build_scgpt_model(args.scgpt_model_path)
    model_full, _ = create_model_from_pretrained(
        args.conch_model_config, args.conch_checkpoint, force_image_size=args.crop_size
    )
    class_embeddings = pd.read_csv(args.reference_embedding, index_col=0).values
    flow_model = ConditionalFlowNet(args.spot_dim, args.spot_dim, 0, args.flow_hidden_dim)
    hist_flow_rna = load_pretrained_model(
        HistFlowRNA(
            spatial_rna_model,
            model_full.visual,
            class_embeddings,
            flow_model,
            args.spot_dim,
        ),
        args.hist_flow_checkpoint,
        "cpu",
    )
    flow_model_prot = ConditionalFlowNet(
        args.spot_dim,
        args.spot_dim,
        class_embeddings.shape[0],
        args.flow_hidden_dim,
    )
    for parameter in flow_model_prot.parameters():
        parameter.requires_grad = False

    dataset_kwargs = {
        "scgpt_model_path": args.scgpt_model_path,
        "protein_embedding_path": args.protein_embedding_path,
        "image_root": args.image_root,
        "crop_size": args.crop_size,
        "cache_batch_size": args.cache_batch_size,
        "flow_steps": args.flow_steps,
        "rebuild": args.rebuild_static_cache,
    }
    train_dataset = make_cached_dataset(
        train_df, args.train_cache, hist_flow_rna, flow_model_prot, **dataset_kwargs
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dataset = {}
    val_loader = {}
    for sample_name, sample_df in sample_dfs.items():
        cache_path = Path(args.val_cache_dir) / f"val_{sample_name}_hist_flow_static.pt"
        dataset = make_cached_dataset(
            sample_df, cache_path, hist_flow_rna, flow_model_prot, **dataset_kwargs
        )
        if list(dataset.get_protein_names()) != list(train_dataset.get_protein_names()):
            raise ValueError(f"Protein order differs for validation sample {sample_name}")
        val_dataset[sample_name] = dataset
        val_loader[sample_name] = DataLoader(
            dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    inference_metadata = None
    if accelerator.is_main_process:
        inference_metadata = build_inference_metadata(
            hist_flow_rna, flow_model_prot, train_dataset, args.conch_model_config
        )
        inference_metadata["cache_checks"] = {
            "train": train_dataset.inference_cache_check,
            "validation": {name: ds.inference_cache_check for name, ds in val_dataset.items()},
            "rebuilt_this_run": args.rebuild_static_cache,
        }

    for parameter in hist_flow_rna.parameters():
        parameter.requires_grad = False
    enable_hist_model_mlp(hist_flow_rna.hist_model)
    train_protein = train_dataset.get_protein_names()
    model = ImageToProteinModel(
        hist_flow_rna,
        flow_model_prot,
        args.spot_dim,
        train_dataset.protein_emb.shape[1],
        len(train_protein),
        train_protein,
    )
    optimizer = optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    model, protein_model, optimizer, train_loader = accelerator.prepare(
        model, protein_model, optimizer, train_loader
    )
    val_loader = {
        name: accelerator.prepare(loader) for name, loader in val_loader.items()
    }
    train_and_evaluate(
        model,
        protein_model,
        train_loader,
        val_loader,
        optimizer,
        train_protein,
        num_epochs=args.epochs,
    )
    save_training_checkpoint(
        model, optimizer, args.epochs, inference_metadata, args.checkpoint
    )


if __name__ == "__main__":
    main()
