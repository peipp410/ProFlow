import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torch.optim as optim
from lora_pytorch import LoRA
from itertools import cycle
from pretrained_model import HistFlowRNA, ConditionalFlowNet, load_pretrained_model
from dataset import SpatialDataset

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
    def __init__(self, path_model_visual, rna_model, protein_model, rna_flow, spot_dim, protein_dim, num_proteins, protein_names=None):
        super(ImageToProteinModel, self).__init__()
        self.path_model_visual = path_model_visual
        self.rna_model = rna_model
        self.protein_model = protein_model
        self.rna_flow = rna_flow
        self.num_proteins = num_proteins

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

    def forward(self, image, input_gene_ids, expressions, src_key_padding_mask, 
        protein_emb, enrichment, steps=60):

        device = image.device

        image_latent, _ = accelerator.unwrap_model(self.path_model_visual)(image)

        cell_embeddings = accelerator.unwrap_model(self.rna_model)._encode(
            input_gene_ids,
            expressions,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None
        )
        cell_embeddings = cell_embeddings[:, 0, :]

        # === 2. RNA embedding -> Protein embedding (cond_vec = one-hot sample_group) ===
        cond_vec = enrichment

        try:
            pred_prot_emb = integrate_flow(self.rna_flow, cell_embeddings, steps=steps, cond_vec=cond_vec)
        except:
            print(cell_embeddings.shape, cond_vec.shape)

        # === 3. integration ===
        spot_embeddings = torch.cat((image_latent, cell_embeddings, pred_prot_emb), dim=-1)  # [B, spot_dim*3]

        protein_projected = self.protein_projection(protein_emb)  # [B, num_proteins, spot_dim]
        spot_embeddings = spot_embeddings.unsqueeze(1).repeat(1, self.num_proteins, 1)
        x = torch.cat((protein_projected, spot_embeddings), dim=-1)  # [B, num_proteins, spot_dim*4]
        x = self.trans_dim(x)  # [B, num_proteins, spot_dim]

        # === 4. prediction ===
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
        regression_loss = F.mse_loss(predicted_expression_values, target_expression)

        # # 1. Construct the ground truth for expression status (0/1)
        # target_presence = (target_expression > 0).float()

        # # regression_loss = F.mse_loss(predicted_expression_values, target_expression)
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
        for batch_idx, (image, input_gene_ids, expressions, src_key_padding_mask, input_protein_ids, protein_expressions, src_key_padding_mask_protein, protein_expression_ori, protein_emb, enrichment, indices) in enumerate(train_loader):

            optimizer.zero_grad()  # Zero the gradients
            protein_cell_embeddings = accelerator.unwrap_model(protein_model)._encode(
                input_protein_ids,
                protein_expressions,
                src_key_padding_mask=src_key_padding_mask_protein,
                batch_labels=None
            )
            protein_cell_embeddings = protein_cell_embeddings[:, 0, :]

            with accelerator.autocast():
                pred_prot_emb, expression_predictions = model(image, input_gene_ids, expressions, src_key_padding_mask, protein_emb, enrichment)  # Forward pass
                loss_final = cosine_similarity_loss(pred_prot_emb, protein_cell_embeddings) + mmd_loss_fn(pred_prot_emb, protein_cell_embeddings)
                loss_pred, regression_loss, mmd_loss = criterion(expression_predictions, protein_expression_ori)  # Calculate the loss

            loss = loss_final + loss_pred
            accelerator.backward(loss)  # Backpropagation
            optimizer.step()  # Update the weights

            # if epoch < num_epochs // 2:
            #     pass  
            # else:
            #     scheduler.step()

            train_loss += loss.item()
            train_reg_loss += regression_loss.item()
            train_mmd_loss += mmd_loss.item()
            train_flow_loss += loss_final.item()

            # expression_predictions[expression_predictions < 0] = 0
            all_train_predictions.append(expression_predictions.detach())
            all_train_targets.append(protein_expression_ori.detach())

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
            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Training Flow Loss: {avg_train_flow_loss:.4f}, "
                  f"Training Regression Loss: {avg_train_reg_loss:.4f}, Training MMD Loss: {avg_train_mmd_loss:.4f}, "
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
            with torch.no_grad():  # Disable gradient calculation during evaluation
                test_loss = 0.0
                all_test_predictions = []
                all_test_targets = []
                all_indices = []
                for image, input_gene_ids, expressions, src_key_padding_mask, input_protein_ids, protein_expressions, src_key_padding_mask_protein, protein_expression_ori, protein_emb, enrichment, indices in test_loader:
                    with accelerator.autocast():
                        pred_prot_emb, expression_predictions = model(image, input_gene_ids, expressions, src_key_padding_mask, protein_emb, enrichment)  # Forward pass
                        # protein_expression_ori = protein_expression_ori.to(expression_predictions.device)
                        # print(protein_exp.device)
                        loss, regression_loss, mmd_loss = criterion(expression_predictions, protein_expression_ori)  # Loss计算
                    test_loss += loss.item()
                    # expression_predictions[expression_predictions < 0] = 0
                    all_test_predictions.append(expression_predictions.detach())
                    all_test_targets.append(protein_expression_ori.detach())
                    all_indices.append(indices.detach())

                all_test_predictions = torch.cat(all_test_predictions, dim=0)
                all_test_targets = torch.cat(all_test_targets, dim=0)
                all_indices = torch.cat(all_indices, dim=0)
                # Gather metrics across devices if using distributed training
                all_test_predictions = accelerator.gather_for_metrics(all_test_predictions)
                all_test_targets = accelerator.gather_for_metrics(all_test_targets)
                all_indices = accelerator.gather_for_metrics(all_indices)
                accelerator.print(all_test_predictions.shape)
                test_correlations = calculate_correlations(all_test_predictions, all_test_targets)
                test_pearson_corrs, test_spearman_corrs = zip(*test_correlations)
                avg_test_pearson = np.mean(test_pearson_corrs)
                avg_test_spearman = np.mean(test_spearman_corrs)
                max_test_pearson = np.max(test_pearson_corrs)
                min_test_pearson = np.min(test_pearson_corrs)
                max_test_spearman = np.max(test_spearman_corrs)
                min_test_spearman = np.min(test_spearman_corrs)
                results.append({
                    'epoch': epoch + 1,
                    'avg_pearson_corr': avg_test_pearson,
                    'max_pearson_corr': max_test_pearson,
                    'min_pearson_corr': min_test_pearson,
                    'avg_spearman_corr': avg_test_spearman,
                    'max_spearman_corr': max_test_spearman,
                    'min_spearman_corr': min_test_spearman
                })
            if accelerator.is_main_process:
                for result in results:
                    print(f"Epoch {result['epoch']} | "
                        f"Avg Pearson: {result['avg_pearson_corr']:.4f}, Max Pearson: {result['max_pearson_corr']:.4f}, "
                        f"Min Pearson: {result['min_pearson_corr']:.4f}, Avg Spearman: {result['avg_spearman_corr']:.4f}, "
                        f"Max Spearman: {result['max_spearman_corr']:.4f}, Min Spearman: {result['min_spearman_corr']:.4f}")
                if epoch + 1 == num_epochs:
                    all_test_predictions = pd.DataFrame(all_test_predictions.cpu().numpy(), columns=train_protein)
                    all_test_targets = pd.DataFrame(all_test_targets.cpu().numpy(), columns=train_protein)
                    all_test_predictions.to_csv(f'result/test_prediction_80_lr_scale.csv')
                    all_test_targets.to_csv(f'result/test_target_80_lr_scale.csv')
                    torch.save(all_indices, 'result/all_indices_scale.pkl')


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


if __name__ == '__main__':

    batch_size = 1500

    pos = pd.read_parquet('/macroverse-nas/pjz/proteomics/rcc_xenium_lr/pos_filtered.parquet')
    train_barcodes = pos[pos['split'] == 'train'].index.tolist()
    test_barcodes  = pos[pos['split'] == 'test'].index.tolist()

    train_dataset = SpatialDataset(
        barcode_tsv = train_barcodes,
        image_path='/macroverse-nas/pjz/proteomics/rcc_xenium/Xenium_V1_Human_Kidney_FFPE_normalized_he_image.ome.tif',
        spatial_pos_path='/macroverse-nas/pjz/proteomics/rcc_xenium_lr/pos_filtered.parquet',
        count_mtx_path='/macroverse-nas/pjz/proteomics/rcc_xenium_lr/rna_exp_filtered_train.parquet',
        protein_exp_path='/macroverse-nas/pjz/proteomics/rcc_xenium_lr/prot_exp_scale_filtered_train.parquet',
        enrichment_path='/macroverse-nas/pjz/proteomics/rcc_xenium_lr/train_enrichment_filtered_train.parquet',
        scgpt_model_path='/macroverse-nas/pjz/proteomics/data_model/scgpt_model',
        protein_emb_path='/macroverse-nas/pjz/proteomics/train_brca/pickle/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2_new.pt'
    )
    test_dataset = SpatialDataset(
        barcode_tsv = test_barcodes,
        image_path='/macroverse-nas/pjz/proteomics/rcc_xenium/Xenium_V1_Human_Kidney_FFPE_normalized_he_image.ome.tif',
        spatial_pos_path='/macroverse-nas/pjz/proteomics/rcc_xenium_lr/pos_filtered.parquet',
        count_mtx_path='/macroverse-nas/pjz/proteomics/rcc_xenium_lr/rna_exp_filtered_test.parquet',
        protein_exp_path='/macroverse-nas/pjz/proteomics/rcc_xenium_lr/prot_exp_scale_filtered_test.parquet',
        enrichment_path='/macroverse-nas/pjz/proteomics/rcc_xenium_lr/train_enrichment_filtered_test.parquet',
        scgpt_model_path='/macroverse-nas/pjz/proteomics/data_model/scgpt_model',
        protein_emb_path='/macroverse-nas/pjz/proteomics/train_brca/pickle/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2_new.pt'
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)

    total_test_samples = len(test_loader.dataset)
    train_protein = train_dataset.get_protein_names()

    # load pretrained weights: hist-rna
    scgpt_model_path = "/macroverse-nas/pjz/proteomics/data_model/scgpt_model"
    coach_checkpoint_path = "/macroverse-nas/pjz/proteomics/data_model/coach_model/pytorch_model.bin"
    reference_embedding_path = "/macroverse-nas/pjz/crc_codex/pretrain/data/reference_embedding.csv"

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

    # load pretrained weights: rna-protein
    flow_model_prot = ConditionalFlowNet(512, 512, 40)
    # flow_model_prot = load_pretrained_model(model = flow_model_prot, 
    #                                  pretrained_weights_path = '/root/code/crc_codex/predict_exp/flow_model.pt', device='cpu')

    for p in path_model_visual.parameters():
        p.requires_grad = False

    for p in flow_model_prot.parameters():
        p.requires_grad = True

    rna_model = build_scgpt_model()
    protein_model = build_scgpt_model()
    for p in rna_model.parameters(): p.requires_grad = False
    for p in protein_model.parameters(): p.requires_grad = False
    
    model = ImageToProteinModel(path_model_visual, rna_model, protein_model, flow_model_prot, 512, 5120, 27, train_protein)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    model, flow_model_prot, optimizer, scheduler, train_loader, test_loader = accelerator.prepare(
        model, flow_model_prot, optimizer, scheduler, train_loader, test_loader
    )

    if accelerator.is_main_process:
        accelerator.print(f"Test loader 中总共有 {total_test_samples} 个样本。")
    train_and_evaluate(model, protein_model, train_loader, test_loader, optimizer, scheduler, train_protein, num_epochs=60)

    # # if accelerator.is_main_process:  
    # #     unwrapped_model = accelerator.unwrap_model(model)
    # #     save_path = "/macroverse-nas/pjz/crc_codex/codex/codex/model/image_to_protein_final.pt"
    # #     torch.save(unwrapped_model.state_dict(), save_path)
    # #     print(f"Model weights saved to {save_path}")