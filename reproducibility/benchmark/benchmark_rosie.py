import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import os
import json
from pathlib import Path
from typing import Optional
from PIL import Image
import torch.nn.functional as F
from Dataset import ProteinExpressionDataset
import os
# Import accelerate
from accelerate import Accelerator

# ==========================================
# 1. Model Definition
# ==========================================
def get_model(num_outputs: Optional[int] = None) -> nn.Module:

    model = models.convnext_small(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False
    for param in model.features[7].parameters():
        param.requires_grad = True
    for param in model.classifier[0].parameters():
        param.requires_grad = True
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_outputs)
    
    return model

# ==========================================
# 2. Correlation Calculation Helper Functions
# ==========================================
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
        # Handle zero standard deviation case to avoid errors
        if np.std(p) < 1e-6 or np.std(t) < 1e-6:
            pcc, scc = 0.0, 0.0
        else:
            pcc, _ = pearsonr(p, t)
            scc, _ = spearmanr(p, t)
        pccs.append(pcc)
        sccs.append(scc)
        
    return np.mean(pccs), np.mean(sccs), pccs, sccs

# ==========================================
# 3. Main Training Loop
# ==========================================
def train():
    # Initialize Accelerator
    accelerator = Accelerator()
    
    # Parameter settings
    BATCH_SIZE = 5120
    LR = 1e-4
    NUM_EPOCHS = 50

    # --- 1. Data Preparation ---
    if accelerator.is_main_process:
        print("Loading data...")
    train_df = pd.read_csv('/mnt/vdd/pjz/crc/ORIONCRC_dataset_tile_20x/train_ori.csv')
    val_df = pd.read_csv('/mnt/vdd/pjz/crc/ORIONCRC_dataset_tile_20x/test_ori.csv')
    train_df = train_df.drop(['PD-1'], axis=1)
    val_df = val_df.drop(['PD-1'], axis=1)
    
    col_names = ['new_filenames', 'PECAM1', 'PTPRC', 'CD68', 'CD4', 'FOXP3', 'CD8A', 
                 'PTPRCRO', 'MS4A1', 'CD274', 'CD3E', 'CD163', 'CDH1', 'MKI67', 'KRT19', 'ACTA2']
    train_df.columns = col_names
    val_df.columns = col_names
    
    # Group validation set by sample
    val_df['sample_name'] = val_df['new_filenames'].apply(lambda x: x.split('/')[1].split('__')[0])
    sample_dfs = {
        sample: group.drop(columns=['sample_name']).reset_index(drop=True) 
        for sample, group in val_df.groupby('sample_name')
    }
    
    # Instantiate training set
    train_dataset = ProteinExpressionDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
    
    # Get protein list
    protein_names = train_dataset.get_protein_names()
    num_proteins = len(protein_names)

    # Prepare validation set Dataloaders dictionary
    val_loaders_dict = {}
    for sample_name, s_df in sample_dfs.items():
        v_dataset = ProteinExpressionDataset(s_df)
        v_loader = DataLoader(v_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)
        val_loaders_dict[sample_name] = v_loader

    # --- 2. Model and Optimizer ---
    model = get_model(num_outputs=num_proteins)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # --- 3. Use Accelerator Prepare ---
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    
    for sample_name in val_loaders_dict:
        val_loaders_dict[sample_name] = accelerator.prepare(val_loaders_dict[sample_name])

    # --- 4. Training Loop ---
    for epoch in range(NUM_EPOCHS):
        # ==========================================
        # 1. Training Phase
        # ==========================================
        model.train()
        total_train_loss = 0.0
        train_pccs = []
        train_sccs = []
        all_preds = []
        all_targets = []
        
        # Iterate over training set
        for batch in train_loader:
            images, targets = batch
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            
            # --- Record metrics ---
            # To avoid slowing down training, we compute PCC/SCC on each batch and record it.
            # Note: This computes the "average PCC of the current batch", only for monitoring convergence trends.
            # detach() is important to prevent gradient graph accumulation.
            batch_pcc, batch_scc, _, _ = calculate_metrics(outputs, targets)
            
            total_train_loss += loss.item()
            train_pccs.append(batch_pcc)
            train_sccs.append(batch_scc)
            
        # --- Compute and output training set summary for this epoch ---
        # Gather metrics across all GPUs and average (for simplicity, we only print the average seen by the main process,
        # which is usually sufficient for monitoring trends. Use accelerator.gather for absolute precision.)
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_pcc = np.mean(train_pccs)
        avg_train_scc = np.mean(train_sccs)
        
        if accelerator.is_main_process:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
                  f"| Train Loss: {avg_train_loss:.4f} "
                  f"| Train PCC: {avg_train_pcc:.4f} "
                  f"| Train SCC: {avg_train_scc:.4f}")
            
        #         print("="*20 + "\n")
        if (epoch + 1) % 5 == 0:
            model.eval()

            results = []

            # Global statistics (existing)
            epoch_avg_pearsons = []
            epoch_avg_spearmans = []

            # ===== New: Collect per-protein PCC for all samples =====
            # For storing shape: (num_samples, num_proteins)
            all_samples_list_pcc = []
            all_samples_list_scc = []  # Keep this if you want Spearman later too

            if (epoch + 1) == NUM_EPOCHS and accelerator.is_main_process:
                os.makedirs('result/corr', exist_ok=True)

            with torch.no_grad():
                for sample_name, loader in val_loaders_dict.items():
                    all_preds = []
                    all_targets = []

                    for batch in loader:
                        images, targets = batch
                        outputs = model(images)

                        preds_gathered, targets_gathered = accelerator.gather_for_metrics((outputs, targets))
                        all_preds.append(preds_gathered)
                        all_targets.append(targets_gathered)

                    accelerator.wait_for_everyone()

                    if accelerator.is_main_process:
                        sample_preds = torch.cat(all_preds, dim=0)
                        sample_targets = torch.cat(all_targets, dim=0)

                        mean_pcc, mean_scc, list_pcc, list_scc = calculate_metrics(sample_preds, sample_targets)

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

                        # ===== New: Collect per-protein PCC =====
                        # list_pcc is typically an array/list of length = num_proteins
                        all_samples_list_pcc.append(np.array(list_pcc, dtype=np.float32))
                        all_samples_list_scc.append(np.array(list_scc, dtype=np.float32))

                    if (epoch + 1) == NUM_EPOCHS and accelerator.is_main_process:
                        save_dir = 'result/corr'
                        df_corr = pd.DataFrame({
                            'Pearson': list_pcc,
                            'Spearman': list_pcc
                        }, index=['PECAM1', 'PTPRC', 'CD68', 'CD4', 'FOXP3', 'CD8A', 'PTPRCRO', 'MS4A1', 'CD274', 'CD3E', 'CD163', 'CDH1', 'MKI67', 'KRT19', 'ACTA2'])

                        csv_path = os.path.join(save_dir, f"{sample_name}.csv")
                        df_corr.to_csv(csv_path)

            # --- Validation finished, output summary ---
            if accelerator.is_main_process:
                print(f"\n" + "="*30 + f" Epoch {epoch+1} Test Summary (Train Loss: {avg_train_loss:.4f}) " + "="*30)

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

                # ===== New: Per-protein average PCC + mean/median of these average PCCs =====
                all_samples_list_pcc = np.stack(all_samples_list_pcc, axis=0)  # (num_samples, num_proteins)

                # Average Pearson of each protein across all test samples
                protein_mean_pcc = np.mean(all_samples_list_pcc, axis=0)       # (num_proteins,)

                # Statistics of “protein average PCC” across the protein dimension
                protein_mean_pcc_avg = float(np.mean(protein_mean_pcc))
                protein_mean_pcc_median = float(np.median(protein_mean_pcc))

                print("\n[Per-protein PCC across all test samples]")
                print(f"Protein-wise mean PCC -> Mean over proteins: {protein_mean_pcc_avg:.4f}, Median over proteins: {protein_mean_pcc_median:.4f}")

                # If you have protein_names, print per protein
                # Note: assumes protein_names length == num_proteins
                if 'protein_names' in globals():
                    print("\nProtein\t\tMean_PCC(over samples)")
                    for i, pname in enumerate(protein_names):
                        print(f"{pname}\t\t{protein_mean_pcc[i]:.4f}")
                else:
                    # If no protein_names, just print numeric indices
                    print("\nProtein_idx\tMean_PCC(over samples)")
                    for i in range(protein_mean_pcc.shape[0]):
                        print(f"{i}\t\t{protein_mean_pcc[i]:.4f}")

                if (epoch + 1) == NUM_EPOCHS:
                    print(f"\n[Info] Detailed correlation CSVs saved to 'result/corr_rosie/'")
                print("="*20 + "\n")

if __name__ == "__main__":
    train()