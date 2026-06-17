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
# 导入 accelerate
from accelerate import Accelerator

# ==========================================
# 1. 模型定义
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
# 2. 相关性计算辅助函数
# ==========================================
def calculate_metrics(preds, targets):
    """
    preds/targets: (N, Num_Proteins) torch tensors
    返回: (mean_pcc, mean_scc, list_pcc, list_scc)
    """
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    
    num_proteins = preds.shape[1]
    pccs = []
    sccs = []
    
    for i in range(num_proteins):
        p = preds[:, i]
        t = targets[:, i]
        # 处理标准差为0的情况，避免报错
        if np.std(p) < 1e-6 or np.std(t) < 1e-6:
            pcc, scc = 0.0, 0.0
        else:
            pcc, _ = pearsonr(p, t)
            scc, _ = spearmanr(p, t)
        pccs.append(pcc)
        sccs.append(scc)
        
    return np.mean(pccs), np.mean(sccs), pccs, sccs

# ==========================================
# 3. 训练主流程
# ==========================================
def train():
    # 初始化 Accelerator
    accelerator = Accelerator()
    
    # 参数设置
    BATCH_SIZE = 5120
    LR = 1e-4
    NUM_EPOCHS = 50

    # --- 1. 数据准备 ---
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
    
    # 验证集按样本分组
    val_df['sample_name'] = val_df['new_filenames'].apply(lambda x: x.split('/')[1].split('__')[0])
    sample_dfs = {
        sample: group.drop(columns=['sample_name']).reset_index(drop=True) 
        for sample, group in val_df.groupby('sample_name')
    }
    
    # 实例化训练集
    train_dataset = ProteinExpressionDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
    
    # 获取蛋白列表
    protein_names = train_dataset.get_protein_names()
    num_proteins = len(protein_names)

    # 准备验证集 Dataloaders 字典
    val_loaders_dict = {}
    for sample_name, s_df in sample_dfs.items():
        v_dataset = ProteinExpressionDataset(s_df)
        v_loader = DataLoader(v_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)
        val_loaders_dict[sample_name] = v_loader

    # --- 2. 模型与优化器 ---
    model = get_model(num_outputs=num_proteins)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # --- 3. 使用 Accelerator Prepare ---
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    
    for sample_name in val_loaders_dict:
        val_loaders_dict[sample_name] = accelerator.prepare(val_loaders_dict[sample_name])

    # --- 4. 训练循环 ---
    for epoch in range(NUM_EPOCHS):
        # ==========================================
        # 1. 训练阶段 (Training)
        # ==========================================
        model.train()
        total_train_loss = 0.0
        train_pccs = []
        train_sccs = []
        all_preds = []
        all_targets = []
        
        # 遍历训练集
        for batch in train_loader:
            images, targets = batch
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            
            # --- 记录指标 ---
            # 为了不拖慢训练，我们在每个 batch 上计算一次 PCC/SCC 并记录
            # 注意：这里计算的是"当前 batch 的平均 PCC"，仅用于监控收敛趋势
            # detach() 很重要，防止梯度图累积
            batch_pcc, batch_scc, _, _ = calculate_metrics(outputs, targets)
            
            total_train_loss += loss.item()
            train_pccs.append(batch_pcc)
            train_sccs.append(batch_scc)
            
        # --- 计算并输出训练集本轮汇总 ---
        # 收集所有 GPU 上的指标取平均（为了打印准确，这里简化为只打印主进程看到的平均值，
        # 通常足够观察趋势。若要绝对精确可使用 accelerator.gather）
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

            # 全局统计（现有）
            epoch_avg_pearsons = []
            epoch_avg_spearmans = []

            # ===== 新增：收集所有样本的每蛋白 PCC =====
            # 用于存储 shape: (num_samples, num_proteins)
            all_samples_list_pcc = []
            all_samples_list_scc = []  # 如你以后也想做 Spearman 可留着

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

                        # ===== 新增：把每蛋白 PCC 收集起来 =====
                        # list_pcc 一般是长度 = num_proteins 的 array/list
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

            # --- 验证结束，输出汇总 ---
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

                # ===== 新增：每个蛋白平均 PCC + 这些平均 PCC 的均值/中位数 =====
                all_samples_list_pcc = np.stack(all_samples_list_pcc, axis=0)  # (num_samples, num_proteins)

                # 每个蛋白在所有测试样本上的平均 Pearson
                protein_mean_pcc = np.mean(all_samples_list_pcc, axis=0)       # (num_proteins,)

                # 这些“蛋白的平均 PCC”在蛋白维度的统计
                protein_mean_pcc_avg = float(np.mean(protein_mean_pcc))
                protein_mean_pcc_median = float(np.median(protein_mean_pcc))

                print("\n[Per-protein PCC across all test samples]")
                print(f"Protein-wise mean PCC -> Mean over proteins: {protein_mean_pcc_avg:.4f}, Median over proteins: {protein_mean_pcc_median:.4f}")

                # 如果你有 protein_names，就逐个蛋白打印
                # 注意：这里假设 protein_names 长度 == num_proteins
                if 'protein_names' in globals():
                    print("\nProtein\t\tMean_PCC(over samples)")
                    for i, pname in enumerate(protein_names):
                        print(f"{pname}\t\t{protein_mean_pcc[i]:.4f}")
                else:
                    # 没有 protein_names 就只打印数值索引
                    print("\nProtein_idx\tMean_PCC(over samples)")
                    for i in range(protein_mean_pcc.shape[0]):
                        print(f"{i}\t\t{protein_mean_pcc[i]:.4f}")

                if (epoch + 1) == NUM_EPOCHS:
                    print(f"\n[Info] Detailed correlation CSVs saved to 'result/corr_rosie/'")
                print("="*20 + "\n")

if __name__ == "__main__":
    train()