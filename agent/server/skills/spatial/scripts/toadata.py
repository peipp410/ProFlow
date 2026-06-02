import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
import glob
import os
from sklearn.cluster import DBSCAN

file_paths = glob.glob('result/*.csv')

result_ids = []

for path in file_paths:
    filename = os.path.basename(path)
    clean_name = filename.replace('_predictions.csv', '')
    df = pd.read_csv(path)

    data = df.iloc[:, 0:2].values
    db = DBSCAN(eps=1500, min_samples=100).fit(data)
    df['cluster'] = db.labels_
    unique_labels = [lbl for lbl in set(db.labels_) if lbl != -1]

    if len(unique_labels) < 2:
        print("只检测到一个主要区域，无需过滤。")
        final_df = df[df['cluster'] != -1].drop(columns=['cluster'])
    else:
        def get_shape_fingerprint(cluster_data, bins=50):
            """
            将坐标数据转换为归一化的二维直方图（即形状的‘指纹’图片）
            """
            x = cluster_data[:, 0]
            y = cluster_data[:, 1]
            
            # 1. 中心化：将中心移动到 (0,0)
            x_centered = x - np.mean(x)
            y_centered = y - np.mean(y)
            
            # 2. 生成二维直方图 (相当于生成一张低分辨率的黑白图片)
            # range根据整体数据的最大跨度来定，保证尺度一致
            heatmap, _, _ = np.histogram2d(x_centered, y_centered, bins=bins)
            
            # 3. 展平为一维向量，用于计算相关性
            return heatmap.flatten()
    
        # 取出最大的两个区域进行对比 (通常主要就是这两个)
        # 按点数排序
        sorted_labels = df[df['cluster']!=-1]['cluster'].value_counts().index.tolist()
        label_a = sorted_labels[0]
        label_b = sorted_labels[1]
    
        coords_a = df[df['cluster'] == label_a].iloc[:, 0:2].values
        coords_b = df[df['cluster'] == label_b].iloc[:, 0:2].values
    
        # 获取形状指纹
        vec_a = get_shape_fingerprint(coords_a)
        vec_b = get_shape_fingerprint(coords_b)
    
        # 计算皮尔逊相关系数 (Correlation Coefficient)
        # 结果在 -1 到 1 之间。1 表示形状完全一样。
        similarity = np.corrcoef(vec_a, vec_b)[0, 1]
        
        print(f"区域 {label_a} 和 区域 {label_b} 的形状相似度: {similarity:.4f}")
    
        # ==========================================
        # 4. 决策逻辑
        # ==========================================
        # 设定阈值，比如 0.75 (75%相似)
        # 你可以根据实际情况调整这个阈值。如果是重复切片，通常 > 0.85
        THRESHOLD = 0.5 
    
        if similarity > THRESHOLD:
            print(">>> 判定为：重复切片 (形状相似)。只保留最大的区域。")
            # 只保留最大的那个簇 (label_a)
            final_df = df[df['cluster'] == label_a].copy()
        else:
            print(">>> 判定为：不同组织 (形状不同)。全部保留。")
            # 保留所有非噪点区域
            final_df = df[df['cluster'] != -1].copy()
    
        # 清理临时列
        final_df = final_df.drop(columns=['cluster'])

    expression_data = final_df.iloc[:, 2:]
    adata = ad.AnnData(expression_data)
    spatial_coords = final_df.iloc[:, 0:2].values
    adata.obsm['spatial'] = spatial_coords
    adata.write(f'/macroverse-nas/pjz/proteomics/tcga_coad_adata/{clean_name}.h5ad')