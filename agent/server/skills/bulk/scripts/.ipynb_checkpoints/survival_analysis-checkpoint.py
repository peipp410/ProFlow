import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--cancer_type', type=str, required=True, help='TCGA cancer type, e.g. UCEC')
args = parser.parse_args()

cancer_type = args.cancer_type

print(f"以下是TCGA-{cancer_type}癌种的生存分析报告：")


sample = pd.read_csv('/home/peijiazheng/agent/server/skills/bulk/data/test_all_samples.csv', index_col=0)
exp = pd.read_csv('/home/peijiazheng/agent/server/skills/bulk/results/bulk_protein_prediction.csv', index_col=0)
exp_tumor = exp[exp.index.str[13] != '1']
sample_tumor = sample.loc[exp_tumor.index]

group = sample.loc[sample['name'] == cancer_type]
exp_cancer = exp.loc[exp.index.isin(group.index)]
adata = ad.AnnData(exp_cancer.values, obs=pd.DataFrame(index=exp_cancer.index), var=pd.DataFrame(index=exp_cancer.columns))
sc.tl.pca(adata, svd_solver='arpack', n_comps=50)
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=50)
sc.tl.umap(adata)

resolutions = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
scores = {}

for res in resolutions:
    sc.tl.leiden(adata, resolution=res, key_added=f'leiden_{res}')
    labels = adata.obs[f'leiden_{res}']
    score = silhouette_score(adata.obsm['X_pca'], labels)
    scores[res] = score
    # print(f"resolution={res}, silhouette_score={score:.4f}")

best_res = max(scores, key=scores.get)
print(f"Best resolution: {best_res} (silhouette_score={scores[best_res]:.4f})")

sc.tl.leiden(adata, resolution=best_res, key_added='leiden')

surv = pd.read_table(f'https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-{cancer_type}.survival.tsv.gz', index_col=0)
surv = surv.loc[surv.index.isin(adata.obs.index)]
adata = adata[adata.obs.index.isin(surv.index)]
surv = surv.loc[adata.obs.index]
surv['cluster'] = adata.obs['leiden']

clusters_to_test = [c for c in surv["cluster"].unique()]

pvals = {}

kmf1 = KaplanMeierFitter()
kmf2 = KaplanMeierFitter()

for cluster in clusters_to_test:
    mask_cluster = surv["cluster"] == cluster
    mask_other = (surv["cluster"].isin(clusters_to_test)) & (surv["cluster"] != cluster)

    t1, e1 = surv.loc[mask_cluster, "OS.time"], surv.loc[mask_cluster, "OS"]
    t2, e2 = surv.loc[mask_other, "OS.time"], surv.loc[mask_other, "OS"]

    # log-rank
    results = logrank_test(t1, t2, event_observed_A=e1, event_observed_B=e2)
    pval = results.p_value

    kmf1.fit(t1, event_observed=e1)
    kmf2.fit(t2, event_observed=e2)

    median1 = kmf1.median_survival_time_
    median2 = kmf2.median_survival_time_

    if median1 > median2:
        trend = "更好"
    elif median1 < median2:
        trend = "更差"
    else:
        trend = "相似"

    pvals[cluster] = (pval, median1, median2, trend)

sc.tl.rank_genes_groups(
    adata, 
    groupby='leiden', 
    method='wilcoxon'
)

result = adata.uns['rank_genes_groups']
clusters = result['names'].dtype.names  

df_list = []
for clust in clusters:
    genes = result['names'][clust]
    pvals_adj = result['pvals_adj'][clust]
    
    tmp = pd.DataFrame({
        'cluster': clust,
        'gene': genes,
        'pvals_adj': pvals_adj
    })
    tmp = tmp[tmp['pvals_adj'] < 0.05]   
    df_list.append(tmp)

df_all = pd.concat(df_list, axis=0)

for cluster, (p, m1, m2, trend) in pvals.items():
    if p < 0.05:
        print(f"Cluster {cluster}: p = {p:.4g}, 生存趋势{trend}")
        sub = df_all[df_all['cluster'] == cluster]
    
        if sub.empty:
            print(f"Cluster {cluster}无显著蛋白marker")
        else:
            sub = sub.sort_values('pvals_adj', ascending=True)
            top = sub.head(20)
            
            genes = top['gene'].tolist()
            print(f"Cluster {cluster}的差异蛋白: {','.join(genes)}")


