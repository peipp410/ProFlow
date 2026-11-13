import warnings

warnings.filterwarnings("ignore")

import argparse

import anndata as ad
import pandas as pd
import scanpy as sc
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.metrics import silhouette_score


parser = argparse.ArgumentParser()
parser.add_argument(
    "--cancer_type", type=str, required=True, help="TCGA cancer type, e.g. UCEC"
)
parser.add_argument(
    "--sample_path",
    type=str,
    default="/home/peijiazheng/agent/server/skills/bulk/data/sample_rna.csv",
    help="Sample metadata CSV; sample IDs must be in the first column and include a name column",
)
parser.add_argument(
    "--exp_path",
    type=str,
    default="/home/peijiazheng/agent/server/skills/bulk/results/bulk_protein_prediction.csv",
    help="Predicted sample-by-protein expression CSV",
)
args = parser.parse_args()

cancer_type = args.cancer_type
print(f"Here is the survival analysis report for the TCGA-{cancer_type} cancer type:")

sample = pd.read_csv(args.sample_path, index_col=0)
exp = pd.read_csv(args.exp_path, index_col=0)

# Select the requested cancer's tumor samples using expression data and metadata.
exp_tumor = exp[(exp.index.str[13] != "1") & (exp.index.str[15] != "B")]
sample_tumor = sample.loc[exp_tumor.index]
group = sample_tumor.loc[sample_tumor["name"] == cancer_type]
exp_cancer = exp_tumor.loc[exp_tumor.index.isin(group.index)]

if len(exp_cancer) <= 50:
    raise ValueError(
        "Need more than 50 expression samples for 50-component PCA; "
        f"found {len(exp_cancer)}"
    )

# Cluster every available expression sample first. Survival availability must
# not affect the PCA, neighbor graph, Leiden labels, or protein marker analysis.
adata = ad.AnnData(
    exp_cancer.values,
    obs=pd.DataFrame(index=exp_cancer.index),
    var=pd.DataFrame(index=exp_cancer.columns),
)
sc.tl.pca(adata, svd_solver="arpack", n_comps=50)
sc.pp.neighbors(adata, n_neighbors=20, n_pcs=30)
sc.tl.umap(adata)

resolutions = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
scores = {}

for res in resolutions:
    sc.tl.leiden(adata, resolution=res, key_added=f"leiden_{res}")
    labels = adata.obs[f"leiden_{res}"]
    score = silhouette_score(adata.obsm["X_pca"], labels)
    scores[res] = score
    print(f"resolution={res}, silhouette_score={score:.4f}")

best_res = max(scores, key=scores.get)
print(f"Best resolution: {best_res} (silhouette_score={scores[best_res]:.4f})")
sc.tl.leiden(adata, resolution=best_res, key_added="leiden")
print(f"Clustered expression samples before survival matching: {adata.n_obs}")

# Match clinical follow-up only after cluster assignments are fixed.
surv = pd.read_table(
    f"https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-{cancer_type}.survival.tsv.gz",
    index_col=0,
)
surv = surv.loc[surv.index.isin(adata.obs.index), ["OS.time", "OS"]].copy()
surv["OS.time"] = pd.to_numeric(surv["OS.time"], errors="coerce")
surv["OS"] = pd.to_numeric(surv["OS"], errors="coerce")
surv = surv.dropna(subset=["OS.time", "OS"])
surv = surv.loc[(surv["OS.time"] >= 0) & surv["OS"].isin([0, 1])]
surv["cluster"] = adata.obs["leiden"].reindex(surv.index)
surv = surv.dropna(subset=["cluster"])
print(f"Samples with valid matched survival data: {len(surv)}")

clusters_to_test = list(surv["cluster"].unique())
pvals = {}
kmf1 = KaplanMeierFitter()
kmf2 = KaplanMeierFitter()

if surv.empty:
    print("No valid matched survival records; skipping log-rank analysis.")
elif len(clusters_to_test) < 2:
    print("At least two clusters with valid survival data are required for log-rank tests.")
else:
    for cluster in clusters_to_test:
        mask_cluster = surv["cluster"] == cluster
        mask_other = surv["cluster"] != cluster

        t1, e1 = surv.loc[mask_cluster, "OS.time"], surv.loc[mask_cluster, "OS"]
        t2, e2 = surv.loc[mask_other, "OS.time"], surv.loc[mask_other, "OS"]

        results = logrank_test(t1, t2, event_observed_A=e1, event_observed_B=e2)
        pval = results.p_value

        kmf1.fit(t1, event_observed=e1)
        kmf2.fit(t2, event_observed=e2)
        median1 = kmf1.median_survival_time_
        median2 = kmf2.median_survival_time_

        if median1 > median2:
            trend = "better"
        elif median1 < median2:
            trend = "worse"
        else:
            trend = "similar"

        pvals[cluster] = (pval, median1, median2, trend)

sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon")
result = adata.uns["rank_genes_groups"]
clusters = result["names"].dtype.names

df_list = []
for cluster in clusters:
    genes = result["names"][cluster]
    pvals_adj = result["pvals_adj"][cluster]
    tmp = pd.DataFrame(
        {"cluster": cluster, "gene": genes, "pvals_adj": pvals_adj}
    )
    df_list.append(tmp[tmp["pvals_adj"] < 0.05])

df_all = pd.concat(df_list, axis=0)
significant_survival_clusters = []

for cluster, (p, median1, median2, trend) in pvals.items():
    cluster_n = int((surv["cluster"] == cluster).sum())
    print(f"Cluster {cluster}: n = {cluster_n}, log-rank p = {p:.4g}, survival trend is {trend}")
    if p < 0.05:
        significant_survival_clusters.append(cluster)
        sub = df_all[df_all["cluster"] == cluster]
        if sub.empty:
            print(f"Cluster {cluster} has no significant protein markers")
        else:
            top = sub.sort_values("pvals_adj", ascending=True).head(20)
            print(
                f"Differentially expressed proteins in Cluster {cluster}: "
                f"{','.join(top['gene'].tolist())}"
            )

if not pvals:
    print("No survival comparisons were performed.")
elif not significant_survival_clusters:
    print("No clusters had nominally significant survival differences (p < 0.05).")

if not significant_survival_clusters:
    for cluster in clusters:
        sub = df_all[df_all["cluster"] == cluster].sort_values("pvals_adj").head(10)
        if sub.empty:
            print(
                f"Exploratory differential proteins in Cluster {cluster}: "
                "none at adjusted p < 0.05"
            )
        else:
            print(
                f"Exploratory differential proteins in Cluster {cluster}: "
                f"{','.join(sub['gene'].tolist())}"
            )
