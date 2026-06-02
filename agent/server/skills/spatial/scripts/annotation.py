import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
import glob
import os
import random
from sklearn.preprocessing import StandardScaler


def extract_spot_features(adata, th=0.5, k_neighbors=6):
    """
    Extract clinical/spatial feature vectors from spatial transcriptome AnnData.
    Returns: dict of feature_name -> float value
    """

    feats = {}
    # 1. Convert expression matrix to DataFrame
    if isinstance(adata.X, np.ndarray):
        df = pd.DataFrame(adata.X, columns=adata.var_names)
    else:
        df = pd.DataFrame(adata.X.toarray(), columns=adata.var_names)

    frac_pos = (df > th).mean(axis=0)
    for g in df.columns:
        feats[f"frac_pos_{g}"] = float(frac_pos[g])

    df[df < 0] = 0

    coords = adata.obsm['spatial']  # (n_spots, 2)
    n_spots = df.shape[0]

    # Guard against k_neighbors > n_spots-1
    k = min(k_neighbors, max(n_spots - 1, 1))

    # 2. Cell-type scores
    tumor_score = df[['CDH1', 'KRT19']].mean(axis=1)
    immune_score = df['PTPRC']
    t_cell_score = df[['CD3E', 'CD4', 'CD8A']].mean(axis=1)
    cd8_score = df['CD8A']
    treg_score = df[['CD4', 'FOXP3']].mean(axis=1)
    macro_score = df[['CD68', 'CD163']].mean(axis=1)
    m2_score = df['CD163']
    stroma_score = df['ACTA2']
    prolif_score = df['MKI67']
    checkpoint_score = df['CD274']
    bcell_score = df['MS4A1']
    endothelial_score = df['PECAM1']
    
    all_mean = df.mean(axis=0)
    for g in df.columns:
        feats[f"mean_{g}"] = float(all_mean[g])

    # 3. Binary masks (to numpy)
    tumor_mask = (tumor_score > th).to_numpy()
    immune_mask = (immune_score > th).to_numpy()
    stroma_mask = (stroma_score > th).to_numpy()
    treg_mask = (treg_score > th).to_numpy()
    cd8_mask = (cd8_score > th).to_numpy()
    m2_mask = (m2_score > th).to_numpy()
    prolif_mask = (prolif_score > th).to_numpy()
    pdl1_mask = (checkpoint_score > th).to_numpy()
    bcell_mask = (bcell_score > th).to_numpy()

    # 4. Basic spatial distance features
    def get_min_dist(set_a, set_b):
        if (len(set_a) == 0) or (len(set_b) == 0):
            return 1000.0
        tree = cKDTree(set_b)
        dists, _ = tree.query(set_a, k=1)
        return float(np.mean(dists))

    tumor_spots = coords[tumor_mask]
    immune_spots = coords[immune_mask]
    stroma_spots = coords[stroma_mask]

    # 5. Nearest-neighbor structure
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    dists_knn, idxs_knn = nn.kneighbors(coords)  # (n_spots, k+1)
    neighbor_idx = idxs_knn[:, 1:]  # exclude self, (n_spots, k)

    # 6.1 Global / interaction features
    feats['tumor_burden'] = float(tumor_mask.mean())
    feats['immune_infiltration'] = float(immune_mask.mean())
    feats['stroma_ratio'] = float(stroma_mask.mean())
    feats['m2_polarization'] = float(m2_score.mean() / (macro_score.mean() + 1e-6))
    feats['treg_cd8_ratio'] = float(treg_score.mean() / (cd8_score.mean() + 1e-6))
    feats['hot_spot_score'] = float((tumor_score * cd8_score).mean())
    feats['fibrotic_tumor_score'] = float((tumor_score * stroma_score).mean())
    feats['exhaustion_score'] = float((cd8_score * checkpoint_score).mean())
    feats['aggressive_tumor_score'] = float((tumor_score * prolif_score).mean())

    feats['dist_tumor_immune'] = get_min_dist(tumor_spots, immune_spots)
    feats['dist_immune_stroma'] = get_min_dist(immune_spots, stroma_spots)

    # 6.2 Non-spatial supplements
    feats['t_cell_score_global'] = float(t_cell_score.mean())
    feats['b_cell_score_global'] = float(bcell_score.mean())
    feats['endothelial_score_global'] = float(endothelial_score.mean())
    feats['cd8_spatial_cv'] = float(cd8_score.std() / (cd8_score.mean() + 1e-6))
    feats['pdl1_spatial_cv'] = float(checkpoint_score.std() / (checkpoint_score.mean() + 1e-6))

    # 6.3 Neighborhood-based local spatial features
    neigh_tumor = tumor_mask[neighbor_idx]
    neigh_immune = immune_mask[neighbor_idx]
    neigh_stroma = stroma_mask[neighbor_idx]
    neigh_treg = treg_mask[neighbor_idx]
    neigh_cd8 = cd8_mask[neighbor_idx]
    neigh_m2 = m2_mask[neighbor_idx]
    neigh_prolif = prolif_mask[neighbor_idx]
    neigh_pdl1 = pdl1_mask[neighbor_idx]
    neigh_bcell = bcell_mask[neighbor_idx]

    # 6.3.1 Tumor-immune contact
    if tumor_mask.any():
        feats['tumor_immune_contact'] = float(neigh_immune[tumor_mask].mean())
    else:
        feats['tumor_immune_contact'] = 0.0

    # 6.3.2 Peritumoral M2 enrichment
    if tumor_mask.any():
        feats['peritumoral_m2_enrichment'] = float(neigh_m2[tumor_mask].mean())
    else:
        feats['peritumoral_m2_enrichment'] = 0.0

    # 6.3.3 Peritumoral Treg/CD8
    if tumor_mask.any():
        peritumoral_treg = float(neigh_treg[tumor_mask].mean())
        peritumoral_cd8 = float(neigh_cd8[tumor_mask].mean())
    else:
        peritumoral_treg = 0.0
        peritumoral_cd8 = 0.0
    feats['peritumoral_treg_ratio'] = peritumoral_treg
    feats['peritumoral_cd8_ratio'] = peritumoral_cd8
    feats['peritumoral_treg_cd8_ratio'] = float(
        peritumoral_treg / (peritumoral_cd8 + 1e-6)
    )

    # 6.3.4 CD8/Treg clustering
    if cd8_mask.any():
        feats['cd8_cluster_score'] = float(neigh_cd8[cd8_mask].mean())
    else:
        feats['cd8_cluster_score'] = 0.0

    if treg_mask.any():
        feats['treg_cluster_score'] = float(neigh_treg[treg_mask].mean())
    else:
        feats['treg_cluster_score'] = 0.0

    # 6.3.5 Encapsulating stroma
    if tumor_mask.any():
        feats['encapsulating_stroma_index'] = float(neigh_stroma[tumor_mask].mean())
    else:
        feats['encapsulating_stroma_index'] = 0.0

    # 6.3.6 Peritumoral proliferation
    if tumor_mask.any():
        feats['peritumoral_proliferation'] = float(neigh_prolif[tumor_mask].mean())
    else:
        feats['peritumoral_proliferation'] = 0.0

    # 6.3.7 Hotspot fraction
    hotspot_mask = tumor_mask & cd8_mask
    feats['hotspot_fraction'] = float(hotspot_mask.mean())

    # 6.3.8 Tumor-stroma-immune "triple interface"
    if immune_mask.any():
        imm_neigh_tumor_frac = neigh_tumor[immune_mask].mean(axis=1)
        imm_neigh_stroma_frac = neigh_stroma[immune_mask].mean(axis=1)
        triple_interface = (imm_neigh_tumor_frac > 0) & (imm_neigh_stroma_frac > 0)
        feats['triple_interface_fraction'] = float(triple_interface.mean())
    else:
        feats['triple_interface_fraction'] = 0.0

    # 6.3.9 Peritumoral CD8-PD-L1 axis
    cd8_pdl1_high = cd8_mask & pdl1_mask
    neigh_cd8_pdl1 = cd8_pdl1_high[neighbor_idx]
    if tumor_mask.any():
        feats['peritumoral_cd8_pdl1_fraction'] = float(
            neigh_cd8_pdl1[tumor_mask].mean()
        )
    else:
        feats['peritumoral_cd8_pdl1_fraction'] = 0.0

    # 6.3.10 B cell spatial features
    if bcell_mask.any():
        feats['bcell_cluster_score'] = float(neigh_bcell[bcell_mask].mean())
    else:
        feats['bcell_cluster_score'] = 0.0

    if tumor_mask.any() and bcell_mask.any():
        feats['peritumoral_bcell_enrichment'] = float(neigh_bcell[tumor_mask].mean())
    else:
        feats['peritumoral_bcell_enrichment'] = 0.0

    return feats


feature_dicts = {}
samples = []
# todo: you should process all the predicted files (using a loop)
# e.g. for sample_name in xxx
h5ad_files = glob.glob('results/*.h5ad')
for h5ad_path in h5ad_files:
    filename = os.path.basename(h5ad_path).replace('.h5ad', '')
    sample_name = '-'.join(filename.split('-')[:3])
    adata = sc.read_h5ad(h5ad_path)

    sc.tl.pca(adata, n_comps=10)
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=8)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.5)
    sc.pl.umap(adata, color=['leiden'])

    markers_tumor = ['KRT19', 'CDH1']
    markers_stroma = ['ACTA2', 'PECAM1']
    markers_immune = ['CD3E', 'MS4A1', 'CD68']

    all_markers = markers_tumor + markers_stroma + markers_immune
    df_slide = pd.DataFrame(adata.X, index=adata.obs.index, columns=adata.var.index)
    X = df_slide[all_markers].copy()

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df_slide['Cluster_ID'] = clusters
    adata.obs['kmeans_cluster'] = clusters.astype(str)

    cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_),
                                   columns=X.columns)
    cluster_mapping = {}

    for cluster_id in range(10):
        centroid = cluster_centers.iloc[cluster_id]

        centroid_z = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns).iloc[cluster_id]

        score_tumor = centroid_z[markers_tumor].mean()
        score_stroma = centroid_z[markers_stroma].mean()
        score_immune = centroid_z[markers_immune].mean()

        scores = {'Tumor': score_tumor, 'Stroma': score_stroma, 'Immune': score_immune}
        best_match = max(scores, key=scores.get)

        if max(scores.values()) < 0:
            best_match = 'Background/Unknown'

        cluster_mapping[cluster_id] = best_match

    df_slide['Tissue_Type'] = df_slide['Cluster_ID'].map(cluster_mapping)

    adata.obs['Tissue_Type'] = df_slide['Tissue_Type'].tolist()

    sc.pp.scale(adata, max_value=10)
    feats = extract_spot_features(adata)
    feature_dicts[sample_name] = feats

    annotated_path = os.path.join('results', f'{sample_name}_annotated.h5ad')
    adata.write(annotated_path)

features_df = pd.DataFrame.from_dict(feature_dicts, orient='index')

features_df.to_csv('data/features_df.csv')