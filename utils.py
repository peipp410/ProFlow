import torch
from scipy.stats import pearsonr, spearmanr
import numpy as np

def euclidean_barycentric_projection(T, Y):
    row_sums = T.sum(dim=1, keepdim=True)
    row_sums = torch.clamp(row_sums, min=1e-9)
    return torch.mm(T, Y) / row_sums


def cosine_barycentric_projection(T, Y):
    weighted_sum = torch.mm(T, Y)
    norms = torch.norm(weighted_sum, dim=1, keepdim=True)
    norms = torch.clamp(norms, min=1e-9)
    return weighted_sum / norms


def barycentric_projection(T, Y, metric='cosine'):
    if metric == 'cosine':
        return cosine_barycentric_projection(T, Y)
    else:  # euclidean
        return euclidean_barycentric_projection(T, Y)
    

def calculate_correlations(predictions, targets):
    """Calculates Pearson and Spearman correlations for each gene.
    
    Ignores positions where targets == 0.
    Returns 0 if after filtering, data is insufficient or constant.
    """
    num_genes = predictions.shape[1]  # Assuming second axis = genes
    correlations = []
    
    for j in range(num_genes):
        gene_predictions = predictions[:, j].cpu().numpy()
        gene_targets = targets[:, j].cpu().numpy()

        mask = gene_targets != 0
        gene_predictions = gene_predictions[mask]
        gene_targets = gene_targets[mask]
        
        if len(gene_predictions) < 2 or np.all(gene_predictions == gene_predictions[0]) or np.all(gene_targets == gene_targets[0]):
            correlations.append((0, 0))
        else:
            pearson_corr, _ = pearsonr(gene_predictions, gene_targets)
            spearman_corr, _ = spearmanr(gene_predictions, gene_targets)
            correlations.append((pearson_corr, spearman_corr))
    
    return correlations