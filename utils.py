import torch
from scipy.stats import pearsonr, spearmanr
import numpy as np


def calculate_correlations(predictions, targets, feature_names=None, missing_mask=None):
    """Calculate correlations while retaining observed zero values.

    ``missing_mask`` marks values that were missing in the original data and
    later filled with zero for model input. These entries are excluded, while
    observed target values equal to zero remain in the correlation analysis.
    Non-finite prediction/target pairs are also excluded. Undefined
    correlations are reported as NaN rather than 0.
    """
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    if torch.is_tensor(missing_mask):
        missing_mask = missing_mask.detach().cpu().numpy()

    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    if predictions.shape != targets.shape:
        raise ValueError(
            f"predictions and targets must have the same shape, got "
            f"{predictions.shape} and {targets.shape}"
        )

    if missing_mask is None:
        missing_mask = np.zeros_like(targets, dtype=bool)
    else:
        missing_mask = np.asarray(missing_mask, dtype=bool)
        if missing_mask.shape != targets.shape:
            raise ValueError(
                f"missing_mask must have the same shape as targets, got "
                f"{missing_mask.shape} and {targets.shape}"
            )

    num_features = predictions.shape[1]
    if feature_names is None:
        feature_names = [f"feature_{j}" for j in range(num_features)]
    elif len(feature_names) != num_features:
        raise ValueError(
            f"feature_names has length {len(feature_names)}, "
            f"but predictions has {num_features} features"
        )

    correlations = []
    for j, feature_name in enumerate(feature_names):
        feature_predictions = np.asarray(predictions[:, j], dtype=float)
        feature_targets = np.asarray(targets[:, j], dtype=float)

        valid = (
            ~missing_mask[:, j]
            & np.isfinite(feature_predictions)
            & np.isfinite(feature_targets)
        )
        feature_predictions = feature_predictions[valid]
        feature_targets = feature_targets[valid]

        n_valid = len(feature_targets)
        if (
            n_valid < 3
            or np.all(feature_predictions == feature_predictions[0])
            or np.all(feature_targets == feature_targets[0])
        ):
            pearson_corr = pearson_p = np.nan
            spearman_corr = spearman_p = np.nan
        else:
            pearson_corr, pearson_p = pearsonr(feature_predictions, feature_targets)
            spearman_corr, spearman_p = spearmanr(feature_predictions, feature_targets)

        correlations.append({
            "protein": feature_name,
            "pearson_r": pearson_corr,
            "pearson_p": pearson_p,
            "spearman_rho": spearman_corr,
            "spearman_p": spearman_p,
        })

    return correlations


def summarize_correlations(correlation_records):
    """Summarize the distribution of per-feature correlations."""
    summary = {}
    for metric in ("pearson_r", "spearman_rho"):
        values = np.asarray(
            [record[metric] for record in correlation_records], dtype=float
        )
        values = values[np.isfinite(values)]
        if values.size == 0:
            summary[metric] = {
                "mean": np.nan,
                "max": np.nan,
                "min": np.nan,
                "median": np.nan,
                "q1": np.nan,
                "q3": np.nan,
                "iqr": np.nan,
                "n_defined": 0,
            }
            continue

        q1, q3 = np.percentile(values, [25, 75])
        summary[metric] = {
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
            "min": float(np.min(values)),
            "median": float(np.median(values)),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(q3 - q1),
            "n_defined": int(values.size),
        }

    return summary
