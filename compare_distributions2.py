import numpy as np
from scipy.stats import spearmanr

def abs_diffs(distr_a, distr_b):
    """Compute element-wise absolute differences and their mean.

    Args:
        distr_a (array-like): First array of values
        distr_b (array-like): Second array of values

    Returns:
        all_abs_diffs (ndarray): Element-wise absolute differences
        mean_diffs (float): Mean of the absolute differences
    """
    all_abs_diffs = np.abs(distr_a - distr_b)
    mean_diffs = np.mean(all_abs_diffs)
    return all_abs_diffs, mean_diffs


def correlation(distr_a, distr_b):
    """Compute Spearman correlation between two distributions.

    Args:
        distr_a (array-like): First array of values
        distr_b (array-like): Second array of values

    Returns:
        corr (float): Spearman correlation coefficient, or NaN if not computable.
    """
    if np.std(distr_a) == 0 or np.std(distr_b) == 0 or len(distr_a) == 0 or len(distr_b) == 0:
        return np.nan  # Handle cases where correlation cannot be computed

    # Calculate Spearman's rank correlation
    corr, _ = spearmanr(distr_a, distr_b)

    return corr #np.corrcoef(distr_a, distr_b)[0, 1]


def compute_local_ssim_score(local1, local2, C1=1e-4, C2=9e-4):

    # Mean (luminance)
    mu1 = np.mean(local1)
    mu2 = np.mean(local2)

    # Variance (constrast)
    sigma1_sq = np.var(local1)
    sigma2_sq = np.var(local2)

    # Covariance (structure)
    sigma12 = np.cov(local1.flatten(), local2.flatten())[0, 1]

    # Luminance term (mean comparison)
    luminance = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)

    # Contrast term (variance comparison)
    contrast = (2 * np.sqrt(sigma1_sq) * np.sqrt(sigma2_sq) + C2) / (sigma1_sq + sigma2_sq + C2)

    # Structure term (correlation comparison)
    structure = (sigma12 + C2 / 2) / (np.sqrt(sigma1_sq) * np.sqrt(sigma2_sq) + C2 / 2)

    # Final SSIM-like score: combine the three terms
    local_ssim_score = luminance * contrast * structure

    return local_ssim_score


def ssim_like(distr_a, distr_b, grid_near, xs, ys, zs):

    n_cells = len(xs)
    all_ssim_like, cells_per_score = np.empty(n_cells), np.empty(n_cells)

    for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
        idxs_near = grid_near[(x, y, z)]

        local_distr_a = distr_a[idxs_near]
        local_distr_b = distr_b[idxs_near]

        ssim_score = compute_local_ssim_score(local_distr_a, local_distr_b)
        all_ssim_like[i] = ssim_score
        cells_per_score[i] = len(idxs_near)

    mean_ssim_like = np.average(all_ssim_like, weights=cells_per_score)  # Larger neighbourhoods -> higher weight
    return all_ssim_like, mean_ssim_like