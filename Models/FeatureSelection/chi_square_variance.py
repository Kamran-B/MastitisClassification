import numpy as np
from sklearn.feature_selection import chi2
from scipy.stats import rankdata
from multiprocessing import Pool


def compute_variance_for_feature_batch(args):
    """Compute variance for a batch of features."""
    X, idxs = args
    return np.var(X[:, idxs], axis=0)


def compute_chi2_for_feature_batch(args):
    """Compute chi-squared for a batch of features."""
    X, y, idxs = args
    return chi2(X[:, idxs], y)[0]


def calculate_feature_importance(X, y, output_file="top_features.txt", top_n=10000, num_workers=4):
    """
    Calculate feature importance by combining Chi-squared and variance rankings.
    Parallelize the calculation of variance and Chi-squared scores.

    Args:
        X (numpy.ndarray): Feature matrix (samples x features).
        y (numpy.ndarray): Target vector (samples,).
        output_file (str): Path to the file where top features will be written.
        top_n (int): Number of top features to write to the file.
        num_workers (int): Number of parallel workers to use.

    Returns:
        None
    """
    # Ensure X and y are NumPy arrays
    X = np.array(X)
    y = np.array(y)

    num_features = X.shape[1]
    feature_indices = np.array_split(range(num_features), num_workers)

    # Step 1: Calculate variances in parallel
    print("Calculating variances in parallel...")
    with Pool(num_workers) as pool:
        variances_list = pool.map(compute_variance_for_feature_batch, [(X, idxs) for idxs in feature_indices])
    variances = np.concatenate(variances_list)

    # Step 2: Calculate Chi-squared scores in parallel
    print("Calculating Chi-squared scores in parallel...")
    with Pool(num_workers) as pool:
        chi2_scores_list = pool.map(compute_chi2_for_feature_batch, [(X, y, idxs) for idxs in feature_indices])
    chi2_scores = np.concatenate(chi2_scores_list)

    # Step 3: Rank variances and Chi-squared scores
    print("Ranking features based on variance and Chi-squared scores...")
    variance_rankings = rankdata(-variances, method="min")
    chi2_rankings = rankdata(-chi2_scores, method="min")

    # Step 4: Combine rankings
    combined_scores = variance_rankings + chi2_rankings

    # Step 5: Sort features by combined ranking
    combined_rankings = np.argsort(combined_scores)

    # Step 6: Write top N features to a file
    with open(output_file, "w") as f:
        f.write("Feature\tCombined_Score\n")
        for i in range(min(top_n, len(combined_rankings))):
            feature_idx = combined_rankings[i]
            f.write(f"{feature_idx}\t{combined_scores[feature_idx]}\n")

    print(f"Top {top_n} features written to {output_file}")
