import numpy as np
from sklearn.feature_selection import chi2
from scipy.stats import rankdata
from multiprocessing import Pool  # Importing multiprocessing


# Move the variance calculation and chi-squared calculation functions outside the main function
def compute_variance_for_feature(X, feature_idx):
    return np.var(X[:, feature_idx])


def compute_chi2_for_feature(X, y, feature_idx):
    chi2_score, _ = chi2(X[:, feature_idx].reshape(-1, 1), y)  # Reshape to 2D for chi2
    return chi2_score[0]  # Chi-squared value for the feature


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

    # Ensure X is a NumPy array
    X = np.array(X)
    y = np.array(y)

    # Step 1: Calculate variance in parallel for each feature
    print("Calculating variances in parallel...")
    with Pool(num_workers) as pool:
        variances = pool.starmap(compute_variance_for_feature, [(X, feature_idx) for feature_idx in range(X.shape[1])])

    # Step 2: Calculate Chi-squared scores in parallel for each feature
    print("Calculating Chi-squared scores in parallel...")
    with Pool(num_workers) as pool:
        chi2_scores = pool.starmap(compute_chi2_for_feature, [(X, y, feature_idx) for feature_idx in range(X.shape[1])])

    # Step 3: Rank variances and Chi-squared scores
    print("Ranking features based on variance and Chi-squared scores...")
    variance_rankings = rankdata(-np.array(variances), method="min")  # Negative for descending order
    chi2_rankings = rankdata(-np.array(chi2_scores), method="min")  # Negative for descending order

    # Step 4: Combine rankings
    combined_scores = variance_rankings + chi2_rankings

    # Step 5: Sort features by combined ranking
    combined_rankings = np.argsort(combined_scores)  # Indices of sorted features (ascending order)

    # Step 6: Write top N features to a file
    with open(output_file, "w") as f:
        f.write("Feature\tCombined_Score\n")
        for i in range(min(top_n, len(combined_rankings))):
            feature_idx = combined_rankings[i]
            f.write(f"{feature_idx}\t{combined_scores[feature_idx]}\n")

    print(f"Top {top_n} features written to {output_file}")