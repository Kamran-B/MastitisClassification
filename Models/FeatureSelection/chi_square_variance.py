import numpy as np
from sklearn.feature_selection import chi2
from scipy.stats import rankdata
import os
import psutil

def log_memory_usage(stage):
    """Log current memory usage for debugging."""
    process = psutil.Process(os.getpid())
    print(f"[{stage}] Memory usage: {process.memory_info().rss / (1024 ** 2):.2f} MB")

def calculate_feature_importance(X, y, output_file="top_features.txt", top_n=10000):
    """
    Calculate feature importance by combining Chi-squared and variance rankings.
    Process features one by one for maximum memory efficiency.

    Args:
        X (numpy.ndarray): Feature matrix (samples x features).
        y (numpy.ndarray): Target vector (samples,).
        output_file (str): Path to the file where top features will be written.
        top_n (int): Number of top features to write to the file.

    Returns:
        None
    """
    log_memory_usage("Start")
    # Ensure X and y are NumPy arrays
    X = np.array(X)
    y = np.array(y)

    num_features = X.shape[1]
    variances = []
    chi2_scores = []

    print("Calculating feature importance one by one...")
    for feature_idx in range(num_features):
        # Calculate variance for the current feature
        variance = np.var(X[:, feature_idx])
        variances.append(variance)

        # Calculate Chi-squared score for the current feature
        chi2_score, _ = chi2(X[:, feature_idx].reshape(-1, 1), y)
        chi2_scores.append(chi2_score[0])

        # Log progress
        if feature_idx % 1000 == 0 or feature_idx == num_features - 1:
            print(f"Processed {feature_idx + 1}/{num_features} features...")
            log_memory_usage(f"After feature {feature_idx + 1}")

    # Step 3: Rank variances and Chi-squared scores
    print("Ranking features based on variance and Chi-squared scores...")
    variance_rankings = rankdata(-np.array(variances), method="min")
    chi2_rankings = rankdata(-np.array(chi2_scores), method="min")

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
            if i % 100 == 0:  # Log progress every 100 features
                print(f"Written top {i}/{top_n} features so far...")
    print(f"Top {top_n} features written to {output_file}")
    log_memory_usage("End")
