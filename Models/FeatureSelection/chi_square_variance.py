import numpy as np
from sklearn.feature_selection import chi2
from scipy.stats import rankdata
import os
import psutil

def log_memory_usage(stage):
    """Log current memory usage for debugging."""
    process = psutil.Process(os.getpid())
    print(f"[{stage}] Memory usage: {process.memory_info().rss / (1024 ** 2):.2f} MB")

def calculate_feature_importance(X, y, output_file="top_features.txt", top_n=10000, batch_size=100):
    """
    Calculate feature importance by combining Chi-squared and variance rankings.
    Optimize by processing features in small batches for speed and memory efficiency.

    Args:
        X (numpy.ndarray): Feature matrix (samples x features).
        y (numpy.ndarray): Target vector (samples,).
        output_file (str): Path to the file where top features will be written.
        top_n (int): Number of top features to write to the file.
        batch_size (int): Number of features to process in each batch.

    Returns:
        None
    """
    log_memory_usage("Start")
    # Ensure X and y are NumPy arrays
    X = np.array(X)
    y = np.array(y)

    num_features = X.shape[1]
    variances = np.zeros(num_features)
    chi2_scores = np.zeros(num_features)

    print("Calculating feature importance in batches...")
    for start_idx in range(0, num_features, batch_size):
        end_idx = min(start_idx + batch_size, num_features)
        batch_indices = range(start_idx, end_idx)

        # Compute variance for the batch
        variances[start_idx:end_idx] = np.var(X[:, batch_indices], axis=0)

        # Compute Chi-squared for the batch
        chi2_batch, _ = chi2(X[:, batch_indices], y)
        chi2_scores[start_idx:end_idx] = chi2_batch

        # Log progress
        print(f"Processed features {start_idx + 1}-{end_idx}/{num_features}...")
        log_memory_usage(f"After processing batch {start_idx // batch_size + 1}")

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
            if i % 100 == 0:  # Log progress every 100 features
                print(f"Written top {i}/{top_n} features so far...")
    print(f"Top {top_n} features written to {output_file}")
    log_memory_usage("End")
