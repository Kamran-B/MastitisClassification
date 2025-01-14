import numpy as np
from sklearn.feature_selection import chi2
from scipy.stats import rankdata
from multiprocessing import Pool
import os
import psutil

def log_memory_usage(stage):
    """Log current memory usage for debugging."""
    process = psutil.Process(os.getpid())
    print(f"[{stage}] Memory usage: {process.memory_info().rss / (1024 ** 2):.2f} MB")

def compute_variance_for_feature_batch(args):
    """Compute variance for a batch of features."""
    X, idxs = args
    return np.var(X[:, idxs], axis=0)

def compute_chi2_for_feature_batch(args):
    """Compute chi-squared for a batch of features."""
    X, y, idxs = args
    return chi2(X[:, idxs], y)[0]

def calculate_feature_importance(X, y, output_file="top_features.txt", top_n=10000, num_workers=4, chunk_size=100):
    """
    Calculate feature importance by combining Chi-squared and variance rankings.
    Optimize for memory efficiency by using smaller chunks.

    Args:
        X (numpy.ndarray): Feature matrix (samples x features).
        y (numpy.ndarray): Target vector (samples,).
        output_file (str): Path to the file where top features will be written.
        top_n (int): Number of top features to write to the file.
        num_workers (int): Number of parallel workers to use.
        chunk_size (int): Number of features to process per chunk.

    Returns:
        None
    """
    log_memory_usage("Start")
    # Ensure X and y are NumPy arrays
    X = np.array(X)
    y = np.array(y)

    num_features = X.shape[1]
    feature_indices = [range(start, min(start + chunk_size, num_features))
                       for start in range(0, num_features, chunk_size)]

    # Step 1: Calculate variances in chunks
    print("Calculating variances in parallel with small chunks...")
    variances = []
    with Pool(num_workers) as pool:
        for i, variance_batch in enumerate(
                pool.imap_unordered(compute_variance_for_feature_batch, [(X, idxs) for idxs in feature_indices])):
            variances.append(variance_batch)
            print(f"Variance calculation completed for chunk {i + 1}/{len(feature_indices)}")
            log_memory_usage("Variance Calculation")
    variances = np.concatenate(variances)

    # Step 2: Calculate Chi-squared scores in chunks
    print("Calculating Chi-squared scores in parallel with small chunks...")
    chi2_scores = []
    with Pool(num_workers) as pool:
        for i, chi2_batch in enumerate(
                pool.imap_unordered(compute_chi2_for_feature_batch, [(X, y, idxs) for idxs in feature_indices])):
            chi2_scores.append(chi2_batch)
            print(f"Chi-squared calculation completed for chunk {i + 1}/{len(feature_indices)}")
            log_memory_usage("Chi-squared Calculation")
    chi2_scores = np.concatenate(chi2_scores)

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
