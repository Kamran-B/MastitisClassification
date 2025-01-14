import numpy as np
from sklearn.feature_selection import chi2
from scipy.stats import rankdata
import os
import psutil
import csv

def log_memory_usage(stage):
    """Log current memory usage for debugging."""
    process = psutil.Process(os.getpid())
    print(f"[{stage}] Memory usage: {process.memory_info().rss / (1024 ** 2):.2f} MB")

def calculate_feature_importance(X, y, output_file="top_features.csv", top_n=10000, batch_size=500):
    """
    Calculate feature importance using only Chi-squared scores and related columns.
    Includes: combined ranking, chi-squared p-value, and chi-squared ranking.
    Process features in batches for efficiency. Writes output to CSV.

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
    chi2_scores = np.zeros(num_features)
    chi2_p_values = np.zeros(num_features)

    print("Calculating feature importance using Chi-squared in batches...")
    for start_idx in range(0, num_features, batch_size):
        end_idx = min(start_idx + batch_size, num_features)
        batch_indices = range(start_idx, end_idx)

        # Compute Chi-squared for the batch
        chi2_batch, p_values = chi2(X[:, batch_indices], y)
        chi2_scores[start_idx:end_idx] = chi2_batch
        chi2_p_values[start_idx:end_idx] = p_values

        # Log progress
        print(f"Processed features {start_idx + 1}-{end_idx}/{num_features}...")
        log_memory_usage(f"After processing batch {start_idx // batch_size + 1}")

    # Step 3: Rank Chi-squared scores
    print("Ranking features based on Chi-squared scores...")
    chi2_rankings = rankdata(-chi2_scores, method="min")

    # Step 4: Combine rankings (using Chi-squared ranking only here)
    combined_scores = chi2_rankings

    # Step 5: Sort features by combined ranking
    combined_rankings = np.argsort(combined_scores)

    # Step 6: Write results to a CSV file
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Feature", "Combined_Ranking", "Chi2_P_Value", "Chi2_Ranking"])
        for i in range(min(top_n, len(combined_rankings))):
            feature_idx = combined_rankings[i]
            writer.writerow([feature_idx, combined_scores[feature_idx], chi2_p_values[feature_idx], chi2_rankings[feature_idx]])
            if i % 100 == 0:  # Log progress every 100 features
                print(f"Written top {i}/{top_n} features so far...")
    print(f"Top {top_n} features written to {output_file}")
    log_memory_usage("End")
