import numpy as np
from sklearn.feature_selection import chi2
import os
import psutil
import csv


def log_memory_usage(stage):
    """Log current memory usage for debugging."""
    process = psutil.Process(os.getpid())
    print(f"[{stage}] Memory usage: {process.memory_info().rss / (1024 ** 2):.2f} MB")


def calculate_feature_importance(X, y, output_file="top_features_chi_2.csv", top_n=10000, batch_size=2000):
    """
    Calculate feature importance using Chi-squared scores and related columns.
    Sort features by Chi-squared p-value (ascending order).
    Writes the top features to a CSV.

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

    print("Calculating Chi-squared scores in batches...")
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

    # Step 3: Sort features by Chi-squared p-value (ascending order)
    print("Sorting features by Chi-squared p-value (lower is more important)...")
    sorted_indices = np.argsort(chi2_p_values)  # Sort by p-value (ascending)

    # Step 4: Write sorted results to CSV
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Feature", "Chi2_P_Value"])
        for i in range(min(top_n, len(sorted_indices))):
            feature_idx = sorted_indices[i]
            writer.writerow([feature_idx, chi2_p_values[feature_idx]])
            if i % 100 == 0:  # Log progress every 100 features
                print(f"Written top {i}/{top_n} features so far...")

    print(f"Top {top_n} features written to {output_file}")
    log_memory_usage("End")
