import numpy as np
from sklearn.feature_selection import chi2
import csv


def benjamini_hochberg(p_values):
    """
    Apply Benjamini-Hochberg FDR correction to p-values.

    Args:
        p_values (numpy.ndarray): Array of p-values.

    Returns:
        numpy.ndarray: Adjusted p-values.
    """
    p_values_sorted = np.sort(p_values)
    m = len(p_values)  # Total number of hypotheses
    adjusted_p_values = np.zeros(m)

    for i in range(m):
        adjusted_p_values[i] = p_values_sorted[i] * m / (i + 1)

    # Map back to the original order
    rank_order = np.argsort(p_values)
    return adjusted_p_values[np.argsort(rank_order)]


def calculate_feature_importance(X, y, output_file="ranked_snps_chi_2_full.csv", batch_size=5000):
    """
    Calculate feature importance using Chi-squared scores and apply FDR control (Benjamini-Hochberg).
    Sort features by adjusted p-value (lowest FDR). Writes all features to a CSV.

    Args:
        X (numpy.ndarray): Feature matrix (samples x features).
        y (numpy.ndarray): Target vector (samples,).
        output_file (str): Path to the file where features will be written.
        batch_size (int): Number of features to process in each batch.

    Returns:
        None
    """
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

    # Step 3: Apply Benjamini-Hochberg procedure to control FDR
    print("Applying Benjamini-Hochberg FDR correction...")
    adjusted_p_values = benjamini_hochberg(chi2_p_values)

    # Step 4: Sort features by adjusted p-value (FDR)
    print("Sorting features by adjusted p-value (FDR)...")
    sorted_indices = np.argsort(adjusted_p_values)  # Sort by adjusted p-value (ascending)

    # Step 5: Write all results to CSV
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Feature", "Adjusted_P_Value"])
        for i in range(len(sorted_indices)):
            feature_idx = sorted_indices[i]
            writer.writerow([feature_idx, adjusted_p_values[feature_idx]])
            if i % 10000 == 0:  # Log progress every 10000 features
                print(f"Written feature {i + 1}/{len(sorted_indices)}")

    print(f"All features written to {output_file}")
