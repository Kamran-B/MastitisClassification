import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import KMeansSMOTE
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix

from DataQuality.to_array import bit_reader
from Models.FeatureSelection.helper import read_numbers_from_file
from Models.FeatureSelection.rf_grid_search import run_grid_search


# Read data
X = bit_reader("Data/output_hd_exclude_binary_herd.txt")
y = read_numbers_from_file("Data/Phenotypes/phenotypes_sorted.txt")

# Convert X to a NumPy array if it's not already
X = np.array(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Deletes from memory to free up RAM space
del X, y


def sample_features_for_smote(X, num_features):
    """
    Sample a subset of features for SMOTE.

    Parameters:
    - X: Original dataset with all features.
    - num_features: Number of features to sample for SMOTE.

    Returns:
    - X_sampled: Dataset with sampled features.
    - selected_indices: Indices of selected features.
    """
    selected_indices = np.random.choice(X.shape[1], size=num_features, replace=False)
    return X[:, selected_indices], selected_indices


def parallel_kmeans_smote(X, y, num_chunks=10, kmeans_args=None):
    """
    Apply KMeansSMOTE in parallel to handle large datasets.

    Parameters:
    - X: Feature matrix.
    - y: Target labels.
    - num_chunks: Desired number of chunks to divide the dataset into.
    - kmeans_args: Additional arguments for KMeansSMOTE.

    Returns:
    - X_resampled: Resampled feature array.
    - y_resampled: Resampled label array.
    """
    kmeans_args = kmeans_args or {}

    # Calculate chunk size based on the number of chunks
    chunk_size = max(1, X.shape[0] // num_chunks)

    def process_chunk(start_idx):
        end_idx = min(start_idx + chunk_size, X.shape[0])

        # Check if the chunk has more than one class
        unique_classes = np.unique(y[start_idx:end_idx])
        if len(unique_classes) < 2:
            print(f"Skipping chunk {start_idx}-{end_idx} due to single class: {unique_classes}")
            return None  # Skip this chunk

        smote = KMeansSMOTE(random_state=42, **kmeans_args)  # Ensure only valid arguments
        return smote.fit_resample(X[start_idx:end_idx], y[start_idx:end_idx])

    results = Parallel(n_jobs=-1)(
        delayed(process_chunk)(i) for i in range(0, X.shape[0], chunk_size)
    )

    # Filter out any None results due to skipped chunks
    results = [result for result in results if result is not None]

    # Ensure that results are not empty before stacking
    if results:
        X_resampled, y_resampled = zip(*results)
        return np.vstack(X_resampled), np.hstack(y_resampled)
    else:
        raise ValueError("No valid chunks found for resampling.")


def print_class_distribution(y, message="Class distribution"):
    """
    Print the number of examples in each class.

    Parameters:
    - y: Array of class labels.
    - message: Optional message to display.
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    print(f"{message}:")
    for cls, count in zip(unique_classes, counts):
        print(f"Class {cls}: {count} examples")


# Print the original class distribution
print_class_distribution(y_train, message="Original class distribution")

# Use 10,000 features for SMOTE, but apply SMOTE on this subset
X_train_sampled, selected_features = sample_features_for_smote(X_train, num_features=10000)

# Convert to sparse format to save memory
X_train_sparse = csr_matrix(X_train_sampled)

# Apply SMOTE with KMeans in parallel, adjust for high-dimensional data
X_train_resampled, y_train_resampled = parallel_kmeans_smote(
    X_train_sparse, y_train, num_chunks=10
)

# Print the resampled class distribution
print_class_distribution(y_train_resampled, message="Resampled class distribution")

# Deletes from memory to free up RAM space
del X_train, y_train

# Run the grid search with resampled data
run_grid_search(X_train_resampled, y_train_resampled, X_test, y_test)
