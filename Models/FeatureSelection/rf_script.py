import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC

from DataQuality.to_array import bit_reader
from Models.FeatureSelection.helper import read_numbers_from_file, duplicate_and_insert
from Models.FeatureSelection.rf_grid_search import run_grid_search


# Read data
X = bit_reader("Data/output_hd_exclude_binary_herd.txt")
y = read_numbers_from_file("Data/Phenotypes/phenotypes_sorted.txt")

# Convert X to a NumPy array if it's not already
X = np.array(X)

# Define categorical feature indices (all features in your case)
categorical_features = list(range(X.shape[1]))  # Ensures it's a list of indices

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


def smote_resampling(X, y, categorical_features):
    """
    Apply SMOTENC to handle imbalanced datasets with categorical features.

    Parameters:
    - X: Feature matrix.
    - y: Target labels.
    - categorical_features: Indices of categorical features.

    Returns:
    - X_resampled: Resampled feature array.
    - y_resampled: Resampled label array.
    """
    smote = SMOTENC(categorical_features=categorical_features, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


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

X_train_augmented = X_train.copy()
y_train_augmented = y_train.copy()
duplicate_and_insert(
        X_train, X_train_augmented, y_train, y_train_augmented, 1, 16, seed=42
    )

# Print the resampled class distribution
print_class_distribution(y_train_augmented, message="Resampled class distribution")

# Deletes from memory to free up RAM space
del X_train, y_train

# Run the grid search with resampled data
run_grid_search(X_train_augmented, y_train_augmented, X_test, y_test)
