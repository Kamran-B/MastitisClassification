import csv

import numpy as np
from sklearn.model_selection import train_test_split
from DataQuality.to_array import bit_reader
from Models.FeatureSelection.helper import read_numbers_from_file, duplicate_and_insert
from Models.FeatureSelection.rf_grid_search import run_grid_search


# Read data
X = bit_reader("Data/output_hd_exclude_binary_herd.txt")
y = read_numbers_from_file("Data/Phenotypes/phenotypes_sorted.txt")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Deletes from memory to free up RAM space
del X, y


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

# Initialize a dictionary to accumulate feature importances
combined_feature_importance_dict = {}

# Run the grid search with different random seeds and combine results
seeds = [101, 202, 303, 404, 42]
for seed in seeds:
    result_dict = run_grid_search(X_train_augmented, y_train_augmented, X_test, y_test, seed)

    # Add the feature importances to the combined dictionary
    for feature, importance in result_dict.items():
        if feature in combined_feature_importance_dict:
            combined_feature_importance_dict[feature] += importance
        else:
            combined_feature_importance_dict[feature] = importance

# Sort the dictionary by importance values in descending order
sorted_feature_importance_dict = dict(sorted(combined_feature_importance_dict.items(), key=lambda item: item[1], reverse=True))

# Write the sorted dictionary to a CSV file
with open("combined_feature_importances.csv", mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Feature", "Combined Importance"])

    for feature, importance in sorted_feature_importance_dict.items():
        writer.writerow([f"Feature {feature}", f"{importance:.6f}"])

print("Results written to combined_feature_importances.csv")
