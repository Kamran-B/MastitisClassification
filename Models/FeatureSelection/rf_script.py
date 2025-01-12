import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from DataQuality.to_array import bit_reader
from helper import read_numbers_from_file
from rf_grid_search import run_grid_search

# Read data
X = bit_reader("Data/output_hd_exclude_binary_herd.txt")
y = read_numbers_from_file("Data/Phenotypes/phenotypes_sorted.txt")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Deletes from memory to free up RAM space
del X, y


def incremental_smote(X, y, chunk_size=10000, random_state=42):
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = [], []

    # Process data in chunks
    for i in range(0, len(X), chunk_size):
        end = min(i + chunk_size, len(X))
        X_chunk, y_chunk = smote.fit_resample(X[i:end], y[i:end])
        X_resampled.append(X_chunk)
        y_resampled.append(y_chunk)

    # Concatenate all chunks
    return np.vstack(X_resampled), np.hstack(y_resampled)


# Apply incremental SMOTE
X_train_resampled, y_train_resampled = incremental_smote(X_train, y_train)

# Deletes from memory to free up RAM space
del X_train, y_train

# Run the grid search with resampled data
run_grid_search(X_train_resampled, y_train_resampled, X_test, y_test)
