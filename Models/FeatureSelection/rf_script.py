from sklearn.model_selection import train_test_split
from DataQuality.to_array import bit_reader
from helper import read_numbers_from_file, duplicate_and_insert
from rf_grid_search import run_grid_search

X = bit_reader("Data/output_hd_exclude_binary_herd.txt")
y = read_numbers_from_file("Data/Phenotypes/phenotypes_sorted.txt")


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Deletes from memory to free up RAM space
del X, y

# Duplicate and insert rows in the training set
X_train_augmented = X_train.copy()
y_train_augmented = y_train.copy()
duplicate_and_insert(
    X_train, X_train_augmented, y_train, y_train_augmented, 1, 16, seed=42
)

# Deletes from memory to free up RAM space
del X_train, y_train

run_grid_search(X_train_augmented, y_train_augmented, X_test, y_test)

