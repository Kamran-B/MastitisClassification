import random
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
#import matplotlib.pyplot as plt
from to_array import bit_reader
import keras
from keras import layers


def split_data(X, y, test_size=0.2, random_state=None):
    """
    Split the dataset into training and testing sets.

    Parameters:
        X (list or numpy.ndarray): Input features.
        y (list or numpy.ndarray): Target labels.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int or None): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing the training and testing sets for features and labels.
    """
    # Convert lists to numpy arrays if not already
    X = np.array(X)
    y = np.array(y)

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Shuffle indices
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    # Calculate split index
    split_index = int((1 - test_size) * len(X))

    # Split the data
    X_train, X_test = X[indices[:split_index]], X[indices[split_index:]]
    y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]

    return X_train, X_test, y_train, y_test

def read_numbers_from_file(file_path):
    numbers = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Convert each line to a number and append to the list
                numbers.append(int(line.strip()))
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return numbers


X = bit_reader("output_hd_exclude_binary.txt")
y = read_numbers_from_file('mast_lact1_sorted.txt')

print("splitting up data")
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("done splitting up data")

# Deletes from memory to free up RAM space
del X, y

model = keras.Sequential([
    layers.Dense(100000, activation="relu"),
    layers.Dense(1000, activation="relu"),
    layers.Dense(512, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(2, activation="softmax")
])

print("Beginning training")
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=7, batch_size=512)
results = model.evaluate(X_test, y_test)
print(results)