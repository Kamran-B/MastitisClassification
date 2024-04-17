import random
import numpy as np
import tensorflow as tf
from keras import Sequential, Input, regularizers
from keras.layers import LSTM, Dense, Dropout
from keras.src.layers import Bidirectional
from sklearn.model_selection import train_test_split

from to_array import bit_reader

tf.keras.backend.clear_session()


# tf.config.set_visible_devices([], 'GPU')

def duplicate_and_insert(original_list, target_list, original_target_labels, target_labels, label_value, num_duplicates,
                         seed=42):
    random.seed(seed)
    for d in range(len(original_list)):
        if original_target_labels[d] == label_value:
            for j in range(num_duplicates):
                random_position = random.randint(0, len(target_list))
                target_list.insert(random_position, original_list[d].copy())
                target_labels.insert(random_position, label_value)


def preprocess_data_with_pca(X, y, n_components=1400, test_size=0.2, random_state=42):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    '''
    # Apply PCA for dimensionality reduction without standardization
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_test_pca, y_train, y_test'''
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

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = preprocess_data_with_pca(X, y, 1400, test_size=0.2, random_state=42)

# Duplicate and insert rows in the training set
X_train_augmented = X_train.copy()
y_train_augmented = y_train.copy()

# Duplicate and insert rows in the test set
X_test_augmented = X_test.copy()
y_test_augmented = y_test.copy()

# Duplicate and insert rows with different random states
duplicate_and_insert(X_train, X_train_augmented, y_train, y_train_augmented, 1, 16, seed=42)
duplicate_and_insert(X_test, X_test_augmented, y_test, y_test_augmented, 1, 16, seed=42)

y_train = np.array(y_train_augmented)
y_test = np.array(y_test_augmented)
X_train = np.array(X_train_augmented)
X_test = np.array(X_test_augmented)

# Deletes from memory to free up RAM space
del X, y, X_train_augmented, y_train_augmented, X_test_augmented, y_test_augmented
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))


model = Sequential([
    Input(shape=(1, 624300)),
    Bidirectional(LSTM(128, return_sequences=True)),  # Bidirectional LSTM
    Dropout(0.3),
    Bidirectional(LSTM(64, return_sequences=True)),  # Bidirectional LSTM
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train model with class weights
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# Predictions on test set
predictions = model.predict(X_test)

# Convert predictions to binary values (0 or 1) based on threshold 0.5
binary_predictions = (predictions > 0.5).astype(int)

# Print actual and predicted values
for actual, predicted in zip(y_test, binary_predictions):
    print(f"Actual: {actual}, Predicted: {predicted}")
