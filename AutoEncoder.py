import random

import keras.callbacks
import numpy as np
import tensorflow as tf
from keras import Sequential, Input, regularizers, layers
from keras.src.layers import Bidirectional
from sklearn.model_selection import train_test_split

from to_array import bit_reader

tf.keras.backend.clear_session()
# tf.config.set_visible_devices([], 'GPU')

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

def removeOnes(x_train, y_train):
    ret_x = []
    ret_y = []
    pos_x = []
    pos_y = []
    for i in range(len(x_train)):
        if y_train[i] == 0:
            ret_x.append(x_train[i])
            ret_y.append(y_train[i])
        else:
            pos_x.append(x_train[i])
            pos_y.append(y_train[i])
    return np.array(ret_x), np.array(ret_y), np.array(pos_x)
print(1)
#X = bit_reader("output_hd_exclude_binary_herd.txt")
#y = read_numbers_from_file('mast_lact1_sorted_herd.txt')
#X = bit_reader("output_hd_exclude_top_SNPs_binary.txt")
X = bit_reader("output_hd_exclude_binary_herd.txt")
y = read_numbers_from_file('mast_lact1_sorted_herd.txt')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = preprocess_data_with_pca(X, y, 1400, test_size=0.2, random_state=42)
print("original X_train shape:", len(X_train), len(X_train[0]))
#print(X_train[:5][:5])
# Remove cows with mastitis from the training set
X_train, y_train, X_mast = removeOnes(X_train, y_train)
y_test = np.array(y_test)
X_test = np.array(X_test)
print(1.1)
# Deletes from memory to free up RAM space
del X, y #, X_train_augmented, y_train_augmented, X_test_augmented, y_test_augmented
#X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
print(1.2)
X_train = np.reshape(X_train, (1417, 624300, 1))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

latent_space_size = 50
print(2)
model = keras.Sequential([
    # Encoder
    layers.Input(shape=(624300, 1)),
    layers.Conv1D(
        filters=32,
        kernel_size=7,
        padding="same",
        strides=2,
        activation="relu"
    ),
    layers.MaxPool1D(pool_size=2, padding="same"),
    layers.Conv1D(
        filters=16,
        kernel_size=7,
        padding="same",
        strides=2,
        activation="relu"
    ),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(latent_space_size, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(624300 // 4, activation='linear'),
    layers.Reshape((624300 // 4, 1)),

    layers.Conv1DTranspose(
        filters=16,
        kernel_size=7,
        padding="same",
        strides=2,
        activation="relu"
    ),
    layers.Conv1DTranspose(
        filters=32,
        kernel_size=7,
        padding="same",
        strides=2,
        activation="relu"
    )
])
print(3)
# Compile model
model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
model.summary()
print(4)
print("X_train shape:", X_train.shape)
#print(X_train[:5][:5])
# Train model with class weights
model.fit(X_train,
          X_train,
          epochs=150,
          batch_size=16,
          validation_split=0.1,
          callbacks=[
              keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
          ])#, validation_data=(X_test, y_test))
print(5)
# Evaluate model
#loss, accuracy = model.evaluate(X_test, y_test)
#print("Test Accuracy:", accuracy)
print(6)
# Predictions on test set
reconstructions = model.predict(X_train)
errors = []
print(f"Healthy: {len(X_train)}")
for i in range(len(X_train)):
    mae = abs(X_train[i] - reconstructions[i]).mean()
    errors.append(mae)
print(errors)
print(np.mean(errors), min(errors), max(errors))

print(f"Mastitis: {len(X_mast)}")
reconstructions = model.predict(X_mast)
errors = []
for i in range(len(X_mast)):
    mae = abs(X_mast[i] - reconstructions[i]).mean()
    errors.append(mae)
print(errors)
print(np.mean(errors), min(errors), max(errors))

"""for stat in X_train:
    prediction = model.predict(stat)
    mse = (np.square(stat - prediction)).mean()
    recon_errors.append(mse)"""

"""predictions = model.predict(X_test)
print(7)
# Convert predictions to binary values (0 or 1) based on threshold 0.5
binary_predictions = (predictions > 0.5).astype(int)

# Print actual and predicted values
for actual, predicted in zip(y_test, binary_predictions):
    print(f"Actual: {actual}, Predicted: {predicted}")"""



"""encoder = Sequential([
    Input(shape=(1, 624300)),
    Bidirectional(LSTM(128, activation='leaky_relu', return_sequences=True)),  # Bidirectional LSTM
    Dropout(0.3),
    Bidirectional(LSTM(64, activation='leaky_relu', return_sequences=True)),  # Bidirectional LSTM
    Dropout(0.3),
    Dense(latent_space_size, activation=None)
])

decoder = Sequential([
    Dense(250, activation='leaky_relu'),
    Bidirectional(LSTM(128, activation='leaky_relu', return_sequences=True)),  # Bidirectional LSTM
    Dropout(0.3),
    Bidirectional(LSTM(64, activation='leaky_relu', return_sequences=True))  # Bidirectional LSTM
])


model = keras.Sequential([
    # Encoder
    Input((624300,)),
    Dense(512, activation='relu'),
    Dense(64, activation='relu'),
    Dense(latent_space_size, activation='relu'),

    # Decoder
    Dense(64, activation='relu'),
    Dense(512, activation='relu'),
    Dense(624300, activation='linear')
])"""