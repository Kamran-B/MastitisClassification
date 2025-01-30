import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from DataQuality.to_array import bit_reader
from DataQuality.funtional_consequences import load_1d_array_from_file, duplicate_and_insert


def process_chunk(X_chunk, y_chunk, params, num_boost_round=500):
    """ Train XGBoost on a chunk of data and return the model and feature importances. """
    dtrain = xgb.DMatrix(X_chunk, label=y_chunk)
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round, verbose_eval=1)
    feature_importance = model.get_score(importance_type='weight')
    return model, feature_importance


def main(chunk_size=300, num_chunks=None, seed_value=42):
    top_4000_snps_binary = '../../Data/output_hd_exclude_binary_herd.txt'
    phenotypes = '../../Data/Phenotypes/phenotypes_sorted.txt'

    # Load the full dataset
    X = bit_reader(top_4000_snps_binary)
    y = load_1d_array_from_file(phenotypes)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed_value)

    del X, y

    X_train_augmented = X_train.copy()
    y_train_augmented = y_train.copy()
    duplicate_and_insert(
        X_train, X_train_augmented, y_train, y_train_augmented, 1, 16, seed=seed_value
    )
    X_train = np.array(X_train_augmented, dtype=np.int8)  # Ensure dtype is float
    y_train = np.array(y_train_augmented, dtype=np.int8)
    del X_train_augmented, y_train_augmented

    # Augment testing data
    X_test_augmented = X_test.copy()
    y_test_augmented = y_test.copy()
    duplicate_and_insert(
        X_test, X_test_augmented, y_test, y_test_augmented, 1, 16, seed=seed_value
    )

    X_test = np.array(X_test_augmented, dtype=np.int8)
    y_test = np.array(y_test_augmented, dtype=np.int8)
    del X_test_augmented, y_test_augmented

    num_samples = len(y_train)
    num_features = X_train.shape[1]
    feature_importance_aggregated = np.zeros(num_features)

    # Set XGBoost parameters
    params = {
        'objective': 'binary:logistic',  # Binary classification
        'max_depth': 8,  # Maximum tree depth
        'eta': 0.05,  # Learning rate
        'subsample': 0.5,  # Fraction of samples used per tree
        'colsample_bytree': 0.005,  # Fraction of features used per tree
        #'scale_pos_weight': 4000,  # Balancing the dataset for imbalanced classes
        'eval_metric': 'logloss',  # Evaluation metric
        'seed': seed_value,
        'tree_method': 'hist',
    }

    # Determine the number of chunks if not provided
    if num_chunks is None:
        num_chunks = int(np.ceil(num_samples / chunk_size))

    print(f"Processing training data in {num_chunks} chunks...")

    # Process each chunk for training
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, num_samples)

        X_chunk = X_train[start_idx:end_idx]
        y_chunk = y_train[start_idx:end_idx]

        # Train model on the current chunk and get feature importances
        model, feature_importance = process_chunk(X_chunk, y_chunk, params)

        # Aggregate feature importances
        for feature, importance in feature_importance.items():
            feature_index = int(feature[1:])  # Convert 'f0', 'f1'... to integers
            feature_importance_aggregated[feature_index] += importance

        print(f"Chunk {i + 1}/{num_chunks} processed.")

    # Normalize the aggregated feature importances
    feature_importance_normalized = feature_importance_aggregated / num_chunks

    # Identify important SNP indices based on a threshold
    threshold = 0.00000001
    important_snp_indices = [
        i for i, importance in enumerate(feature_importance_normalized) if importance > threshold
    ]

    print("Indices of important SNPs identified by XGBoost: " + str(important_snp_indices))
    print("Length of indices of important SNPs identified by XGBoost: " + str(len(important_snp_indices)))

    # Predict on the test set and generate the classification report
    dtest = xgb.DMatrix(X_test)
    y_pred = (model.predict(dtest) > 0.5).astype(int)

    print("\nClassification Report on Test Set:")
    report = classification_report(
        y_test,
        y_pred,
        target_names=["No mastitis (Control)", "Mastitis Present (Case)"]
    )
    print(report)


if __name__ == "__main__":
    main()
