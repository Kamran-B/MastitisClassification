import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from DataQuality.funtional_consequences import *
from DataQuality.to_array import bit_reader

from sklearn.datasets import dump_svmlight_file
import numpy as np
import os

def convert_to_libsvm(X, y, output_dir, chunk_size=10000):
    """
    Converts the dataset to LibSVM format and splits it into chunks for external memory use.

    Parameters:
    - X: np.ndarray, SNP data (samples x features)
    - y: np.ndarray, labels (binary classification)
    - output_dir: str, directory to save the chunks
    - chunk_size: int, number of rows per chunk

    Returns:
    - paths: list of str, paths to the generated LibSVM files
    """
    os.makedirs(output_dir, exist_ok=True)
    num_samples = X.shape[0]
    paths = []

    # Split data into chunks and save each chunk as a separate LibSVM file
    for i in range(0, num_samples, chunk_size):
        chunk_X = X[i:i + chunk_size]
        chunk_y = y[i:i + chunk_size]
        chunk_path = os.path.join(output_dir, f"chunk_{i // chunk_size}.libsvm")
        dump_svmlight_file(chunk_X, chunk_y, chunk_path)
        paths.append(chunk_path)

    print(f"Converted dataset into {len(paths)} chunks.")
    return paths


def main(seed_value=42, printStats=True, savePerf=False):
    '''top_4000_snps_binary = '../../Data/output_hd_exclude_binary_herd.txt'

    # Load the SNPs data and phenotypes
    X = bit_reader(top_4000_snps_binary)'''
    phenotypes = '../../Data/Phenotypes/phenotypes_sorted_herd.txt'
    y = load_1d_array_from_file(phenotypes)
    output_dir = './XGBoostFiles'

    # Set XGBoost parameters
    params = {
        'objective': 'binary:logistic',  # Binary classification
        'max_depth': 12,                 # Maximum tree depth
        'eta': 0.05,                     # Learning rate
        'subsample': 0.8,                # Fraction of samples used per tree
        'colsample_bytree': 0.005,       # Fraction of features used per tree
        'scale_pos_weight': 4000,        # Balancing the dataset for imbalanced classes
        'eval_metric': 'logloss',        # Evaluation metric
        'seed': seed_value,
    }

    # Load the dataset using external memory
    dtrain = xgb.DMatrix(f"{output_dir}?format=csv#cache")

    # Train the XGBoost model
    num_boost_round = 2000
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, 'Train')],
        early_stopping_rounds=30,
        verbose_eval=True
    )

    # Since external memory is used, predictions can be done directly using the model
    y_pred = model.predict(dtrain)
    y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]

    # Evaluate the model
    accuracy = accuracy_score(y, y_pred_binary)
    print(f"Train Accuracy: {accuracy}")

    # Create and print a classification report
    report = classification_report(
        y,
        y_pred_binary,
        target_names=["No mastitis (Control)", "Mastitis Present (Case)"],
    )
    print(report)

    # Extract feature importance
    feature_importance = model.get_score(importance_type='weight')

    # Identify important SNP indices based on importance scores
    threshold = 0.00000001
    important_snp_indices = [
        int(i[1:]) for i, importance in feature_importance.items() if importance > threshold
    ]
    print("Indices of important SNPs identified by XGBoost: " + str(important_snp_indices))
    print(
        "Length of indices of important SNPs identified by XGBoost: "
        + str(len(important_snp_indices))
    )



if __name__ == "__main__":
    main()
