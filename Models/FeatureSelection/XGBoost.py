import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from DataQuality.funtional_consequences import *
from DataQuality.to_array import bit_reader


def main(seed_value=42, printStats=True, savePerf=False):
    top_4000_snps_binary = '../../Data/TopSNPs/top_4000_SNPs_binary.txt'
    #top_4000_snps_binary = '../../Data/output_hd_exclude_binary_herd.txt'
    phenotypes = '../../Data/Phenotypes/phenotypes_sorted_herd_func_conseq.txt'

    # Load the SNPs data and phenotypes
    X = bit_reader(top_4000_snps_binary)
    y = load_1d_array_from_file(phenotypes)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed_value
    )

    # Augment training data
    X_train_augmented = X_train.copy()
    y_train_augmented = y_train.copy()
    duplicate_and_insert(
        X_train, X_train_augmented, y_train, y_train_augmented, 1, 16, seed=seed_value
    )

    # Augment testing data
    X_test_augmented = X_test.copy()
    y_test_augmented = y_test.copy()
    '''duplicate_and_insert(
        X_test, X_test_augmented, y_test, y_test_augmented, 1, 16, seed=seed_value
    )'''

    # Convert to NumPy arrays
    X_train_augmented = np.array(X_train_augmented, dtype=float)
    y_train_augmented = np.array(y_train_augmented, dtype=int)
    X_test_augmented = np.array(X_test_augmented, dtype=float)
    y_test_augmented = np.array(y_test_augmented, dtype=int)

    # Set XGBoost parameters
    params = {
        'objective': 'binary:logistic',  # Binary classification
        'max_depth': 12,                 # Maximum tree depth
        'eta': 0.05,                      # Learning rate
        'subsample': 0.8,                # Fraction of samples used per tree
        'colsample_bytree': 0.005,       # Fraction of features used per tree
        'scale_pos_weight': 4000,        # Balancing the dataset for imbalanced classes
        'eval_metric': 'logloss',        # Evaluation metric
        'seed': seed_value,
    }

    # Convert data to DMatrix format for XGBoost
    dtrain = xgb.DMatrix(X_train_augmented, label=y_train_augmented)
    dtest = xgb.DMatrix(X_test_augmented, label=y_test_augmented)

    # Train the XGBoost model
    num_boost_round = 2000
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtest, 'Test')],
        early_stopping_rounds=30,
        verbose_eval=True
    )

    # Make predictions on the test set
    y_pred = model.predict(dtest)
    y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]

    # Evaluate the model
    accuracy = accuracy_score(y_test_augmented, y_pred_binary)
    print(f"Test Accuracy: {accuracy}")

    # Create and print a classification report
    report = classification_report(
        y_test_augmented,
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
