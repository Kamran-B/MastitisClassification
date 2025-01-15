import csv
import itertools
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

def run_grid_search(X_train_augmented, y_train_augmented, X_test, y_test, output_file="rf_results.txt"):
    # Define a grid of hyperparameters
    param_grid = {
        "n_estimators": [20, 25],
        "min_samples_split": [10],
        "min_samples_leaf": [3],
        "max_depth": [15],
        "random_state": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }

    mtry_fraction = 0.005  # Mtry as a fraction of the total number of predictors (0.005 seems to be best)

    # Calculate the actual Mtry value based on the fraction
    num_predictors = len(X_train_augmented[0])
    mtry = int(np.ceil(mtry_fraction * num_predictors))

    # Create all combinations of hyperparameters
    param_combinations = list(itertools.product(
        param_grid["n_estimators"],
        param_grid["min_samples_split"],
        param_grid["min_samples_leaf"],
        param_grid["max_depth"],
        param_grid["random_state"],
    ))

    best_avg_f1_score = 0
    best_params = None
    best_model = None

    # Iterate over each combination of parameters with a progress bar
    for params in tqdm(param_combinations, desc="Training models"):
        n_estimators, min_samples_split, min_samples_leaf, max_depth, random_state = params

        # Create a RandomForestClassifier with the current hyperparameters
        random_forest_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=mtry,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            random_state=random_state,
            class_weight={0: 1, 1: 4000},
            oob_score=True,
            n_jobs=-1,
            #criterion='entropy',  # Experiment with this
            #max_samples=0.9,  # Optional to try different sample sizes
            #min_impurity_decrease=0.001,  # To avoid overfitting
            #max_leaf_nodes=100,  # Limit number of leaf nodes
            #warm_start=True,  # Optionally keep adding trees
            #avg f1 scorebootstrap=True  # Use bootstrap sampling (default)
        )

        # Train the model
        random_forest_model.fit(X_train_augmented, y_train_augmented)

        # Make predictions on the test set
        y_pred = random_forest_model.predict(X_test)

        # Evaluate the F1 score for each class (majority and minority)
        f1_majority = f1_score(y_test, y_pred, pos_label=0)  # Majority class (0)
        f1_minority = f1_score(y_test, y_pred, pos_label=1)  # Minority class (1)

        # Average F1 score between the majority and minority class
        avg_f1_score = (f1_majority + f1_minority) / 2

        # Update the best model if necessary
        if avg_f1_score > best_avg_f1_score:
            best_avg_f1_score = avg_f1_score
            best_params = params
            best_model = random_forest_model

    # Generate a classification report for the best model
    y_pred_best = best_model.predict(X_test)
    report = classification_report(
        y_test,
        y_pred_best,
        target_names=["No mastitis (Control)", "Mastitis Present (Case)"],
    )

    # Get feature importance scores from the best model
    feature_importances = best_model.feature_importances_
    feature_importance_indices = np.argsort(feature_importances)[::-1]  # Sort in descending order

    # Write classification report to rf_report.txt
    with open("rf_report.txt", "w") as report_file:
        report_file.write(f"Best Average F1 Score: {best_avg_f1_score:.4f}\n")
        report_file.write(f"Best Parameters: {best_params}\n\n")
        report_file.write("Classification Report:\n")
        report_file.write(report + "\n")

    # Write feature importances to ranked_snps_rf.csv
    with open("ranked_snps_rf.csv", mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Feature", "Importance"])

        for idx in feature_importance_indices:
            writer.writerow([f"Feature {idx}", f"{feature_importances[idx]:.6f}"])

    print("Results written to rf_report.txt and ranked_snps_rf.csv")
