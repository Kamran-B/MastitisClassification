import itertools
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm


def run_grid_search(X_train_augmented, y_train_augmented, X_test, y_test, output_file="smote_rf_results.txt"):
    # Define a grid of hyperparameters
    param_grid = {
        "n_estimators": [15, 30, 50],
        "max_features": ["sqrt"],
        "min_samples_split": [10, 12],
        "min_samples_leaf": [3, 5, 7],
        "max_depth": [7, 10, 12],
    }

    mtry_fraction = 0.005  # Mtry as a fraction of the total number of predictors (0.005 seems to be best)

    # Calculate the actual Mtry value based on the fraction
    num_predictors = len(X_train_augmented[0])
    mtry = int(np.ceil(mtry_fraction * num_predictors))

    # Create all combinations of hyperparameters
    param_combinations = list(itertools.product(
        param_grid["n_estimators"],
        param_grid["max_features"],
        param_grid["min_samples_split"],
        param_grid["min_samples_leaf"],
        param_grid["max_depth"],
    ))

    best_accuracy = 0
    best_params = None
    best_model = None

    # Iterate over each combination of parameters with a progress bar
    for params in tqdm(param_combinations, desc="Training models"):
        n_estimators, max_features, min_samples_split, min_samples_leaf, max_depth = params

        # Create a RandomForestClassifier with the current hyperparameters
        random_forest_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=mtry,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            random_state=42,
            class_weight={0: 1, 1: 4000},
            oob_score=True,
            n_jobs=-1,  # Use all CPUs
        )

        # Train the model
        random_forest_model.fit(X_train_augmented, y_train_augmented)

        # Make predictions on the test set
        y_pred = random_forest_model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)

        # Check if this is the best model so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
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

    # Write results to the output file
    with open(output_file, "w") as file:
        file.write(f"Best Accuracy: {best_accuracy:.4f}\n")
        file.write(f"Best Parameters: {best_params}\n\n")
        file.write("Classification Report:\n")
        file.write(report + "\n")

        file.write("Feature Importances (sorted):\n")
        for idx in feature_importance_indices:
            file.write(f"Feature {idx}: {feature_importances[idx]:.6f}\n")

    print(f"Results written to {output_file}")
