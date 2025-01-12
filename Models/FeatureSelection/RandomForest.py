import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from DataQuality.funtional_consequences import *
from DataQuality.to_array import bit_reader


def main(seed_value=42, printStats=True, savePerf=False):
    top_4000_snps_binary = 'Data/output_hd_exclude_binary_herd.txt'
    phenotypes = 'Data/Phenotypes/phenotypes_sorted_herd.txt'

    X = bit_reader(top_4000_snps_binary)
    y = load_1d_array_from_file(phenotypes)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_augmented = X_train.copy()
    y_train_augmented = y_train.copy()
    duplicate_and_insert(
        X_train, X_train_augmented, y_train, y_train_augmented, 1, 16, seed=seed_value
    )

    # Augment testing data
    X_test_augmented = X_test.copy()
    y_test_augmented = y_test.copy()
    duplicate_and_insert(
        X_test, X_test_augmented, y_test, y_test_augmented, 1, 16, seed=seed_value
    )
    X_train_augmented = np.array(X_train_augmented, dtype=float)  # Ensure dtype is float
    y_train_augmented = np.array(y_train_augmented, dtype=int)
    X_test_augmented = np.array(X_test_augmented, dtype=float)
    y_test_augmented = np.array(y_test_augmented, dtype=int)

    # Set the parameters based on the research findings
    n_trees = 10  # Ntree
    mtry_fraction = 0.005  # Mtry as a fraction of the total number of predictors (0.005 seems to be best)

    # Calculate the actual Mtry value based on the fraction
    num_predictors = len(X_train_augmented[0])
    mtry = int(np.ceil(mtry_fraction * num_predictors))
    print("Number of predictors: ", num_predictors)

    # Create a RandomForestClassifier with the best hyperparameters
    random_forest_model = RandomForestClassifier(
        n_estimators=30,
        max_features=mtry,
        random_state=42,
        min_samples_split=12,
        min_samples_leaf=5,
        max_depth=10,
        class_weight={0: 1, 1: 4000},
        oob_score=True,
        verbose=2,
        n_jobs=-1,
    )

    # Train the model
    random_forest_model.fit(X_train_augmented, y_train_augmented)

    # Make predictions on the test set
    y_pred = random_forest_model.predict(X_test_augmented)

    # Evaluate the model
    accuracy = accuracy_score(y_test_augmented, y_pred)
    print(f"Test Accuracy: {accuracy}")

    # Create and print a classification report
    report = classification_report(
        y_test_augmented,
        y_pred,
        target_names=["No mastitis (Control)", "Mastitis Present (Case)"],
    )
    print(report)

    feature_importance = random_forest_model.feature_importances_

    # Threshold for the importance of SNPs
    threshold = 0.00000001

    # Identify important SNP indices based on importance scores
    important_snp_indices = [
        i for i, importance in enumerate(feature_importance) if importance > threshold
    ]
    print("Indices of important SNPs identified by RF: " + str(important_snp_indices))
    print(
        "Length of indices of important SNPs identified by RF: "
        + str(len(important_snp_indices))
    )

if __name__ == "__main__":
    main()
