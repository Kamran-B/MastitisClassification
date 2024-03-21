import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from to_array import bit_reader
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier


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


X = bit_reader("output_hd_exclude_binary_herd.txt")
y = read_numbers_from_file('mast_lact1_sorted_herd.txt')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Deletes from memory to free up RAM space
del X, y

'''Iterates through the training data and adds rows for oversampling'''


# Function to duplicate rows and insert at random positions
def duplicate_and_insert(original_list, target_list, original_target_labels, target_labels, label_value, num_duplicates,
                         seed=None):
    random.seed(seed)
    for d in range(len(original_list)):
        if original_target_labels[d] == label_value:
            for j in range(num_duplicates):
                random_position = random.randint(0, len(target_list))
                target_list.insert(random_position, original_list[d].copy())
                target_labels.insert(random_position, label_value)


# This ensures that between runs the random placements of the duplicate data stays the same
seed_value = 42

# Duplicate and insert rows in the training set
X_train_augmented = X_train.copy()
y_train_augmented = y_train.copy()
duplicate_and_insert(X_train, X_train_augmented, y_train, y_train_augmented, 1, 16, seed=seed_value)

# Duplicate and insert rows in the test set
X_test_augmented = X_test.copy()
y_test_augmented = y_test.copy()
duplicate_and_insert(X_test, X_test_augmented, y_test, y_test_augmented, 1, 16, seed=seed_value)

# Deletes from memory to free up RAM space
del X_train, X_test, y_train, y_test

# Set the parameters based on the research findings
n_trees = 30  # Ntree
mtry_fraction = 0.01  # Mtry as a fraction of the total number of predictors (0.01 seems to be best)

# Calculate the actual Mtry value based on the fraction
num_predictors = len(X_train_augmented[0])
mtry = int(np.ceil(mtry_fraction * num_predictors))
print("Number of predictors: ", num_predictors)

# Create a RandomForestClassifier with the best hyperparameters
random_forest_model = RandomForestClassifier(
    n_estimators=n_trees,
    max_features=mtry,
    random_state=42,
    min_samples_split=10,
    min_samples_leaf=5,
    max_depth=10,
    class_weight={0: 1, 1: 4000},
    oob_score=True,
    verbose=3,

    # If you have issues try changing this to 1
    n_jobs=4,
)

# Train the model
random_forest_model.fit(X_train_augmented, y_train_augmented)

# Define GBM Model
gbm_model = GradientBoostingClassifier(random_state=123, verbose=2)

# Define Meta-Model
meta_model = LogisticRegression()

# Define Stacking Classifier
stacking_classifier = StackingClassifier(
    estimators=[
        ('rf', random_forest_model),  # Random Forest model
        ('gbm', gbm_model)  # GBM model
    ],
    final_estimator=meta_model,
    stack_method='predict_proba',  # Use 'predict_proba' for probability estimates
    n_jobs=1,  # Use parallel processing
    passthrough=True  # Preserve original features in the final estimator
)

# Train Stacking Classifier
stacking_classifier.fit(X_train_augmented, y_train_augmented)

# Make predictions on the test set
y_pred_stacking = stacking_classifier.predict(X_test_augmented)

# Evaluate the stacking model
accuracy_stacking = accuracy_score(y_test_augmented, y_pred_stacking)
print(f'Stacking Model Test Accuracy: {accuracy_stacking}')

# Make predictions on the test set
y_pred = random_forest_model.predict(X_test_augmented)

# Evaluate the model
accuracy = accuracy_score(y_test_augmented, y_pred)
print(y_test_augmented)
print(y_pred)
print(f'Random Forest Test Accuracy: {accuracy}')


feature_importances = random_forest_model.feature_importances_

# Threshold for the importance of SNPs
threshold = 0.005

# Identify important SNP indices based on importance scores
important_snp_indices = [i for i, importance in enumerate(feature_importances) if importance > threshold]

# Extract the corresponding SNPs from your dataset (assuming your_snp_list is a list of lists)
important_snps = [[cow[i] for i in important_snp_indices] for cow in X_train_augmented + X_test_augmented]

# Assuming you have a list of SNP labels (e.g., ['SNP1', 'SNP2', ...])
snp_labels = ['SNP{}'.format(i + 1) for i in range(len(important_snp_indices))]

# Plotting the importance scores
plt.figure(figsize=(10, 6))
plt.bar(snp_labels, [feature_importances[i] for i in important_snp_indices], color='blue', alpha=0.7)
plt.title('Importance Scores of Important SNPs')
plt.xlabel('SNP Indices')
plt.ylabel('Importance Scores')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Show or save the plot
plt.show()
