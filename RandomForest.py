from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm


print("about to load data")
# Load your dataset
file_path = './output_hd_exclude.raw'
file_path2 = './mast_lact1.phen'

# Data is space-separated
delimiter = ' '

# Read the raw file and create a 2D array
data = []
data2 = []
i = 1

with open(file_path, 'r') as file:
    # Skip the header if present
    header = file.readline()
    # Get the total number of lines in the file for progress bar
    total_lines = sum(1 for _ in file)
    # Return to the beginning of the file
    file.seek(0)
    # Create a tqdm progress bar for the loop
    for line in tqdm(file, total=total_lines, desc="Processing file 1", unit="line"):
        row = line.strip().split(delimiter)
        data.append(row)
        i += 1

with open(file_path2, 'r') as file:
    # Get the total number of lines in the file for progress bar
    total_lines = sum(1 for _ in file)
    # Return to the beginning of the file
    file.seek(0)
    # Create a tqdm progress bar for the loop
    for line in tqdm(file, total=total_lines, desc="Processing file 2", unit="line"):
        row = line.strip().split(delimiter)
        data2.append(row)
        i += 1

print("Done Loading CSV")
data = data[1:]

# Convert values to integers for both data and data2
for row in tqdm(data, desc="Converting data to integers", unit="row"):
    for i in range(2, len(row)):
        row[i] = int(row[i])
for row in tqdm(data2, desc="Converting data2 to integers", unit="row"):
    for i in range(2, len(row)):
        row[i] = int(row[i])


# Extract first column values from both arrays
col1_data = set(row[0] for row in data)
col1_data2 = set(row[0] for row in data2)

# Find common values in the first column
common_values = col1_data.intersection(col1_data2)

# Filter both arrays to keep only the common values
data_filtered = [row for row in data if row[0] in common_values]
data2_filtered = [row for row in data2 if row[0] in common_values]

# Sort both arrays based on the first column of data
data_sorted = sorted(data_filtered, key=lambda x: x[0])
data2_sorted = sorted(data2_filtered, key=lambda x: x[0])

X = [row[6:] for row in data_sorted]
y = [row[2] for row in data2_sorted]

print(X[0][:10])
print(y[:10])
print(X[1][:10])
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Ensure that X_train and X_test have at least one feature
if not X_train or not any(X_train):
    raise ValueError("X_train does not contain valid features.")
if not X_test or not any(X_test):
    raise ValueError("X_test does not contain valid features.")

print("about to train model")
# Create a random forest classifier
rf_model = RandomForestClassifier(random_state=42)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

best_params = {
    'bootstrap': False,
    'max_depth': 20,
    'max_features': 'sqrt',
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'n_estimators': 700
}

# Create a RandomForestClassifier with the best hyperparameters
best_rf_model = RandomForestClassifier(**best_params, random_state=42)

# Train the model
best_rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy}')

'''
# Plot the feature importances if needed
feature_importances = best_rf_model.feature_importances_
plt.bar(range(len(feature_importances)), feature_importances)
plt.xlabel('Feature Index')
plt.ylabel('Feature Importance')
plt.title('Random Forest Feature Importances')
plt.show()
'''