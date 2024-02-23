import numpy as np
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

dataset = """
58 58 1
1001 1001 1
1032J 1032J 1
1042J 1042J 1
1074J 1074J 1
202J 202J 1
218J 218J 1
245 245 1
252J 252J 1
159251D 159251D 1
159227D 159227D 1
183321D 183321D 1
159181D 159181D 1
479 479 1
724 724 1
759J 759J 1
767J 767J 1
811J 811J 1
865 865 1
986 986 1
996J 996J 1
998J 998J 1
2005 2005 1
159256D 159256D 1
2621H 2621H 1
2740 2740 1
2845 2845 1
2853H 2853H 1
2907 2907 1
3652 3652 1
3735 3735 1
4341 4341 1
4523 4523 1
4561 4561 1
4599 4599 1
4616 4616 1
4629 4629 1
4639 4639 1
4641 4641 1
4648 4648 1
4659 4659 1
4681 4681 1
4692 4692 1
4728 4728 1
4751 4751 1
4778 4778 1
4799 4799 1
4842 4842 1
4918 4918 1
4924 4924 1
4928 4928 1
5642 5642 1
7837 7837 1
8508 8508 1
8519 8519 1
8656 8656 1
8720 8720 1
8790 8790 1
11694 11694 1
15007 15007 1
16231 16231 1
17162 17162 1
17428 17428 1
17770 17770 1
18567 18567 1
18831 18831 1
18894 18894 1
19080 19080 1
19373 19373 1
19498 19498 1
19576 19576 1
19838 19838 1
19881 19881 1
19909 19909 1
19988 19988 1
20026 20026 1
20238 20238 1
20293 20293 1
20398 20398 1
20473 20473 1
20651 20651 1
21272 21272 1
21806 21806 1
21810 21810 1
21815 21815 1
21834 21834 1
22114 22114 1
22129 22129 1
22206 22206 1
22524 22524 1
22738 22738 1
23093 23093 1
23256 23256 1
23314 23314 1
51902 51902 1
"""

# Split the data into rows
rows = dataset.strip().split('\n')

# Create a dictionary with the first item in each row as the key
id_dict = {line.split()[0]: None for line in dataset.strip().split('\n')}

# Print the resulting dictionary
print(id_dict)

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
        '''if row[0] in id_dict:
            for i in range(20):
                data.append(row)
        else:
            data.append(row)'''
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
        '''if row[0] in id_dict:
            for i in range(20):
                data2.append(row)
        else:
            data2.append(row)'''
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

print(y_test)

#Ensure that X_train and X_test have at least one feature
if not X_train or not any(X_train):
    raise ValueError("X_train does not contain valid features.")
if not X_test or not any(X_test):
    raise ValueError("X_test does not contain valid features.")

for i in range(len(X_train)):
    if y_train[i] == 1:
        for j in range(20):
            X_train.append(X_train[i])
            y_train.append(1)

for i in range(len(X_test)):
    if y_test[i] == 1:
        for j in range(20):
            X_test.append(X_test[i])
            y_test.append(1)

print(y_test)

print("about to train model")

# Set the parameters based on the research findings
n_trees = 1000  # Ntree
mtry_fraction = 0.01  # Mtry as a fraction of the total number of predictors

# Calculate the actual Mtry value based on the fraction
num_predictors = len(X_train[0])
mtry = int(np.ceil(mtry_fraction * num_predictors))

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
    'bootstrap': True,
    'max_depth': 10,
    'max_features': 'sqrt',
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'n_estimators': 300,
    'random_state': 67}
'''
    'criterion': 'gini',
    'class_weight': 'balanced',
    'oob_score': True,
}'''

# Create a RandomForestClassifier with the best hyperparameters
random_forest_model = RandomForestClassifier(
    n_estimators=n_trees,
    max_features=mtry,
    random_state=42,
    oob_score=True,
    verbose=1,
)

# Train the model
random_forest_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = random_forest_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(y_test)
print(y_pred)
print(f'Test Accuracy: {accuracy}')

feature_importances = random_forest_model.feature_importances_

# Set a threshold for importance score (adjust based on your analysis)
threshold = 0.01

# Identify important SNP indices based on importance scores
important_snp_indices = [i for i, importance in enumerate(feature_importances) if importance > threshold]

# Extract the corresponding SNPs from your dataset (assuming your_snp_list is a list of lists)
important_snps = [[cow[i] for i in important_snp_indices] for cow in X]

# Plot the feature importances if needed
feature_importances = random_forest_model.feature_importances_
plt.bar(range(len(feature_importances)), feature_importances)
plt.xlabel('Feature Index')
plt.ylabel('Feature Importance')
plt.title('Random Forest Feature Importances')
plt.show()
