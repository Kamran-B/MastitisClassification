from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm


print("about to load data")
# Load your dataset
file_path = '.git /output_hd_exclude.raw'
file_path2 = './mast_lact1.phen'
# Assuming your data is space-separated
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
data2_sorted = [row2 for _, row2 in sorted(zip(data_sorted, data2_filtered), key=lambda x: x[0][0])]

X = [row[6:] for row in data_sorted]
y = [row[2] for row in data2_sorted]

print(y[0])
print(y)

print(y.count(0)/len(y))



# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#print(X_train)

#Ensure that X_train and X_test have at least one feature
if not X_train or not any(X_train):
    raise ValueError("X_train does not contain valid features.")
if not X_test or not any(X_test):
    raise ValueError("X_test does not contain valid features.")

print("about to train model")
# Create a random forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1)

# Train the model and track training and validation accuracy over different numbers of trees
train_accuracies = []
val_accuracies = []

for n_trees in range(1, 5):  # Assuming a range from 1 to 100 trees
    rf_model.set_params(n_estimators=n_trees)
    rf_model.fit(X_train, y_train)

    print(n_trees)
    # Calculate training accuracy
    y_train_pred = rf_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_accuracies.append(train_accuracy)

    # Calculate validation accuracy
    y_val_pred = rf_model.predict(X_test)
    val_accuracy = accuracy_score(y_test, y_val_pred)
    val_accuracies.append(val_accuracy)

# Plot the training and validation accuracy over different numbers of trees
plt.plot(range(1, 5), train_accuracies, label='Training Accuracy')
plt.plot(range(1, 5), val_accuracies, label='Validation Accuracy')
plt.xlabel('Number of Trees (Estimators)')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy over Number of Trees')
plt.legend()
plt.show()

'''
# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy}')
'''
