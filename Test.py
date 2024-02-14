import dask
dask.config.set({'dataframe.query-planning': True})
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("about to load data")
# Load your dataset
file_path = 'output_hd_exclude.raw'

# Assuming your data is comma-separated
delimiter = ','

# Read the raw file and create a 2D array
data = []
i = 1
with open(file_path, 'r') as file:
    # Skip the header if present
    header = file.readline()

    # Read the data into a 2D array
    for line in file:
        print("Line " + str(i) + " done.")
        row = line.strip().split(delimiter)
        data.append(row)
        i += 1

print("Done Loading CSV")

print(data)

# Assuming the last column is the target variable
X = [row[6:-1] for row in data]
y = [row[-1] for row in data]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train)

#Ensure that X_train and X_test have at least one feature
if not X_train or not any(X_train):
    raise ValueError("X_train does not contain valid features.")
if not X_test or not any(X_test):
    raise ValueError("X_test does not contain valid features.")

print("about to train model")
# Create a random forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy}')