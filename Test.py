'''df = pd.read_csv('/Users/gavinlynch04/Desktop/CSC487_Project_Data/data/output.csv', nrows=5, header=0)

# Extract and print the first row
first_line = df.iloc[3]
print(first_line)'''

import dask
dask.config.set({'dataframe.query-planning': True})
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("about to load data")
# Load your dataset
file_path = '/Users/gavinlynch04/Desktop/CSC487_Project_Data/data/merged.csv'

with ProgressBar():
    df = dd.read_csv(file_path, sample=5000000, blocksize=100e6).head(5).compute()

print("Done Loading CSV")

# Assuming the last column is the target variable
X = df.iloc[:, 6:-1].values
y = df.iloc[:, -1].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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