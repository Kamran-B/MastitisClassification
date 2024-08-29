import random
import time

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from DataQuality.to_array import bit_reader
from transformers import BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, classification_report

from DataQuality.funtional_consequences import *


# Load data from files

herd = load_2d_array_from_file("Data/breed_herdxyear_lact1_sorted.txt")
X = bit_reader("Data/output_hd_exclude_4000top_SNPs_binary.txt")
y = load_1d_array_from_file("Data/mast_lact1_sorted_herd.txt")

# Start time
start_time = time.time()

X = np.array(X)
impact_scores = np.array(get_impact_scores()['impact_score'])
X = 1 + X * impact_scores
X = X.tolist()

# End time
end_time = time.time()

# Elapsed time in seconds
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")

# Combine herd data with X
for rowX, rowH in zip(X, herd):
    for value in rowH:
        rowX.append(value)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Clean up original data
del X, y

seed_value = 42

# Augment training data
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

# Clean up training data
del X_train, y_train


# Prepare the data for the transformer model
class GeneticDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=512):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = " ".join(map(str, self.sequences[idx]))
        label = self.labels[idx]
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze() for key, val in encoding.items()}
        item["labels"] = torch.tensor(label)
        return item


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_dataset = GeneticDataset(X_train_augmented, y_train_augmented, tokenizer)
test_dataset = GeneticDataset(X_test_augmented, y_test_augmented, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Learning rate scheduler
total_steps = len(train_loader) * 3  # Assuming 3 epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

# Loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop with learning rate scheduling and evaluation at each epoch
for epoch in range(3):  # Loop over the specified number of epochs
    model.train()  # Set the model to training mode
    total_loss = 0  # Initialize total loss for the current epoch
    i = 0  # Batch counter for tracking progress within the epoch

    for batch in train_loader:  # Loop over batches in the training dataset
        optimizer.zero_grad()  # Clear previous gradients

        # Move input data and labels to the specified device (e.g., GPU)
        inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
        labels = batch["labels"].to(device)

        # Forward pass: compute model predictions and calculate loss
        outputs = model(**inputs)
        loss = loss_fn(outputs.logits, labels)
        total_loss += loss.item()  # Accumulate loss for reporting

        # Backward pass and optimization
        loss.backward()
        optimizer.step()  # Update model parameters
        scheduler.step()  # Update learning rate based on the scheduler

        i += 1  # Increment batch counter
        # Print training progress for each batch
        print(
            f'Epoch: {epoch}, Loop {i} of {len(train_loader)}, Loss: {loss.item()}, Learning Rate: {optimizer.param_groups[0]["lr"]}'
        )

    # Calculate and print the average training loss for the epoch
    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch: {epoch}, Average Training Loss: {avg_train_loss}")

    # Evaluate the model on the test set after each epoch
    model.eval()  # Set the model to evaluation mode
    preds = []  # List to store model predictions
    true_labels = []  # List to store true labels

    with torch.no_grad():  # Disable gradient computation during evaluation
        for batch in test_loader:  # Loop over batches in the test dataset
            # Move input data and labels to the specified device (e.g., GPU)
            inputs = {
                key: val.to(device) for key, val in batch.items() if key != "labels"
            }
            labels = batch["labels"].to(device)

            # Forward pass: compute model predictions
            outputs = model(**inputs)
            _, predicted = torch.max(outputs.logits, 1)

            # Collect predictions and true labels for evaluation metrics
            preds.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Calculate and print the accuracy of the model on the test set
    accuracy = accuracy_score(true_labels, preds)
    print(f"Epoch: {epoch}, Test Accuracy: {accuracy}")

    # Generate and print a detailed classification report
    report = classification_report(
        true_labels,
        preds,
        target_names=["No mastitis (Control)", "Mastitis Present (Case)"],
    )
    print(report)


conf_matrix = confusion_matrix(true_labels, preds)
print(conf_matrix)
