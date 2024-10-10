import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from DataQuality.to_array import bit_reader
from transformers import BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, classification_report

from DataQuality.funtional_consequences import *


TOP_PERFORMANCE_FILE = "top_performances.json"
TOP_K = 10
MODEL_SAVE_PATH = "./saved_models"

# Create directory for saving models if it doesn't exist
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

# Load the top 10 performances from file or initialize an empty list
def load_top_performances():
    if os.path.exists(TOP_PERFORMANCE_FILE):
        with open(TOP_PERFORMANCE_FILE, 'r') as f:
            return json.load(f)
    return []

# Save the top 10 performances back to file
def save_top_performances(top_performances):
    with open(TOP_PERFORMANCE_FILE, 'w') as f:
        json.dump(top_performances, f, indent=4)

# Update the top 10 list if the current accuracy is better than the worst in the list
def update_top_performances(top_performances, accuracy, model_name):
    if len(top_performances) < TOP_K or accuracy > min([p["accuracy"] for p in top_performances]):
        # Add the new performance and sort the list
        top_performances.append({"accuracy": accuracy, "model_name": model_name})
        top_performances = sorted(top_performances, key=lambda x: x["accuracy"], reverse=True)
        # Keep only the top 10
        if len(top_performances) > TOP_K:
            # Remove the worst performance and delete the associated model file
            worst_performance = top_performances.pop()
            model_to_delete = os.path.join(MODEL_SAVE_PATH, worst_performance["model_name"])
            if os.path.exists(model_to_delete):
                os.remove(model_to_delete)
        # Save updated list
        save_top_performances(top_performances)
    else:
        # If not top 10, delete the current model
        model_to_delete = os.path.join(MODEL_SAVE_PATH, model_name)
        if os.path.exists(model_to_delete):
            os.remove(model_to_delete)


# Load data from files

herd = load_2d_array_from_file("../Data/breed_herdxyear_lact1_sorted.txt")
X = bit_reader("../Data/output_hd_exclude_4000top_SNPs_binary.txt")
y = load_1d_array_from_file("../Data/mast_lact1_sorted_herd.txt")


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

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

train_dataset = GeneticDataset(X_train_augmented, y_train_augmented, tokenizer)
test_dataset = GeneticDataset(X_test_augmented, y_test_augmented, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the BERT model
model = BertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=2)
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
for epoch in range(3):  # Number of epochs
    model.train()
    total_loss = 0
    i = 0
    for batch in train_loader:
        optimizer.zero_grad()
        inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
        labels = batch["labels"].to(device)

        outputs = model(**inputs)
        loss = loss_fn(outputs.logits, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        i += 1
        print(
            f'Epoch: {epoch}, Loop {i} of {len(train_loader)}, Loss: {loss.item()}, Learning Rate: {optimizer.param_groups[0]["lr"]}'
        )

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch: {epoch}, Average Training Loss: {avg_train_loss}")

    # Evaluate the model on the test set after each epoch
    model.eval()
    preds = []
    true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = {
                key: val.to(device) for key, val in batch.items() if key != "labels"
            }
            labels = batch["labels"].to(device)

            outputs = model(**inputs)
            _, predicted = torch.max(outputs.logits, 1)

            preds.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, preds)
    print(f"Epoch: {epoch}, Test Accuracy: {accuracy}")
    report = classification_report(
        true_labels,
        preds,
        target_names=["No mastitis (Control)", "Mastitis Present (Case)"],
    )
    print(report)
    model_name = f"model_epoch{epoch}_acc{accuracy:.4f}.pt"
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, model_name))

    # Load current top performances and update
    top_performances = load_top_performances()
    update_top_performances(top_performances, accuracy, model_name)

# Confusion matrix
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(true_labels, preds)
print(conf_matrix)
