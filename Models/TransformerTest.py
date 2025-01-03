import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from DataQuality.to_array import bit_reader
from transformers import BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm  # Progress bar
from torch.cuda.amp import GradScaler, autocast  # Mixed precision

def read_numbers_from_file2(file_path):
    numbers = []
    try:
        with open(file_path, "r") as file:
            for line in file:
                row = list(map(int, line.strip().split()))
                numbers.append(row)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return np.array(numbers)


def read_numbers_from_file(file_path):
    numbers = []
    try:
        with open(file_path, "r") as file:
            for line in file:
                numbers.append(int(line.strip()))
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return numbers


herd = read_numbers_from_file2("../Data/breed_herdxyear_lact1_sorted.txt")
X = bit_reader("../Data/output_hd_exclude_4000top_SNPs_binary.txt")
y = read_numbers_from_file("../Data/mast_lact1_sorted_herd.txt")

# Combine herd data into SNP data
for rowX, rowH in zip(X, herd):
    rowX.extend(rowH)  # Directly extend rather than nested loop

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
del X, y  # Free up memory

# Data augmentation
def duplicate_and_insert(original_list, target_list, original_target_labels, target_labels, label_value, num_duplicates, seed=None):
    random.seed(seed)
    for i, (original, label) in enumerate(zip(original_list, original_target_labels)):
        if label == label_value:
            for _ in range(num_duplicates):
                random_position = random.randint(0, len(target_list))
                target_list.insert(random_position, original.copy())
                target_labels.insert(random_position, label_value)


# Augmenting the training and test sets
seed_value = 42
X_train_augmented, y_train_augmented = X_train.copy(), y_train.copy()
duplicate_and_insert(X_train, X_train_augmented, y_train, y_train_augmented, 1, 16, seed=seed_value)
X_test_augmented, y_test_augmented = X_test.copy(), y_test.copy()
duplicate_and_insert(X_test, X_test_augmented, y_test, y_test_augmented, 1, 16, seed=seed_value)
del X_train, y_train  # Free up memory

# Dataset and DataLoader preparation
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


# Tokenizer and DataLoader setup
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = GeneticDataset(X_train_augmented, y_train_augmented, tokenizer)
test_dataset = GeneticDataset(X_test_augmented, y_test_augmented, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Model and training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 3  # Assuming 3 epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
loss_fn = torch.nn.CrossEntropyLoss()
scaler = GradScaler()  # For mixed precision training

# Function to evaluate model
def evaluate_model(model, test_loader, device):
    model.eval()
    preds, true_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
            labels = batch["labels"].to(device)
            outputs = model(**inputs)
            _, predicted = torch.max(outputs.logits, 1)
            preds.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, preds)
    report = classification_report(true_labels, preds, target_names=["No mastitis (Control)", "Mastitis Present (Case)"])
    print(f"Test Accuracy: {accuracy}\n{report}")
    conf_matrix = confusion_matrix(true_labels, preds)
    print(f"Confusion Matrix:\n{conf_matrix}")
    return accuracy

# Training loop with mixed precision
for epoch in range(3):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
        optimizer.zero_grad()
        inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
        labels = batch["labels"].to(device)

        with autocast():  # Mixed precision
            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}: Average Training Loss: {avg_train_loss}")

    # Evaluate model after each epoch
    evaluate_model(model, test_loader, device)

print("Training complete!")
