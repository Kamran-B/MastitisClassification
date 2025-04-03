import time
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader, Dataset

from Models.Transformer.Attention import Transformer
from DataQuality.funtional_consequences import *
from DataQuality.to_array import bit_reader
from DataQuality.model_saving import *

'''HYPERPARAMETERS'''
VOCAB_SIZE = 3 # Size of the source vocabulary
NUM_CLASSES = 2   # Binary classification
EMBED_DIM = 256 # (Needs to be divisible by num_heads)
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 3 # Only encoder layers are used
FFN_DIM = 1024 # Hidden dimension of Feed Forward networks
MAX_SEQ_LEN = 500 # Maximum sequence length for positional encoding
DROPOUT = 0.1
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
NUM_EPOCHS = 10
PAD_IDX = 0 # Assuming 0 is the padding index

device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else
                          "cpu")

class SNPDataset(Dataset):
    def __init__(self, snp_data, labels):
        self.snp_data = torch.tensor(snp_data, dtype=torch.long)  # SNPs as integers (0,1,2)
        self.labels = torch.tensor(labels, dtype=torch.float32)  # Binary labels (0 or 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.snp_data[idx], self.labels[idx]

def prepare_data(seed_value, top_snps):
    breed_herd_year = 'Data/BreedHerdYear/breed_herdxyear_lact1_sorted.txt'
    phenotypes = 'Data/Phenotypes/phenotypes_sorted.txt'

    herd = load_2d_array_from_file(breed_herd_year)
    X = bit_reader(top_snps)
    y = load_1d_array_from_file(phenotypes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed_value, stratify=y)

    return X_train, X_test, y_train, y_test


def augment_data(X, y, seed_value):
    X_aug, y_aug = X.copy(), y.copy()
    duplicate_and_insert(X, X_aug, y, y_aug, 1, 16, seed=seed_value)
    return X_aug, y_aug

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()  # Set model to training mode
    total_loss = 0
    total_correct = 0
    total_samples = 0
    start_time = time.time()

    for i, (src, labels) in enumerate(dataloader):
        src = src.to(device)
        labels = labels.to(device)  # Shape: (B,)
        labels = labels.long()


        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(src)  # Shape: (B, num_classes)

        # Calculate loss
        loss = criterion(logits, labels)  # CrossEntropyLoss expects (B, C) and (B)

        # Calculate accuracy
        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == labels).sum().item()
        total_correct += correct
        total_samples += labels.size(0)

        # Backward pass and optimization
        loss.backward()
        # Optional: Gradient Clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % 10 == 0:  # Print progress every 10 batches
            batch_acc = correct / labels.size(0)
            print(f"  Batch {i + 1}/{len(dataloader)}, Loss: {loss.item():.4f}, Acc: {batch_acc:.4f}")

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / total_samples
    epoch_time = time.time() - start_time
    return avg_loss, avg_acc, epoch_time


def evaluate(model, dataloader, criterion, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient calculations
        for src, labels in dataloader:
            src = src.to(device)
            labels = labels.to(device)
            labels = labels.long()


            # Forward pass
            logits = model(src)  # Shape: (B, num_classes)

            # Calculate loss
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


if __name__ == '__main__':
    top_snps = "Data/TopSNPs/rf/top500_SNPs_rf_binary.txt"
    seed = 50
    X_train, X_test, y_train, y_test = prepare_data(seed, top_snps)
    X_train_aug, y_train_aug = augment_data(X_train, y_train, seed)
    X_test_aug, y_test_aug = augment_data(X_test, y_test, seed)

    train_dataset = SNPDataset(X_train_aug, y_train_aug)
    test_dataset = SNPDataset(X_test_aug, y_test_aug)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Transformer(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        ffn_dim=FFN_DIM,
        num_classes=NUM_CLASSES,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
        pad_idx=PAD_IDX
    ).to(device)

    # Loss Function - CrossEntropyLoss works for multi-class (including binary)
    criterion = nn.CrossEntropyLoss()  # No need for ignore_index if labels don't contain padding

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


    # --- Main Training Loop ---
    print("Starting Training for Binary Classification...")
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")

        train_loss, train_acc, epoch_time = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  Epoch Time: {epoch_time:.2f} seconds")

    print("\nTraining Finished.")

    # --- Optional: Save the model ---
    # torch.save(model.state_dict(), 'transformer_classifier_model.pth')
    # print("Model saved to transformer_classifier_model.pth")

    # --- Testing ---
    print("\nStarting Testing...")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)  # Using val set as test set
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

