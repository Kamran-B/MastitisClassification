import time
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from torch import optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score
from Models.Transformer.Attention import Transformer
from DataQuality.funtional_consequences import *
from DataQuality.to_array import bit_reader
from DataQuality.model_saving import *

'''HYPERPARAMETERS'''
CLS_TOKEN_ID = 4
VOCAB_SIZE = 5 # Size of the source vocabulary
NUM_CLASSES = 2   # Binary classification
EMBED_DIM = 16 # (Needs to be divisible by num_heads)
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 4 # Only encoder layers are used
FFN_DIM = 128 # Hidden dimension of Feed Forward networks
MAX_SEQ_LEN = 501 # Maximum sequence length for positional encoding
DROPOUT = 0.15
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
NUM_EPOCHS = 10
PAD_IDX = 0 # Assuming 0 is the padding index

device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else
                          "cpu")
print('device:', device)

class SNPDataset(Dataset):
    def __init__(self, snp_data, labels, cls_token_id, pad_idx): # Added pad_idx for clarity
        # snp_data received here should ALREADY BE SHIFTED (values 1, 2, 3)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.cls_token = torch.tensor([cls_token_id], dtype=torch.long)
        self.pad_idx = pad_idx

        processed_snp_data = []
        for snp_sequence in snp_data:
            # Ensure input is tensor and long type
            snp_tensor = torch.tensor(snp_sequence, dtype=torch.long)
            # Concatenate CLS token at the beginning
            modified_sequence = torch.cat((self.cls_token, snp_tensor), dim=0)
            processed_snp_data.append(modified_sequence)

        # Pad sequences if necessary (using the correct pad_idx=0)
        # This is only needed if sequences have variable lengths after adding CLS
        # If bit_reader guarantees fixed length L, all sequences here are L+1
        # max_len_with_cls = max(len(seq) for seq in processed_snp_data)
        # print(f"Max sequence length after adding CLS: {max_len_with_cls}")
        # if MAX_SEQ_LEN < max_len_with_cls:
        #      print(f"Warning: MAX_SEQ_LEN ({MAX_SEQ_LEN}) is less than max sequence length with CLS ({max_len_with_cls}). Increase MAX_SEQ_LEN.")

        # Example padding (uncomment if needed):
        # padded_snp_data = torch.nn.utils.rnn.pad_sequence(
        #     processed_snp_data, batch_first=True, padding_value=self.pad_idx # Use 0 for padding
        # )
        # self.snp_data = padded_snp_data

        # If lengths are fixed after adding CLS, just stack:
        self.snp_data = torch.stack(processed_snp_data)

    def __len__(self):
        # This method MUST exist and return the total number of samples
        return len(self.labels)  # Or len(self.snp_data) if that's more reliable

    def __getitem__(self, idx):
        return self.snp_data[idx], self.labels[idx]

def prepare_data(seed_value, top_snps):
    breed_herd_year = 'Data/BreedHerdYear/breed_herdxyear_lact1_sorted.txt'
    phenotypes = 'Data/Phenotypes/phenotypes_sorted.txt'

    herd = load_2d_array_from_file(breed_herd_year)
    print("Reading bits...")
    X = bit_reader(top_snps) # Reads SNP data as 0, 1, 2
    print(f"Finished reading bits. Original shape: {X.shape if hasattr(X, 'shape') else 'N/A'}")

    # Add 1 to all SNP values (0->1, 1->2, 2->3)
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    X = X + 1
    print(f"Shifted SNP values (now 1, 2, 3). Shape: {X.shape}")
    # ------------------------

    y = load_1d_array_from_file(phenotypes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed_value, stratify=y)

    return X_train, X_test, y_train, y_test


def augment_data(X, y, seed_value):
    X_aug, y_aug = X.copy(), y.copy()
    X_aug, y_aug = duplicate_and_insert_memory_efficient(X, X_aug, y, y_aug, 1, 16, seed=seed_value)
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
        #Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    all_eval_preds = []
    all_eval_labels = []

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

            all_eval_preds.append(predictions.detach().cpu())
            all_eval_labels.append(labels.detach().cpu())
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    all_eval_preds_tensor = torch.cat(all_eval_preds)
    all_eval_labels_tensor = torch.cat(all_eval_labels)

    eval_accuracy = (all_eval_preds_tensor == all_eval_labels_tensor).float().mean().item()
    eval_precision = precision_score(all_eval_labels_tensor.numpy(), all_eval_preds_tensor.numpy(), average='binary',
                                     zero_division=0)
    eval_recall = recall_score(all_eval_labels_tensor.numpy(), all_eval_preds_tensor.numpy(), average='binary',
                               zero_division=0)
    eval_f1 = f1_score(all_eval_labels_tensor.numpy(), all_eval_preds_tensor.numpy(), average='binary', zero_division=0)

    return avg_loss, eval_accuracy, eval_precision, eval_recall, eval_f1


if __name__ == '__main__':
    top_snps = "Data/TopSNPs/rf/top500_SNPs_rf_binary.txt"
    seed = 555
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # Numpy random module
    torch.manual_seed(seed)  # PyTorch CPU random seed

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch CUDA random seed (all GPUs)
        torch.cuda.manual_seed_all(seed)

    X_train, X_test, y_train, y_test = prepare_data(seed, top_snps)
    X_train_aug, y_train_aug = augment_data(X_train, y_train, seed)
    X_test_aug, y_test_aug = augment_data(X_test, y_test, seed)
    labels_for_weights = y_train_aug  # Or y_train
    unique_classes = np.unique(labels_for_weights)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=labels_for_weights)
    # Convert weights to a tensor and move to device
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Calculated Class Weights: {class_weights_tensor.cpu().numpy()}")

    print(X_train_aug.shape, y_train_aug.shape)

    train_dataset = SNPDataset(X_train_aug, y_train_aug, CLS_TOKEN_ID, PAD_IDX)
    test_dataset = SNPDataset(X_test_aug, y_test_aug, CLS_TOKEN_ID, PAD_IDX)

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
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)  # No need for ignore_index if labels don't contain padding

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)


    # --- Main Training Loop ---
    print("Starting Training for Binary Classification...")
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")

        train_loss, train_acc, epoch_time = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val P:   {val_prec:.4f} | Val R:   {val_rec:.4f} | Val F1:   {val_f1:.4f}")
        print(f"  Epoch Time: {epoch_time:.2f} seconds")

    print("\nTraining Finished.")

    # --- Optional: Save the model ---
    # torch.save(model.state_dict(), 'transformer_classifier_model.pth')
    # print("Model saved to transformer_classifier_model.pth")

    # --- Testing ---
    print("\nStarting Testing...")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)  # Using val set as test set
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

