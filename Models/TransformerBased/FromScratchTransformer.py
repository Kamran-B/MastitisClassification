import time
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch.optim as optim
from torch.amp import autocast, GradScaler
from DataQuality.funtional_consequences import *
from DataQuality.to_array import bit_reader
from DataQuality.model_saving import *
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = "1"
torch.set_float32_matmul_precision('high')


class SNPDataset(Dataset):
    def __init__(self, snp_data, labels):
        self.snp_data = torch.tensor(snp_data, dtype=torch.long)  # SNPs as integers (0,1,2)
        self.labels = torch.tensor(labels, dtype=torch.float32)  # Binary labels (0 or 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.snp_data[idx], self.labels[idx]


class SNPTransformer(nn.Module):
    def __init__(self, input_dim=3, d_model=60, num_heads=6, num_layers=6, window_size=500, stride=500):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.embedding = nn.Embedding(3, d_model)  # SNPs (0,1,2) to embeddings
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=600, batch_first=True, norm_first=True) # Change to 128 for less memory but slower convergence
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)  # Final classification

    def forward(self, x):
        B, L = x.shape  # (Batch size, Sequence length)
        x = self.embedding(x)  # (B, L, d_model)

        # Apply sliding window
        outputs = []
        for start in range(0, L - self.window_size + 1, self.stride):
            window = x[:, start:start + self.window_size, :]  # Extract window
            window = window.permute(1, 0, 2)  # (Seq_len, Batch, d_model) for Transformer
            out = self.transformer(window)  # Process window
            out = out.mean(dim=0)  # Mean pooling across window
            outputs.append(out)

        # Stack and aggregate all windows
        outputs = torch.stack(outputs, dim=1)  # (B, num_windows, d_model)
        final_out = outputs.mean(dim=1)  # Aggregate over all windows

        return self.fc(final_out).squeeze()  # Final classification


def train_fold(model, train_loader, epochs=15, lr=1e-4, device="cpu"):
    """Trains the model for one fold."""
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # Use GradScaler only if device is CUDA
    use_amp = device == "cuda"
    scaler = GradScaler('cuda', enabled=use_amp)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        for i, (snps, labels) in enumerate(train_loader):
            snps, labels = snps.to(device), labels.to(device)

            optimizer.zero_grad()

            # Use autocast only if device is CUDA
            with autocast('cuda', enabled=use_amp):
                outputs = model(snps)
                loss = criterion(outputs, labels)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        end_time = time.time()
        elapsed_time = end_time - start_time
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}, Time: {elapsed_time:.2f}s")

    return model # Return the trained model


import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def evaluate_fold(model, val_loader, device="cpu"):
    """Evaluates the model on the validation set for one fold."""
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_outputs = [] # Store raw outputs for AUC

    with torch.no_grad():
        for snps, labels in val_loader:
            snps, labels = snps.to(device), labels.to(device)
            outputs = model(snps)
            # Store raw outputs (logits or probabilities after sigmoid)
            # Apply sigmoid here if using BCEWithLogitsLoss for AUC calculation
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float() # Threshold probabilities

            all_outputs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Handle cases where a class might be missing in preds (e.g., during early epochs)
    # Use zero_division=0 to avoid warnings and return 0.0 for metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        # AUC needs probabilities/scores, not binary predictions
        auc = roc_auc_score(all_labels, all_outputs)
    except ValueError:
        # Handle cases where only one class is present in labels or outputs
        auc = 0.0 # Or np.nan, depending on how you want to handle it

    print(f"  Validation Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    # print("Classification Report: \n", classification_report(all_labels, all_preds, zero_division=0))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }


def prepare_data(top_snps):
    breed_herd_year = 'Data/BreedHerdYear/breed_herdxyear_lact1_sorted.txt'
    phenotypes = 'Data/Phenotypes/phenotypes_sorted.txt'

    herd = load_2d_array_from_file(breed_herd_year)
    X = bit_reader(top_snps)
    y = load_1d_array_from_file(phenotypes)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed_value, stratify=y)
    #return X_train, X_test, y_train, y_test
    return np.array(X), np.array(y, dtype=int)

def augment_data(X, y, seed_value):
    X_aug, y_aug = X.copy(), y.copy()
    return duplicate_and_insert_numpy_fast(X, X_aug, y, y_aug, 1, 16, seed=seed_value)


def run_cross_validation(n_splits=5, seed=42, epochs=15, lr=1e-4, batch_size=32, top_snps_file=None, logging_file="cv_results.json"):
    """Runs Stratified K-Fold Cross-Validation."""

    # Set random seed for reproducibility of splitting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Load the full dataset
    X, y = prepare_data(top_snps_file)

    # 2. Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # 3. Prepare to store results for each fold
    fold_results = []

    # Determine device
    device = ("mps" if torch.backends.mps.is_available() else
              "cuda" if torch.cuda.is_available() else
              "cpu")
    print(f"Using device: {device}")

    # 4. Loop through folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print("-" * 30)
        print(f"Fold {fold + 1}/{n_splits}")
        print("-" * 30)

        # --- Get data for this fold ---
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        print(f"  Fold Train shapes: X={X_train_fold.shape}, y={y_train_fold.shape}")
        print(f"  Fold Train distribution: {np.bincount(y_train_fold)}")
        print(f"  Fold Validation shapes: X={X_val_fold.shape}, y={y_val_fold.shape}")
        print(f"  Fold Validation distribution: {np.bincount(y_val_fold)}")


        # --- Augment ONLY the training data for this fold ---
        # Use a different seed for augmentation per fold if desired, or reuse main seed
        fold_aug_seed = seed + fold
        X_train_fold_aug, y_train_fold_aug = augment_data(X_train_fold, y_train_fold, seed_value=fold_aug_seed)
        X_val_fold_aug, y_val_fold_aug = augment_data(X_val_fold, y_val_fold, seed_value=fold_aug_seed)

        # --- Create Datasets and DataLoaders for this fold ---
        train_dataset_fold = SNPDataset(X_train_fold_aug, y_train_fold_aug)
        val_dataset_fold = SNPDataset(X_val_fold_aug, y_val_fold_aug) # Probably should not augment this

        train_loader_fold = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True)
        val_loader_fold = DataLoader(val_dataset_fold, batch_size=batch_size, shuffle=False)

        # --- Initialize a NEW model for this fold ---
        model = SNPTransformer()

        # --- Train the model for this fold ---
        print("  Training...")
        trained_model = train_fold(model, train_loader_fold, epochs=epochs, lr=lr, device=device)

        # --- Evaluate the model on the validation set for this fold ---
        print("  Evaluating...")
        metrics = evaluate_fold(trained_model, val_loader_fold, device=device)
        fold_results.append(metrics)
        print("-" * 30)

    # 5. Aggregate and report results
    print("\n" + "=" * 30)
    print("Cross-Validation Results Summary")
    print("=" * 30)

    avg_metrics = {}
    std_metrics = {}
    metric_keys = fold_results[0].keys()

    for key in metric_keys:
        values = [results[key] for results in fold_results]
        avg_metrics[key] = np.mean(values)
        std_metrics[key] = np.std(values)
        print(f"Average {key.capitalize()}: {avg_metrics[key]:.4f} +/- {std_metrics[key]:.4f}")

    # Log results
    log_data = {
        "parameters": {
            "n_splits": n_splits,
            "seed": seed,
            "epochs": epochs,
            "learning_rate": lr,
            "batch_size": batch_size,
            "top_snps_file": top_snps_file,
            "device": device
        },
        "fold_results": fold_results,
        "average_metrics": avg_metrics,
        "std_dev_metrics": std_metrics
    }
    try:
        with open(logging_file, "w") as f:
            json.dump(log_data, f, indent=4)
        print(f"\nResults saved to {logging_file}")
    except Exception as e:
        print(f"\nError saving results to {logging_file}: {e}")

    return avg_metrics


if __name__ == "__main__":
    start_time = time.time()

    # --- Configuration ---
    N_SPLITS = 3        # Number of folds (e.g., 5 or 10)
    SEED = 123          # Master seed for reproducibility
    EPOCHS = 10          # Number of epochs per fold (adjust as needed)
    LEARNING_RATE = 5e-4
    BATCH_SIZE = 32
    TOP_SNPS_FILE = "Data/TopSNPs/rf/top500_SNPs_rf_binary.txt" # Path to your SNP file
    LOGGING_FILE = "Logging/Transformer/cv_results_transformer.json" # Output log file

    # Create logging directory if it doesn't exist
    log_dir = os.path.dirname(LOGGING_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Created logging directory: {log_dir}")

    run_cross_validation(
        n_splits=N_SPLITS,
        seed=SEED,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        top_snps_file=TOP_SNPS_FILE,
        logging_file=LOGGING_FILE
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal Elapsed time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

