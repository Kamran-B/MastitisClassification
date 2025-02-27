import time

from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer
from DataQuality.funtional_consequences import *
from DataQuality.to_array import bit_reader
from DataQuality.model_saving import *
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = "1"
torch.set_float32_matmul_precision('high')



# Define dataset class
class SNPDataset(Dataset):
    def __init__(self, snp_data, labels):
        self.snp_data = torch.tensor(snp_data, dtype=torch.long)  # SNPs as integers (0,1,2)
        self.labels = torch.tensor(labels, dtype=torch.float32)  # Binary labels (0 or 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.snp_data[idx], self.labels[idx]


class SNPTransformer(nn.Module):
    def __init__(self, input_dim=3, d_model=16, num_heads=4, num_layers=4, window_size=500, stride=500):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.embedding = nn.Embedding(3, d_model)  # SNPs (0,1,2) to embeddings
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=128, batch_first=True, norm_first=True) # Change to 128 for less memory but slower convergence
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


# Example Training Setup
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

def train_model(model, train_loader, val_loader, epochs=20, lr=6e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()  # Gradient scaler for FP16 stability
    accuracies = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        i = 0
        start_time = time.time()

        for snps, labels in train_loader:
            snps, labels = snps.to(device), labels.to(device)

            optimizer.zero_grad()

            with autocast():  # Enable FP16 computation
                outputs = model(snps)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()  # Scale gradients
            scaler.step(optimizer)
            scaler.update()  # Adjust scaling for next step

            i += 1
            print("Current Step: {} of {}, loss: {}".format(i, len(train_loader), loss.item()))

            total_loss += loss.item()
        end_time = time.time()  # End timer
        elapsed_time = end_time - start_time

        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(train_loader)}")
        accuracies.append(evaluate_model(model, val_loader))  # Evaluate after training
    return accuracies




import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def evaluate_model(model, test_loader):
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else
                          "cpu")
    print("Using Device: {}".format(device))
    model.to(device)
    model.eval()  # Set model to evaluation mode

    all_preds = []
    all_labels = []

    with torch.no_grad():  # No gradient calculations during evaluation
        for snps, labels in test_loader:
            snps, labels = snps.to(device), labels.to(device)
            outputs = model(snps)  # Get model predictions
            preds = (outputs >= 0.5).float()  # Convert probabilities to 0 or 1

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)

    print("Classification Report: \n", classification_report(all_labels, all_preds))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    return accuracy


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

def main(seed, epochs, printStats=True, savePerf=False, top_snps=None):
    X_train, X_test, y_train, y_test = prepare_data(seed, top_snps)
    X_train_aug, y_train_aug = augment_data(X_train, y_train, seed)
    X_test_aug, y_test_aug = augment_data(X_test, y_test, seed)

    dataset = SNPDataset(X_train_aug, y_train_aug)
    dataset2 = SNPDataset(X_test_aug, y_test_aug)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)  # Adjust batch size as needed
    val_loader = DataLoader(dataset2, batch_size=32
                            , shuffle=False)

    model = SNPTransformer()
    accuracies = train_model(model, train_loader, val_loader)
    print("\nEvaluating model on test data...")
    evaluate_model(model, val_loader)  # Evaluate after training

    return accuracies

def EvalScript(iterations, top_snps, logging_file):
    results = []
    iterations += 1
    for run_num in range(1, iterations):
        seed = random.randint(1, 1000)
        print(f"Running with seed: {seed}")
        accuracies = main(seed=565, epochs=3, printStats=False, savePerf=True, top_snps=top_snps)
        results.append({
            "run_number": run_num, "seed": seed,
            "accuracies_per_epoch": accuracies,
            "average_accuracy": np.mean(accuracies),
            "max_accuracy": np.max(accuracies)
        })
        print(f"Run {run_num} Avg Accuracy: {np.mean(accuracies)}, Max Accuracy: {np.max(accuracies)}")

    overall_accuracies = np.concatenate([np.array(result["accuracies_per_epoch"]) for result in results])
    print(f"Overall Avg Accuracy: {overall_accuracies.mean()}")
    with open(logging_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {logging_file}")


if __name__ == "__main__":
    start_time = time.time()  # Start timer
    EvalScript(1, "Data/TopSNPs/xgboost/top500_SNPs_xgb_binary.txt", "Logging/Transformer/adsfadsf.json")
    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    '''# Define Transformer Model
    class SNPClassifier(nn.Module):
        def __init__(self, input_dim=500, d_model=64, n_heads=4, num_layers=4):
            super(SNPClassifier, self).__init__()

            # Embedding Layer (since input is categorical 0,1,2)
            self.embedding = nn.Embedding(num_embeddings=3, embedding_dim=d_model)  # SNPs are {0,1,2}

            # Transformer Encoder
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=256, dropout=0.1)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            # Classification head
            self.fc = nn.Linear(d_model, 1)  # Output a single classification value
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.embedding(x)  # Convert SNPs to embeddings
            x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch, d_model)
            x = self.transformer(x)
            x = x.mean(dim=0)  # Mean-pooling over sequence length
            x = self.fc(x)
            return self.sigmoid(x).squeeze(1)  # Output probability'''


