import math
import time
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from DataQuality.funtional_consequences import *
from DataQuality.to_array import bit_reader
from DataQuality.model_saving import *
import torch.nn.functional as F
import math
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = "1"
torch.set_float32_matmul_precision('high')


class Config:
    def __init__(self):
        self.vocab_size = 3
        self.d_model = 16
        self.n_layers = 4
        self.n_heads = 4
        self.d_kv_comp = 6
        self.d_rope = 4
        self.seq_len = 500
        self.batch_size = 8
        self.ffn_dim = 128
config = Config()



class RotaryEmbedding(nn.Module):
    def __init__(self, dim, scale=40):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even for rotary embeddings"
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim//2, 2).float() / (dim//2)))
        self.register_buffer("inv_freq", inv_freq)
        self.scale = 40

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq) / self.scale
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary(x, cos, sin):
    """
    Apply rotary embeddings to the first half of x.
    """
    # Split x into two parts: one for rotary embeddings and the other untouched
    x_rot, x_base = x.split(cos.shape[-1], dim=-1)
    # Apply rotary embeddings to the rotary part
    x_rot = (x_rot * cos) + (rotate_half(x_rot) * sin)
    # Concatenate the rotary-applied and base parts
    return torch.cat([x_rot, x_base], dim=-1)

class MemoryOptimizedMLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_head = config.d_model // config.n_heads
        self.split_dim = self.d_head - config.d_rope

        # Projections
        self.W_dkv = nn.Linear(config.d_model, config.d_kv_comp)
        self.W_dq = nn.Linear(config.d_model, config.d_kv_comp)

        self.W_uk = nn.Linear(config.d_kv_comp, config.n_heads * self.split_dim)
        self.W_uv = nn.Linear(config.d_kv_comp, config.n_heads * self.d_head)
        self.W_uq = nn.Linear(config.d_kv_comp, config.n_heads * self.split_dim)

        self.W_qr = nn.Linear(config.d_kv_comp, config.n_heads * config.d_rope)
        self.W_kr = nn.Linear(config.d_model, config.n_heads * config.d_rope)

        self.rotary = RotaryEmbedding(config.d_rope)
        self.output = nn.Linear(config.n_heads * self.d_head, config.d_model)

    def forward(self, h, past_kv=None):
        batch_size, seq_len, _ = h.shape

        # KV Compression
        c_kv = self.W_dkv(h)
        k = self.W_uk(c_kv).view(batch_size, seq_len, config.n_heads, self.split_dim)
        v = self.W_uv(c_kv).view(batch_size, seq_len, config.n_heads, self.d_head)

        # Query Compression
        c_q = self.W_dq(h)
        q_base = self.W_uq(c_q).view(batch_size, seq_len, config.n_heads, self.split_dim)
        q_rot = self.W_qr(c_q).view(batch_size, seq_len, config.n_heads, config.d_rope)

        # Rotary embeddings with proper dimensions
        rotary_emb = self.rotary(seq_len)
        cos = torch.cos(rotary_emb).view(1, seq_len, 1, -1)  # [1, seq, 1, dim]
        sin = torch.sin(rotary_emb).view(1, seq_len, 1, -1)

        # Apply rotary embeddings
        q_rot = apply_rotary(q_rot, cos, sin)
        k_rot = apply_rotary(
            self.W_kr(h).view(batch_size, seq_len, config.n_heads, config.d_rope),
            cos, sin
        )

        q = torch.cat([q_base, q_rot], dim=-1)
        k = torch.cat([k, k_rot], dim=-1)

        # Attention computation
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("bhqk,bkhd->bqhd", attn, v)

        return self.output(out.contiguous().view(batch_size, seq_len, -1))

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, config, dim_feedforward=128, dropout=0.1, norm_first=True):
        super(CustomTransformerEncoderLayer, self).__init__()

        self.self_attn = MemoryOptimizedMLA()

        self.linear1 = nn.Linear(config.d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, config.d_model)

        self.norm_first = norm_first

        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        if self.norm_first:
            src = src + self._sa_block(self.norm1(src), mask)
            src = src + self._ff_block(self.norm2(src))
        else:
            src = self.norm1(src + self._sa_block(src, mask))
            src = self.norm2(src + self._ff_block(src))

        return src

    def _sa_block(self, x, mask):
        attn_output, _ = self.self_attn(x)  # Discarding kv cache for encoder-only
        return self.dropout1(attn_output)

    def _ff_block(self, x):
        x = self.linear2(F.relu(self.linear1(x)))
        return self.dropout2(x)


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
        '''self.encoder_layer = CustomTransformerEncoderLayer(config) # Change to 128 for less memory but slower convergence
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)'''
        self.attn = MemoryOptimizedMLA()
        self.fc = nn.Linear(d_model, 1)  # Final classification

    def forward(self, x):
        B, L = x.shape  # (Batch size, Sequence length)
        x = self.embedding(x)  # (B, L, d_model)

        # Apply sliding window
        outputs = []
        for start in range(0, L - self.window_size + 1, self.stride):
            window = x[:, start:start + self.window_size, :]  # Extract window
            window = window.permute(1, 0, 2)  # (Seq_len, Batch, d_model) for Transformer
            out = self.attn(window)  # Process window
            out += window
            out = out.mean(dim=0)  # Mean pooling across window
            outputs.append(out)

        # Stack and aggregate all windows
        outputs = torch.stack(outputs, dim=1)  # (B, num_windows, d_model)
        final_out = outputs.mean(dim=1)  # Aggregate over all windows

        return self.fc(final_out).squeeze()  # Final classification


def train_model(model, train_loader, val_loader, epochs=15, lr=1e-4):
    device = torch.device("cpu" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else
                          "cpu")
    print("device: ", device)
    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    accuracies = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        i = 0
        start_time = time.time()

        for snps, labels in train_loader:
            snps, labels = snps.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(snps)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            i += 1
            #print("Current Step: {} of {}, loss: {}".format(i, len(train_loader), loss.item()))

            total_loss += loss.item()
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(train_loader)}")
        accuracies.append(evaluate_model(model, val_loader))
    return accuracies




import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def evaluate_model(model, test_loader):
    device = torch.device("cpu" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else
                          "cpu")
    print("Using Device: {}".format(device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for snps, labels in test_loader:
            snps, labels = snps.to(device), labels.to(device)
            outputs = model(snps)  # Get model predictions
            preds = (outputs >= 0.5).float()  # Convert probabilities to 0 or 1

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

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
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset2, batch_size=32
                            , shuffle=False)

    model = SNPTransformer()
    accuracies = train_model(model, train_loader, val_loader)
    print("\nEvaluating model on test data...")
    evaluate_model(model, val_loader)

    return accuracies

def EvalScript(iterations, top_snps, logging_file):
    results = []
    iterations += 1
    for run_num in range(1, iterations):
        seed = random.randint(1, 1000)
        print(f"Running with seed: {seed}")
        accuracies = main(seed=seed, epochs=3, printStats=False, savePerf=True, top_snps=top_snps)
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
    EvalScript(5, "Data/TopSNPs/rf/top500_SNPs_rf_binary.txt", "Logging/Transformer/adsfadsf.json")
    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.2f} seconds")


