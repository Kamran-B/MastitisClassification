import os
import pandas as pd
import numpy as np
import torch
import sys
import random
import json
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from Models.SCC.helper import split_by_id_ratio_1_to_4

# ========= Tee class to log both to console and file =========
class Logger:
    def __init__(self, filename):
        self.terminal = sys.__stdout__
        self.log = open(filename, "w")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ========= Seed setup =========
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ========== Config ==========
DATA_PATH = "Data/Dairycomp_data/balanced_output.csv"
FEATURES = ["gap", "LGSCC", "DIM", "FCM", "305ME"]

# ========== Load and Preprocess Data ==========
def collate_fn(batch):
    sequences, targets, lengths = zip(*batch)
    return pad_sequence(sequences, batch_first=True), torch.tensor(targets, dtype=torch.float32), torch.tensor(lengths)

def extract_sequences(df):
    sequences, labels = [], []
    max_steps = max(int(col.split("_")[1]) for col in df.columns if any(col.startswith(f) for f in FEATURES))
    for _, row in df.iterrows():
        steps = []
        for j in range(1, max_steps + 1):
            entry = [row.get(f"{f}_{j}", np.nan) for f in FEATURES]
            if all(pd.notna(x) for x in entry):
                steps.append(entry)
        if len(steps) >= 2:
            seq = torch.tensor(steps[:-1], dtype=torch.float32)
            label = float(row["label"])
            sequences.append(seq)
            labels.append(label)
    return sequences, labels

def standardize(data, scaler=None, fit=True):
    flat = torch.cat(data).numpy()
    scaler = scaler or StandardScaler()
    scaled = scaler.fit_transform(flat) if fit else scaler.transform(flat)
    out, i = [], 0
    for seq in data:
        out.append(torch.tensor(scaled[i:i+len(seq)], dtype=torch.float32))
        i += len(seq)
    return out, scaler

df = pd.read_csv(DATA_PATH)
df_train, df_val = split_by_id_ratio_1_to_4(df)
raw_train, train_targets = extract_sequences(df_train)
raw_val, val_targets = extract_sequences(df_val)
scaled_train, scaler = standardize(raw_train, fit=True)
scaled_val, _ = standardize(raw_val, scaler, fit=False)

class LGSCCDataset(Dataset):
    def __init__(self, sequences, labels):
        self.data = [(seq, label, len(seq)) for seq, label in zip(sequences, labels)]
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

train_loader_base = DataLoader(LGSCCDataset(scaled_train, train_targets), shuffle=True, collate_fn=collate_fn)
val_loader_base = DataLoader(LGSCCDataset(scaled_val, val_targets), collate_fn=collate_fn)

# ========== Model ==========
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2], pe[0, :, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, layers, dropout):
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        self.pos = PositionalEncoding(d_model)
        encoder = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=layers)
        self.fc = nn.Linear(d_model, 1)
    def forward(self, x, lengths):
        x = self.pos(self.proj(x))
        mask = torch.zeros(x.size(0), x.size(1), dtype=torch.bool, device=x.device)
        for i, l in enumerate(lengths): mask[i, l:] = True
        x = self.transformer(x, src_key_padding_mask=mask)
        return self.fc(torch.stack([x[i, l - 1] for i, l in enumerate(lengths)])).squeeze()

# ========== Training ==========
def train_transformer_model(
    input_size=5,
    d_model=32,
    nhead=2,
    layers=2,
    dropout=0.3,
    lr=0.0005,
    batch_size=128,
    seed=12,
    max_epochs=100,
    patience=5,
    verbose=True
):
    set_seed(seed)

    train_loader = DataLoader(train_loader_base.dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_loader_base.dataset, batch_size=batch_size, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel(input_size, d_model, nhead, layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # artificially increase penalty for missing a positive (1)
    recall_priority_weight = 2.0  # try 1.5–3.0 for stronger recall emphasis
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([recall_priority_weight], device=device))

    best_loss, no_improve = float("inf"), 0
    for epoch in range(max_epochs):
        model.train(); total = 0
        for xb, yb, lengths in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb, lengths), yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item()

        model.eval(); val_total = 0
        with torch.no_grad():
            for xb, yb, lengths in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_total += criterion(model(xb, lengths), yb).item()

        if verbose:
            print(f"Epoch {epoch+1}: Train Loss = {total:.4f} | Val Loss = {val_total:.4f}")
        if val_total < best_loss:
            best_loss, no_improve = val_total, 0
            best_model = model.state_dict()
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_model)

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb, lengths in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            probs = torch.sigmoid(model(xb, lengths))
            preds = (probs > 0.5).int().cpu().tolist()
            all_preds.extend(preds)
            all_true.extend(yb.int().cpu().tolist())

    print("\nClassification Report:")
    report = classification_report(all_true, all_preds, digits=4)
    print(report)

    # save confusion matrix as png
    cm = confusion_matrix(all_true, all_preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=["No Mastitis", "Subclinical Mastitis"],
        yticklabels=["No Mastitis", "Subclinical Mastitis"]
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=600)
    plt.show()

    accuracy = accuracy_score(all_true, all_preds)
    params = {
        "d_model": d_model, "nhead": nhead, "layers": layers, "dropout": dropout,
        "lr": lr, "batch_size": batch_size, "seed": seed
    }

    return accuracy, report, params

# ========== Run Seed 12 Only ==========
accuracy, report, params = train_transformer_model(seed=12)
print(f"\n✅ Done. Accuracy with seed 12: {accuracy:.4f}")
