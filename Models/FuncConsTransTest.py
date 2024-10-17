import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch import nn
import numpy as np

from DataQuality.funtional_consequences import *
from DataQuality.to_array import bit_reader

# Create variables
breed_herd_year = '../Data/BreedHerdYear/breed_herdxyear_lact1_sorted.txt'
top_4000_snps_binary = '../Data/TopSNPs/top_4000_SNPs_binary.txt'
phenotypes = '../Data/Phenotypes/phenotypes_sorted_herd.txt'
impact_scores = np.array(get_impact_scores()['impact_score'])

# Load data from files
herd = load_2d_array_from_file(breed_herd_year)
X = bit_reader(top_4000_snps_binary)
y = load_1d_array_from_file(phenotypes)

# Combine herd data with X
for rowX, rowH in zip(X, herd):
    rowX.extend(rowH)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

seed_value = 42

# Augment training data
X_train_aug = X_train.copy()
y_train_augmented = y_train.copy()
duplicate_and_insert(
    X_train, X_train_aug, y_train, y_train_augmented, 1, 16, seed=seed_value
)

# Augment testing data
X_test_aug = X_test.copy()
y_test_augmented = y_test.copy()
duplicate_and_insert(
    X_test, X_test_aug, y_test, y_test_augmented, 1, 16, seed=seed_value
)

# Map impact scores to indices
impact_score_mapping = {0: 0, 0.1: 1, 1: 2, 3: 3, 4: 4, 5: 5}
final_scores_idx = np.array([impact_score_mapping[score] for score in impact_scores])

X_train_augmented = []
X_test_augmented = []

for row in X_train_aug:
    snp_sequence = row  # SNP values
    impact_sequence = final_scores_idx[:len(row)].tolist()  # Impact scores
    impact_sequence += [1, 1]  # Append two 1s to the end of the impact scores
    X_train_augmented.append((snp_sequence, impact_sequence))

for row in X_test_aug:
    snp_sequence = row  # SNP values
    impact_sequence = final_scores_idx[:len(row)].tolist()  # Impact scores
    impact_sequence += [1, 1]  # Append two 1s to the end of the impact scores
    X_test_augmented.append((snp_sequence, impact_sequence))

# Custom Dataset for SNPs and Impact Scores
class GeneticDataset(Dataset):
    def __init__(self, snp_sequences, impact_sequences, labels, tokenizer, max_length=512):
        self.snp_sequences = snp_sequences
        self.impact_sequences = impact_sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.snp_sequences)

    def __getitem__(self, idx):
        # Tokenizing SNP sequence
        snp_sequence = " ".join(map(str, self.snp_sequences[idx]))  # Convert SNPs to string format
        snp_encoding = self.tokenizer(
            snp_sequence,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Tokenizing impact sequence
        impact_sequence = " ".join(map(str, self.impact_sequences[idx]))  # Convert impact scores to string format
        impact_encoding = self.tokenizer(
            impact_sequence,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Get the tensors from the encoding
        snp_item = {key: val.squeeze() for key, val in snp_encoding.items()}
        impact_item = {key: val.squeeze() for key, val in impact_encoding.items()}
        label = torch.tensor(self.labels[idx])

        # Combine the SNP and impact items
        item = {
            'snp_sequence': snp_item,
            'impact_sequence': impact_item,
            'labels': label
        }

        return item


import torch
from torch import nn
from transformers import BertModel


class CustomBERTModel(nn.Module):
    def __init__(self, num_snps, num_impact_scores, embedding_dim, hidden_dim, num_labels=2):
        super(CustomBERTModel, self).__init__()

        # Custom SNP and Impact Embedding layers
        self.snp_embedding = nn.Embedding(num_embeddings=num_snps, embedding_dim=embedding_dim)
        self.impact_embedding = nn.Embedding(num_embeddings=num_impact_scores, embedding_dim=embedding_dim)

        # BERT model
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # Dense layer for classification
        self.fc = nn.Linear(embedding_dim * 2 + self.bert.config.hidden_size, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_labels)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, snp_input_ids, impact_input_ids, bert_input_ids, attention_mask=None, token_type_ids=None):
        # Get SNP and Impact embeddings
        '''FIX THIS'''
        impact_input_ids = torch.clamp(impact_input_ids, min=0, max=self.impact_embedding.num_embeddings - 1)
        snp_embeds = self.snp_embedding(snp_input_ids)  # [batch_size, seq_len, embedding_dim]
        impact_embeds = self.impact_embedding(impact_input_ids)  # [batch_size, seq_len, embedding_dim]

        # Concatenate SNP and Impact embeddings along the last dimension
        combined_embeds = torch.cat((snp_embeds, impact_embeds), dim=-1)  # [batch_size, seq_len, 2 * embedding_dim]

        # Get the BERT embeddings
        bert_outputs = self.bert(
            input_ids=bert_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        bert_pooled_output = bert_outputs.pooler_output  # [batch_size, bert_hidden_size]

        # Combine SNP + Impact embeddings with BERT pooled output
        combined_with_bert = torch.cat((combined_embeds.mean(dim=1), bert_pooled_output),
                                       dim=-1)  # [batch_size, combined_dim + bert_hidden_size]

        # Pass through the dense layer and classifier
        hidden_output = self.fc(self.dropout(combined_with_bert))  # [batch_size, hidden_dim]
        logits = self.classifier(hidden_output)  # [batch_size, num_labels]

        return logits


# Define the tokenizer
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Prepare data loaders
train_dataset = GeneticDataset(
    [snp_seq for snp_seq, _ in X_train_augmented],
    [impact_seq for _, impact_seq in X_train_augmented],
    y_train_augmented,
    tokenizer=tokenizer,
)
test_dataset = GeneticDataset(
    [snp_seq for snp_seq, _ in X_test_augmented],
    [impact_seq for _, impact_seq in X_test_augmented],
    y_test_augmented,
    tokenizer=tokenizer,
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the BERT model for classification (or use a custom model)
model = CustomBERTModel(
    num_snps=len(tokenizer.vocab),  # Number of unique SNPs
    num_impact_scores=len(impact_score_mapping),  # Number of unique impact scores
    embedding_dim=64,  # Dimension of SNP and impact embeddings
    hidden_dim=128,  # Dimension of the hidden layer
    num_labels=2  # Binary classification
)
model.to(device)

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

'''# Learning rate scheduler
total_steps = len(train_loader) * 3  # Assuming 3 epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)'''

# Loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop with embedding logic
for epoch in range(3):
    model.train()
    total_loss = 0
    i = 0

    for batch in train_loader:
        optimizer.zero_grad()

        # Move inputs to device
        snp_seq = batch['snp_sequence']['input_ids'].to(device)  # SNP sequence
        impact_seq = batch['impact_sequence']['input_ids'].to(device)  # Impact sequence
        bert_seq = batch['snp_sequence']['input_ids'].to(device)  # BERT input sequence (same as SNP sequence in this case)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(snp_input_ids=snp_seq, impact_input_ids=impact_seq, bert_input_ids=bert_seq)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        i += 1
        print(f'Epoch: {epoch}, Batch {i}/{len(train_loader)}, Loss: {loss.item()}')

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch: {epoch}, Avg Training Loss: {avg_train_loss}")

    # Evaluation
    model.eval()
    preds = []
    true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            snp_seq = batch['snp_sequence']['input_ids'].to(device)
            impact_seq = batch['impact_sequence']['input_ids'].to(device)
            bert_seq = batch['snp_sequence']['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(snp_input_ids=snp_seq, impact_input_ids=impact_seq, bert_input_ids=bert_seq)
            _, predicted = torch.max(outputs, 1)

            preds.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, preds)
    print(f"Epoch: {epoch}, Test Accuracy: {accuracy}")

    report = classification_report(true_labels, preds, target_names=["No mastitis (Control)", "Mastitis Present (Case)"])
    print(report)

    conf_matrix = confusion_matrix(true_labels, preds)
    print(conf_matrix)
