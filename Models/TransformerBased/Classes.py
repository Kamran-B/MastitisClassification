from torch.utils.data import Dataset
import torch
from torch import nn
from transformers import DistilBertModel
from transformers import DistilBertTokenizer


class GeneticDataset(Dataset):
    def __init__(self, snp_sequences, labels, tokenizer, max_length=505):
        self.snp_sequences = snp_sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.snp_sequences)

    def __getitem__(self, idx):
        snp_sequence = self.snp_sequences[idx]
        breed, herd_year = snp_sequence[-2], snp_sequence[-1]

        # Tokenize SNP and impact sequences into chunks (generator instead of list)
        snp_chunks = (
            " ".join(map(str, snp_sequence[i:i + self.max_length]))
            for i in range(0, len(snp_sequence), self.max_length)
        )

        return {
            'snp_chunks': list(snp_chunks),  # Convert generator to list at return time
            'breed': torch.tensor(breed, dtype=torch.int8),  # Use smaller dtype if applicable
            'herd_year': torch.tensor(herd_year, dtype=torch.int16),  # Use smaller dtype if applicable
            'labels': torch.tensor(self.labels[idx], dtype=torch.int8)  # Use smaller dtype if applicable
        }



class CustomBERTModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_labels=2):
        super(CustomBERTModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.max_length = 512

        self.layer_norm = nn.LayerNorm(self.bert.config.hidden_size + 2 * embedding_dim)

        # Dense layer for classification
        self.classifier = nn.Linear(hidden_dim, num_labels)

        # Fully connected layer
        self.fc = nn.Linear(self.bert.config.hidden_size, hidden_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

        # Breed and herd year embeddings
        self.breed_embedding = nn.Embedding(num_embeddings=10, embedding_dim=embedding_dim)
        self.herd_year_embedding = nn.Embedding(num_embeddings=40, embedding_dim=embedding_dim)

    def forward(self, snp_chunks, breed_ids, herd_year_ids):
        device = next(self.parameters()).device

        # Process SNP chunks in mini-batches

        snp_pooled_outputs = []

        for chunk in snp_chunks:
            encodings = self.tokenizer(chunk, padding=True, truncation=False, max_length=self.max_length,
                                       return_tensors="pt")
            encodings = {k: v.to(device) for k, v in encodings.items()}

            # Forward pass through DistilBERT
            outputs = self.bert(**encodings)  # (batch_size, max_length, hidden_dim)

            # Use the [CLS] token embedding (first token) as pooled output
            cls_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_dim)
            snp_pooled_outputs.append(cls_output)

        # Average the pooled outputs from all chunks
        snp_pooled_avg = torch.mean(torch.stack(snp_pooled_outputs), dim=0)  # (batch_size, hidden_dim)

        # Embeddings for breed and herd year
        breed_embeds = self.breed_embedding(breed_ids)  # (batch_size, embedding_dim)
        herd_year_embeds = self.herd_year_embedding(herd_year_ids)  # (batch_size, embedding_dim)

        # Combine all features
        # combined_features = torch.cat((snp_pooled_avg, breed_embeds, herd_year_embeds),
        #                               dim=-1)  # (batch_size, combined_dim)
        #combined_features = self.layer_norm(combined_features)
        hidden_output = self.fc(self.dropout(snp_pooled_avg))  # (batch_size, hidden_dim)
        logits = self.classifier(hidden_output)  # (batch_size, num_labels)

        return logits