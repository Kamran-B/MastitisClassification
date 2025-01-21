from torch.optim import AdamW
from torch.utils import checkpoint
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from transformers import BertModel, BertTokenizer, DistilBertModel, DistilBertTokenizer
from DataQuality.funtional_consequences import *
from DataQuality.to_array import bit_reader
from DataQuality.model_saving import *


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

TOP_PERFORMANCE_FILE = "top_performancesFuncCons.json"
TOP_K = 10
MODEL_SAVE_PATH = "../Data/Saved Models/saved_models_base_transformer"

def main(seed_value=42, epochs=4, printStats=True, savePerf=False):
    torch.cuda.manual_seed(seed_value)
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create directory for saving models if it doesn't exist
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)

        # Create variables
    breed_herd_year = 'Data/BreedHerdYear/breed_herdxyear_lact1_sorted.txt'
    top_4000_snps_binary = ('Data/output_hd_exclude_binary_herd.txt')
    phenotypes = 'Data/Phenotypes/phenotypes_sorted.txt'

    # Load data from files
    herd = load_2d_array_from_file(breed_herd_year)
    X = bit_reader(top_4000_snps_binary)
    y = load_1d_array_from_file(phenotypes)

    #X = X[:, top500].tolist()

    # Combine herd data with X
    for rowX, rowH in zip(X, herd):
        rowX.extend(rowH)


    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed_value
    )

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

    # Custom Dataset for SNPs and Impact Scores
    class GeneticDataset(Dataset):
        def __init__(self, snp_sequences, labels, tokenizer, max_length=500):
            self.snp_sequences = snp_sequences
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.snp_sequences)

        def __getitem__(self, idx):
            snp_sequence = self.snp_sequences[idx][:-2]
            breed = self.snp_sequences[idx][-2]
            herd_year = self.snp_sequences[idx][-1]

            # Tokenize SNP and impact sequences into chunks
            snp_chunks = [
                " ".join(map(str, snp_sequence[i:i + self.max_length]))
                for i in range(0, len(snp_sequence), self.max_length)
            ]

            label = torch.tensor(self.labels[idx])
            breed = torch.tensor(breed, dtype=torch.long)
            herd_year = torch.tensor(herd_year, dtype=torch.long)

            return {
                'snp_chunks': snp_chunks,
                'breed': breed,
                'herd_year': herd_year,
                'labels': label
            }

    class CustomBERTModel(nn.Module):
        def __init__(self, embedding_dim, hidden_dim, num_labels=2):
            super(CustomBERTModel, self).__init__()
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.max_length = 512
            # BERT model
            self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
            self.layer_norm = nn.LayerNorm(self.bert.config.hidden_size + 2 * embedding_dim)

            # Dense layer for classification
            self.classifier = nn.Linear(hidden_dim, num_labels)

            # Fully connected layer
            self.fc = nn.Linear(self.bert.config.hidden_size + 2 * embedding_dim, hidden_dim)

            # Dropout for regularization
            self.dropout = nn.Dropout(0.1)

            # Breed and herd year embeddings
            self.breed_embedding = nn.Embedding(num_embeddings=10, embedding_dim=embedding_dim)
            self.herd_year_embedding = nn.Embedding(num_embeddings=40, embedding_dim=embedding_dim)

        def forward(self, snp_chunks, breed_ids, herd_year_ids, attention=False):
            device = next(self.parameters()).device

            # Process SNP chunks in mini-batches
            snp_pooled_outputs = []
            attentions_list = []
            for chunk in snp_chunks:
                encodings = self.tokenizer(chunk, padding=True, truncation=False, max_length=self.max_length,
                                           return_tensors="pt")
                encodings = {k: v.to(device) for k, v in encodings.items()}

                # Forward pass through DistilBERT
                def checkpointed_bert(encodings):
                    return self.bert(**encodings, output_attentions=True)

                outputs = checkpoint.checkpoint(checkpointed_bert, encodings)
                attentions = outputs.attention

                # Use the [CLS] token embedding (first token) as pooled output
                cls_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_dim)
                snp_pooled_outputs.append(cls_output)
                del encodings, outputs, cls_output
                torch.cuda.empty_cache()

            # Average the pooled outputs from all chunks
            snp_pooled_avg = torch.mean(torch.stack(snp_pooled_outputs), dim=0)  # (batch_size, hidden_dim)

            # Embeddings for breed and herd year
            breed_embeds = self.breed_embedding(breed_ids)  # (batch_size, embedding_dim)
            herd_year_embeds = self.herd_year_embedding(herd_year_ids)  # (batch_size, embedding_dim)

            # Combine all features
            combined_features = torch.cat((snp_pooled_avg, breed_embeds, herd_year_embeds),
                                          dim=-1)  # (batch_size, combined_dim)
            combined_features = self.layer_norm(combined_features)
            hidden_output = self.fc(self.dropout(combined_features))  # (batch_size, hidden_dim)
            logits = self.classifier(hidden_output)  # (batch_size, num_labels)

            if attention:
                attentions_list.append(attentions)
                return logits, attentions_list
            else:
                return logits

    # Define the tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # Prepare data loaders
    train_dataset = GeneticDataset(
        [snp_seq for snp_seq in X_train_augmented],
        y_train_augmented,
        tokenizer=tokenizer,
    )
    test_dataset = GeneticDataset(
        [snp_seq for snp_seq in X_test_augmented],
        y_test_augmented,
        tokenizer=tokenizer,
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the BERT model for classification (or use a custom model)
    model = CustomBERTModel(
        embedding_dim=16,  # Dimension of SNP and impact embeddings
        hidden_dim=128,  # Dimension of the hidden layer
        num_labels=2  # Binary classification
    )
    model.to(device)

    # Define the optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)

    '''# Learning rate scheduler
    total_steps = len(train_loader) * epochs  # Assuming 3 epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )'''

    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    accuracies = []

    # Training loop with embedding logic
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        i = 0
        for batch in train_loader:
            optimizer.zero_grad()

            # SNP and impact chunks
            snp_chunks = batch['snp_chunks']
            breed_ids = batch['breed'].to(device)
            herd_year_ids = batch['herd_year'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(snp_chunks, breed_ids, herd_year_ids)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # Backward pass
            loss.backward()
            optimizer.step()

            i += 1
            if printStats:
                print(f'Epoch: {epoch}, Batch {i}/{len(train_loader)}, Loss: {loss.item()}')

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = total_correct / total_samples * 100  # Percentage accuracy
        if printStats:
            print(f"Epoch {epoch + 1}, Avg Train Loss: {avg_train_loss}, Train Accuracy: {train_accuracy}%")

        # Evaluation
        model.eval()
        preds = []
        true_labels = []
        x = 1
        with torch.no_grad():
            for batch in test_loader:
                snp_chunks = batch['snp_chunks']
                breed_ids = batch['breed'].to(device)
                herd_year_ids = batch['herd_year'].to(device)
                labels = batch['labels'].to(device)

                outputs, attentions_list = model(snp_chunks, breed_ids, herd_year_ids, attention=True)
                if x == 0:
                    print(analyze_cls_attentions(attentions_list))
                    x+=1

                _, predicted = torch.max(outputs, 1)

                preds.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(true_labels, preds)
        report = classification_report(true_labels, preds, target_names=["No mastitis (Control)", "Mastitis Present (Case)"])
        conf_matrix = confusion_matrix(true_labels, preds)

        if(printStats):
            print(f"Epoch: {epoch}, Test Accuracy: {accuracy}")
            print(report)
            print(conf_matrix)
        if(savePerf):
            model_name = f"model_epoch{epoch}_acc{accuracy:.4f}.pt"
            #torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, model_name))

            # Load current top performances and update
            top_performances = load_top_performances(TOP_PERFORMANCE_FILE)
            update_top_performances(top_performances, accuracy, model_name, report, TOP_K, MODEL_SAVE_PATH, TOP_PERFORMANCE_FILE)
        accuracies.append(accuracy)
    return accuracies

def analyze_cls_attentions(attentions_list):
    """
    Analyze CLS attentions to determine SNP importance based on their indices in the input sequence.

    Args:
        attentions_list (list): List of attention weights from each SNP chunk.

    Returns:
        list: List of SNP indices (in input order) with their corresponding importance scores.
    """
    cls_importances = []

    for attentions in attentions_list:  # Iterate through each chunk's attention
        for layer_attentions in attentions:  # Iterate over layers
            # Extract attention weights for the CLS token
            cls_attention = layer_attentions[:, :, 0, :]  # Attention for CLS token (batch_size, num_heads, seq_len)
            cls_importance = cls_attention.mean(dim=1).mean(dim=0)  # Average over heads and batch
            cls_importances.append(cls_importance)

    # Average over all layers and chunks
    cls_importances = torch.mean(torch.stack(cls_importances), dim=0)  # Shape: [seq_len]

    # Generate list of SNP indices and their importance scores
    snp_indices = list(range(len(cls_importances)))
    importance_scores = cls_importances.tolist()

    # Pair indices with their scores and sort by importance
    snp_importance_pairs = sorted(zip(snp_indices, importance_scores), key=lambda x: x[1], reverse=True)
    return snp_importance_pairs

if __name__=="__main__":
    main(52, 4, True, False)
