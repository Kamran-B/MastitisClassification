from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from transformers import BertModel, BertTokenizer, DistilBertTokenizer, DistilBertModel
from DataQuality.funtional_consequences import *
from DataQuality.to_array import bit_reader
from DataQuality.model_saving import *


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

TOP_PERFORMANCE_FILE = "top_performancesFuncCons.json"
TOP_K = 10
MODEL_SAVE_PATH = "../Data/Saved Models/saved_models_base_transformer"

def main(seed_value=42, epochs=4, printStats=True, savePerf=False):
    torch.cuda.empty_cache()

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
    breed_herd_year = '../Data/BreedHerdYear/breed_herdxyear_lact1_sorted.txt'
    top_4000_snps_binary = '../Data/TopSNPs/top_4000_SNPs_binary.txt'
    phenotypes = '../Data/Phenotypes/phenotypes_sorted_herd.txt'

    # Load data from files
    herd = load_2d_array_from_file(breed_herd_year)
    X = bit_reader(top_4000_snps_binary)
    y = load_1d_array_from_file(phenotypes)

    # Combine herd data with X
    for rowX, rowH in zip(X, herd):
        rowX.extend(rowH)


    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed_value
    )
    del X, y

    # Augment training data
    X_train_augmented = X_train.copy()
    y_train_augmented = y_train.copy()
    duplicate_and_insert(
        X_train, X_train_augmented, y_train, y_train_augmented, 1, 16, seed=seed_value
    )
    del X_train, y_train

    # Augment testing data
    X_test_augmented = X_test.copy()
    y_test_augmented = y_test.copy()
    duplicate_and_insert(
        X_test, X_test_augmented, y_test, y_test_augmented, 1, 16, seed=seed_value
    )
    del X_test, y_test

    # Custom Dataset for SNPs and Impact Scores
    class GeneticDataset(Dataset):
        def __init__(self, snp_sequences, labels, tokenizer, max_length=512):
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

            # Tokenize the entire SNP sequence (up to max_length tokens)
            encoding = self.tokenizer(
                " ".join(map(str, snp_sequence)),
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            label = torch.tensor(self.labels[idx], dtype=torch.long)
            breed = torch.tensor(breed, dtype=torch.long)
            herd_year = torch.tensor(herd_year, dtype=torch.long)

            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'breed': breed,
                'herd_year': herd_year,
                'labels': label
            }


    class CustomBERTModel(nn.Module):
        def __init__(self, embedding_dim, hidden_dim, num_labels=2):
            super(CustomBERTModel, self).__init__()
            self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

            # Dense layer for classification
            self.classifier = nn.Linear(hidden_dim, num_labels)

            # Fully connected layer
            self.fc = nn.Linear(self.bert.config.hidden_size + 2 * embedding_dim, hidden_dim)

            # Dropout for regularization
            self.dropout = nn.Dropout(0.1)

            # Breed and herd year embeddings
            self.breed_embedding = nn.Embedding(num_embeddings=10, embedding_dim=embedding_dim)
            self.herd_year_embedding = nn.Embedding(num_embeddings=40, embedding_dim=embedding_dim)

        def forward(self, input_ids, attention_mask, breed_ids, herd_year_ids):
            device = next(self.parameters()).device

            # Pass the input through BERT
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:, 0]  # CLS token output

            # Embeddings for breed and herd year
            breed_embeds = self.breed_embedding(breed_ids)
            herd_year_embeds = self.herd_year_embedding(herd_year_ids)

            # Combine all features
            combined_features = torch.cat((pooled_output, breed_embeds, herd_year_embeds), dim=-1)
            hidden_output = self.fc(self.dropout(combined_features))
            logits = self.classifier(hidden_output)

            return logits


    # Define the tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

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

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the BERT model for classification (or use a custom model)
    model = CustomBERTModel(
        embedding_dim=16,  # Dimension of SNP and impact embeddings
        hidden_dim=64,  # Dimension of the hidden layer
        num_labels=2  # Binary classification
    )
    model.to(device)

    # Define the optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scaler = torch.cuda.amp.GradScaler()  # Initialize gradient scaler for mixed precision

    '''# Learning rate scheduler
    total_steps = len(train_loader) * epochs  # Assuming 3 epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )'''

    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    accuracies = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        i=0

        for batch in train_loader:
            optimizer.zero_grad()

            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    breed_ids=batch['breed'].to(device),
                    herd_year_ids=batch['herd_year'].to(device)
                )
                loss = loss_fn(outputs, batch['labels'].to(device))

            total_loss += loss.item()

            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            i += 1
            if printStats:
                print(f'Epoch: {epoch}, Batch {i}/{len(train_loader)}, Loss: {loss.item()}')

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Avg Loss: {avg_train_loss}")

        # Evaluation (same as before)
        model.eval()
        preds = []
        true_labels = []

        with torch.no_grad():
            for batch in test_loader:
                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=batch['input_ids'].to(device),
                        attention_mask=batch['attention_mask'].to(device),
                        breed_ids=batch['breed'].to(device),
                        herd_year_ids=batch['herd_year'].to(device)
                    )
                    _, predicted = torch.max(outputs, 1)

                preds.extend(predicted.cpu().numpy())
                true_labels.extend(batch['labels'].cpu().numpy())

        accuracy = accuracy_score(true_labels, preds)
        report = classification_report(true_labels, preds, target_names=["No mastitis (Control)", "Mastitis Present (Case)"])
        conf_matrix = confusion_matrix(true_labels, preds)

        if printStats:
            print(f"Epoch: {epoch}, Test Accuracy: {accuracy}")
            print(report)
            print(conf_matrix)

        if savePerf:
            model_name = f"model_epoch{epoch}_acc{accuracy:.4f}.pt"
            # torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, model_name))
            top_performances = load_top_performances(TOP_PERFORMANCE_FILE)
            update_top_performances(top_performances, accuracy, model_name, report, TOP_K, MODEL_SAVE_PATH, TOP_PERFORMANCE_FILE)

    accuracies.append(accuracy)

if __name__=="__main__":
    main(422, 4, True, True)