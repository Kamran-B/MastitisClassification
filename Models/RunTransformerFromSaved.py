import os
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from DataQuality.funtional_consequences import *
from DataQuality.to_array import bit_reader

# Load the same tokenizer used during training
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define your dataset class (reusing the GeneticDataset class from training)
class GeneticDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=512):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = " ".join(map(str, self.sequences[idx]))
        label = self.labels[idx]
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze() for key, val in encoding.items()}
        item["labels"] = torch.tensor(label)
        return item

# Define a function to load and run the saved models
def load_and_evaluate_model(model_path, test_loader, device):
    # Load the model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    preds = []
    true_labels = []

    # Evaluate the model on the test set
    with torch.no_grad():
        for batch in test_loader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
            labels = batch["labels"].to(device)

            outputs = model(**inputs)
            _, predicted = torch.max(outputs.logits, 1)

            preds.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, preds)
    report = classification_report(true_labels, preds, target_names=["No mastitis", "Mastitis Present"])
    conf_matrix = confusion_matrix(true_labels, preds)

    return accuracy, report, conf_matrix

# Function to load the test data
def load_test_data():
    herd = load_2d_array_from_file("../Data/breed_herdxyear_lact1_sorted.txt")
    X = bit_reader("../Data/output_hd_exclude_4000top_SNPs_binary.txt")
    y = load_1d_array_from_file("../Data/mast_lact1_sorted_herd.txt")

    # Combine herd data with X
    for rowX, rowH in zip(X, herd):
        for value in rowH:
            rowX.append(value)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=52
    )

    test_dataset = GeneticDataset(X_test, y_test, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    return test_loader

# Main function to run the models
def main():
    # Define the directory where your saved models are stored
    model_dir = "saved_models"
    device = torch.device("cpu")

    # Load the test data
    test_loader = load_test_data()

    # Iterate over each saved model file and evaluate it
    for model_file in os.listdir(model_dir):
        if model_file.endswith(".pt"):
            model_path = os.path.join(model_dir, model_file)
            print(f"Evaluating model: {model_file}")

            # Run the model on the test data
            accuracy, report, conf_matrix = load_and_evaluate_model(model_path, test_loader, device)

            # Print the results
            print(f"Accuracy: {accuracy}")
            print("Classification Report:\n", report)
            print("Confusion Matrix:\n", conf_matrix)
            print("-" * 80)

if __name__ == "__main__":
    main()
