import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer
from DataQuality.funtional_consequences import *
from DataQuality.to_array import bit_reader
from DataQuality.model_saving import *
from Models.TransformerBased.Classes import GeneticDataset, CustomBERTModel
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
TOP_PERFORMANCE_FILE = "top_performances.json"
MODEL_SAVE_PATH = "Data/Saved Models/saved_models_distil_transformer"

torch.cuda.empty_cache()

def get_top_500_snp_ids(file_path):
    df = pd.read_csv(file_path)
    return [int(snp_id) for snp_id in df.head(500)['SNP'].str.replace('SNP_', '').sort_values()]


def prepare_data(seed_value, top_snps):
    breed_herd_year = 'Data/BreedHerdYear/breed_herdxyear_lact1_sorted.txt'
    phenotypes = 'Data/Phenotypes/phenotypes_sorted.txt'

    herd = load_2d_array_from_file(breed_herd_year)
    X = bit_reader(top_snps)
    y = load_1d_array_from_file(phenotypes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed_value)

    return X_train, X_test, y_train, y_test


def augment_data(X, y, seed_value):
    X_aug, y_aug = X.copy(), y.copy()
    duplicate_and_insert(X, X_aug, y, y_aug, 1, 16, seed=seed_value)
    return X_aug, y_aug


def train_and_eval(model, device, train_loader, test_loader, optimizer, loss_fn, epochs, printStats, savePerf,
                   top_performances):
    accuracies = []
    for epoch in range(epochs):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(batch['snp_chunks'], batch['breed'].to(device), batch['herd_year'].to(device))
            loss = loss_fn(outputs, batch['labels'].to(device))
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == batch['labels'].to(device)).sum().item()
            total_samples += batch['labels'].size(0)

            loss.backward()
            optimizer.step()
            if printStats:
                print(f'Epoch: {epoch}, Batch {i}/{len(train_loader)}, Loss: {loss.item()}')

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Avg Loss: {avg_train_loss}")

        model.eval()
        preds, true_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                outputs = model(batch['snp_chunks'], batch['breed'].to(device), batch['herd_year'].to(device))
                _, predicted = torch.max(outputs, 1)
                preds.extend(predicted.cpu().numpy())
                true_labels.extend(batch['labels'].cpu().numpy())

        accuracy = accuracy_score(true_labels, preds)
        if printStats:
            print(f"Epoch: {epoch}, Test Accuracy: {accuracy}")
            print(classification_report(true_labels, preds, target_names=["No mastitis", "Mastitis Present"]))
            print(confusion_matrix(true_labels, preds))

        if savePerf:
            update_top_performances(top_performances, accuracy, f"model_epoch{epoch}_acc{accuracy:.4f}.pt",
                                    classification_report(true_labels, preds), 10, MODEL_SAVE_PATH,
                                    TOP_PERFORMANCE_FILE)

        accuracies.append(accuracy)
    return accuracies


def main(seed=42, epochs=4, printStats=True, savePerf=False, top_snps=None):
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)

    X_train, X_test, y_train, y_test = prepare_data(seed, top_snps)
    X_train_aug, y_train_aug = augment_data(X_train, y_train, seed)
    X_test_aug, y_test_aug = augment_data(X_test, y_test, seed)

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    train_loader = DataLoader(GeneticDataset(X_train_aug, y_train_aug, tokenizer), batch_size=8, shuffle=True)
    test_loader = DataLoader(GeneticDataset(X_test_aug, y_test_aug, tokenizer), batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomBERTModel(embedding_dim=16, hidden_dim=128, num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-6)
    loss_fn = torch.nn.CrossEntropyLoss()

    top_performances = load_top_performances(TOP_PERFORMANCE_FILE) if savePerf else None
    return train_and_eval(model, device, train_loader, test_loader, optimizer, loss_fn, epochs, printStats, savePerf,
                          top_performances)


def EvalScript(iterations, top_snps, logging_file):
    results = []
    iterations += 1
    for run_num in range(1, iterations):
        seed = random.randint(1, 1000)
        print(f"Running with seed: {seed}")
        accuracies = main(epochs=5, printStats=False, savePerf=True, top_snps=top_snps)
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
    EvalScript(5, "Data/TopSNPs/chi2/top500_SNPs_chi2_binary.txt", "Logging/Transformer/chi2_top500.json")
