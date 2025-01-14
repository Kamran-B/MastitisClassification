from DataQuality.to_array import bit_reader
from helper import read_numbers_from_file, duplicate_and_insert
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


# Example function to compute mutual information and rank SNPs
def rank_snps_by_mutual_information(X, y, snp_names=None):
    """
    Ranks SNPs by mutual information with the target variable.

    Parameters:
    - X: numpy array or pandas DataFrame of shape (n_samples, n_snps)
    - y: numpy array or pandas Series of shape (n_samples,)
    - snp_names: list of SNP names (optional). If None, indices will be used.

    Returns:
    - ranked_snps: DataFrame with SNP names and their mutual information scores, sorted in descending order
    """
    # Compute mutual information scores
    mi_scores = mutual_info_classif(X, y, discrete_features=True, random_state=42)

    # Create SNP names if not provided
    if snp_names is None:
        snp_names = [f"SNP_{i}" for i in range(X.shape[1])]

    # Create a DataFrame for ranked SNPs
    ranked_snps = pd.DataFrame({"SNP": snp_names, "MI Score": mi_scores})
    ranked_snps = ranked_snps.sort_values(by="MI Score", ascending=False).reset_index(drop=True)

    return ranked_snps


# Example usage
if __name__ == "__main__":
    # Example dataset
    np.random.seed(42)
    X = np.array(bit_reader("Data/output_hd_exclude_binary_herd.txt"))
    y = np.array(read_numbers_from_file("Data/Phenotypes/phenotypes_sorted_herd.txt"))
    # Rank SNPs by mutual information
    ranked_snps = rank_snps_by_mutual_information(X, y)
    ranked_snps.to_csv("ranked_snps.csv", index=False)
    print(ranked_snps)
