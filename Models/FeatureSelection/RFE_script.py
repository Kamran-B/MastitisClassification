from DataQuality.to_array import bit_reader
from helper import read_numbers_from_file, duplicate_and_insert
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Example function to compute RFE rankings and scores for SNPs
def rank_snps_by_rfe(X, y, snp_names=None):
    """
    Ranks SNPs using Recursive Feature Elimination (RFE) with an estimator.

    Parameters:
    - X: numpy array or pandas DataFrame of shape (n_samples, n_snps)
    - y: numpy array or pandas Series of shape (n_samples,)
    - snp_names: list of SNP names (optional). If None, indices will be used.

    Returns:
    - ranked_snps: DataFrame with SNP names and their RFE rankings (lower is better), sorted by rank
    """
    # Create a Logistic Regression model for RFE
    estimator = LogisticRegression(max_iter=1000, random_state=42)
    rfe = RFE(estimator=estimator, n_features_to_select=1, step=1, verbose=2)

    # Fit RFE to the data
    rfe.fit(X, y)

    # Create SNP names if not provided
    if snp_names is None:
        snp_names = [f"SNP_{i}" for i in range(X.shape[1])]

    # Create a DataFrame for ranked SNPs
    ranked_snps = pd.DataFrame({
        "SNP": snp_names,
        "RFE Rank": rfe.ranking_,
        "Feature Importance": rfe.estimator_.coef_[0]
    })

    # Sort by RFE Rank
    ranked_snps = ranked_snps.sort_values(by="RFE Rank", ascending=True).reset_index(drop=True)

    return ranked_snps

# Example usage
if __name__ == "__main__":
    # Example dataset
    np.random.seed(42)
    X = bit_reader("Data/output_hd_exclude_binary_herd.txt")
    y = read_numbers_from_file("Data/Phenotypes/phenotypes_sorted_herd.txt")

    # Rank SNPs using RFE
    ranked_snps = rank_snps_by_rfe(X, y)

    # Save to CSV and print
    ranked_snps.to_csv("ranked_snps_RFE.csv", index=False)
    print(ranked_snps)
