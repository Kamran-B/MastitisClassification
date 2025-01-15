import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from DataQuality.to_array import bit_reader


# Function to perform PCA and rank SNPs by their contribution to the principal components
def rank_snps_by_pca(X, n_components=2, snp_names=None, output_file='ranked_snps_pca.csv'):
    """
    Perform PCA on the SNP data and rank SNPs by their contribution to the principal components.
    The results are saved to a CSV file.

    Parameters:
    - X: numpy array or pandas DataFrame of shape (n_samples, n_snps)
    - n_components: number of principal components to consider
    - snp_names: list of SNP names (optional). If None, indices will be used.
    - output_file: the name of the CSV file where the ranked SNPs will be saved.

    Returns:
    - ranked_snps: DataFrame with SNP names and their importance in each principal component, sorted by contribution.
    """
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca.fit(X)

    # Get the loadings (coefficients) for each SNP in each principal component
    loadings = pca.components_.T  # Transpose to get SNP by component layout

    # Calculate the total contribution of each SNP across all components (sum of absolute loadings)
    total_contribution = np.sum(np.abs(loadings), axis=1)

    # Create SNP names if not provided
    if snp_names is None:
        snp_names = [f"SNP_{i}" for i in range(X.shape[1])]

    # Create a DataFrame for ranked SNPs
    ranked_snps = pd.DataFrame({"SNP": snp_names, "Total Contribution": total_contribution})
    ranked_snps = ranked_snps.sort_values(by="Total Contribution", ascending=False).reset_index(drop=True)

    # Write the ranked SNPs to a CSV file
    ranked_snps.to_csv(output_file, index=False)

    return ranked_snps


# Example usage
if __name__ == "__main__":
    # Example dataset
    np.random.seed(42)
    X = np.array(bit_reader("Data/output_hd_exclude_binary_herd.txt"))

    # Rank SNPs by their contribution to the principal components and write the results to a file
    ranked_snps = rank_snps_by_pca(X, n_components=2, output_file='ranked_snps_pca.csv')
    print(ranked_snps)
