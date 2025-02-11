import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from helper import read_numbers_from_file
from DataQuality.to_array import bit_reader

# Load dataset
X = np.array(bit_reader("Data/output_hd_exclude_binary_herd.txt"))
y = np.array(read_numbers_from_file("Data/Phenotypes/phenotypes_sorted.txt"))

# Fit Lasso model
lasso = Lasso(alpha=0.0001)
lasso.fit(X, y)

# Get feature importance (absolute values)
importance = np.abs(lasso.coef_)

# Rank features by importance
feature_ranking = pd.DataFrame({
    "SNP_": np.arange(X.shape[1]),
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

# Save to file
feature_ranking.to_csv("LASSO_SNPs.csv", index=False)

print("Feature ranking saved as LASSO_SNPs.csv")
