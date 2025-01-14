from DataQuality.to_array import bit_reader
from Models.FeatureSelection.helper import read_numbers_from_file
from Models.FeatureSelection.chi_square_variance import calculate_feature_importance

# Read data
X = bit_reader("Data/output_hd_exclude_binary_herd.txt")
y = read_numbers_from_file("Data/Phenotypes/phenotypes_sorted.txt")

# Calculate feature importance and write top 10,000 to a file
calculate_feature_importance(X, y, output_file="top_10000_features.txt", top_n=10000)
