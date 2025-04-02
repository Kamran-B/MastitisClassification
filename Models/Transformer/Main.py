import sys

from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray
from sklearn.model_selection import train_test_split
from Models.Transformer.Attention import Transformer
from DataQuality.funtional_consequences import *
from DataQuality.to_array import bit_reader
from DataQuality.model_saving import *

def prepare_data(seed_value, top_snps):
    breed_herd_year = 'Data/BreedHerdYear/breed_herdxyear_lact1_sorted.txt'
    phenotypes = 'Data/Phenotypes/phenotypes_sorted.txt'

    herd = load_2d_array_from_file(breed_herd_year)
    X = bit_reader(top_snps)
    y = load_1d_array_from_file(phenotypes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed_value, stratify=y)

    return X_train, X_test, y_train, y_test


def augment_data(X, y, seed_value):
    X_aug, y_aug = X.copy(), y.copy()
    duplicate_and_insert(X, X_aug, y, y_aug, 1, 16, seed=seed_value)
    return X_aug, y_aug


if __name__ == '__main__':
    top_snps = "Data/TopSNPs/rf/top500_SNPs_rf_binary.txt"
    seed = 50
    X_train, X_test, y_train, y_test = prepare_data(seed, top_snps)
    X_train_aug, y_train_aug = augment_data(X_train, y_train, seed)
    X_test_aug, y_test_aug = augment_data(X_test, y_test, seed)
    
    transformer = Transformer()
    print(X_train_aug.shape)
    transformer.forward(X_train_aug)