import os

import pandas as pd
from tqdm import tqdm
from DataQuality.funtional_consequences import load_1d_array_from_file


# Reads the first column from two input files, finds the intersection of values in those columns, and returns the lists and their intersection.
def read_and_intersect(file1_path, file2_path):
    def read_first_column(file_path):
        with open(file_path, "r") as file:
            lines = file.readlines()
            return [
                line.split()[0] for line in tqdm(lines, desc=f"Reading {file_path}")
            ]

    list1 = read_first_column(file1_path)
    list2 = read_first_column(file2_path)

    intersection_list = list(set(list1) & set(list2))

    return intersection_list


# Filters lines from a file based on whether the first value of each line is present in the provided intersection list, then copies the filtered lines to an output file.
def filter_and_copy(file_path, intersection_list, output_path):
    with open(file_path, "r") as input_file, open(output_path, "w") as output_file:
        lines = input_file.readlines()
        for line in tqdm(
                lines, desc=f"Filtering {file_path} and copying to {output_path}"
        ):
            first_value = line.split()[0]
            if first_value in intersection_list:
                output_file.write(line)


# Sorts the lines of a file based on the values in their first column.
def sort_file_by_first_column(file_path):
    with open(file_path, "r") as file:
        # Read lines from the file
        lines = file.readlines()

    # Sort lines based on the values in the first column
    sorted_lines = sorted(lines, key=lambda line: line.split()[0])

    with open(file_path, "w") as file:
        # Use tqdm to display a progress bar while writing the sorted lines back to the file
        for line in tqdm(sorted_lines, desc=f"Sorting and writing to {file_path}"):
            file.write(line)


# Removes the first n values from each line of the input file and writes the remaining values to an output file.
def remove_first_n_values(input_file, output_file, n):
    # Count the total number of lines in the input file
    with open(input_file, "r") as infile:
        total_lines = sum(1 for _ in infile)

    # Process the file and display progress
    with open(input_file, "r") as infile, open(output_file, "w") as outfile, tqdm(
        total=total_lines, desc="Processing lines", unit="line"
    ) as pbar:
        for line in infile:
            # Split the line into values using spaces as separators
            values = line.split()

            # Remove the first n values
            remaining_values = values[n:]

            # Join the remaining values and write them to the output file
            outfile.write(" ".join(remaining_values) + "\n")

            # Update progress bar
            pbar.update(1)


# Removes all values except for those specified by the list of indexes from each line of the input file
# and writes the remaining values to an output file.
def extract_specified_columns(input_file, output_file, columns_to_keep):
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            # Split the line into values using spaces as separators
            values = line.split()

            # Filter the values to keep only those specified by the list of indexes
            remaining_values = [values[i] for i in columns_to_keep]

            # Join the remaining values and write them to the output file
            outfile.write(" ".join(remaining_values) + "\n")


# Converts characters in a text file to binary and writes the binary representation to an output file with specified chunk size.
def to_binary(input_file, output_file, chunk_size=8192):
    try:
        with open(input_file, "r") as file:
            with open(output_file, "wb") as output:
                print("opened file")
                # Initialize tqdm for the entire file
                total_size = os.path.getsize(input_file)
                progress_bar = tqdm(
                    desc="Compressing file", unit="char", total=total_size
                )

                while True:
                    # Read a chunk of data from the input file
                    chunk = file.read(chunk_size)
                    if not chunk:
                        break  # Break if no more data to read

                    # Process each character in the chunk
                    compressed_chunk = bytearray()
                    for char in chunk:
                        if char == "0":
                            compressed_chunk.extend(bytes([0b00]))
                        elif char == "1":
                            compressed_chunk.extend(bytes([0b01]))
                        elif char == "2":
                            compressed_chunk.extend(bytes([0b10]))
                        elif char == "\n":
                            compressed_chunk.extend(bytes([0b11]))

                    # Write the compressed chunk to the output file
                    output.write(compressed_chunk)

                    # Update tqdm bar for each chunk processed
                    progress_bar.update(len(chunk))

                # Close tqdm bar after processing the entire file
                progress_bar.close()

    except FileNotFoundError:
        print(f"File not found: {input_file}")
    except Exception as e:
        print(f"An error occurred: {e}")


# Deletes file given filepath
def delete_file(filepath):
    # Check if the file exists before attempting to delete it
    if os.path.exists(filepath):
        # Delete the file
        os.remove(filepath)
        print(f"File '{filepath}' has been deleted.")
    else:
        print(f"File '{filepath}' does not exist.")


# Expands a list to include values within the offset range for every element
def expand_list_with_range(lst, offset):
    expanded_list = []
    for num in lst:
        # Add all numbers from num-offset to num+offset (inclusive)
        expanded_list.extend(range(num - offset, num + offset + 1))
    return expanded_list


def get_common_cows():
    output_hd_exclude = "Data/RawData/raw_data.raw"
    mast_lact1 = "Data/Phenotypes/phenotypes.phen"

    # create intersection list of common cows
    return read_and_intersect(output_hd_exclude, mast_lact1)


# Filters phenotypes by common cows, sorts by cow ID, removes cow ID
def clean_mastlact1():
    # Mast Lact1 versions
    phenotypes = "phenotypes.phen"
    phenotypes_temp = "phenotypes_temp.txt"
    phenotypes_sorted = "phenotypes_sorted_herd.txt"

    common_cows = get_common_cows()

    print("filtering and sorting phenotypes")
    filter_and_copy(phenotypes, common_cows, phenotypes_temp)
    sort_file_by_first_column(phenotypes_temp)
    remove_first_n_values(phenotypes_temp, phenotypes_sorted, 2)

    # delete temp file
    delete_file(phenotypes_temp)


# Filters herdxyear by common cows, sorts by cow ID, removes cow ID
def clean_heardxyear():
    # Heard x Year versions
    herdxyear_lact1 = "herdxyear_lact1.covar"
    herdxyear_lact1_temp = "herdxyear_lact1_temp.txt"
    herdxyear_lact1_sorted = "herdxyear_lact1_sorted.txt"

    common_cows = get_common_cows()

    print("filtering and sorting herdxyear_lact1")
    filter_and_copy(herdxyear_lact1, common_cows, herdxyear_lact1_temp)
    sort_file_by_first_column(herdxyear_lact1_temp)
    remove_first_n_values(herdxyear_lact1_temp, herdxyear_lact1_sorted, 2)

    # delete temp file
    delete_file(herdxyear_lact1_temp)


# Converts SNPs from raw_data to binary format
def clean_raw_data():
    # All SNPS versions:
    raw_data_exclude = "Data/RawData/raw_data.raw"
    raw_data_cleaned = "Data/RawData/raw_data_cleaned.txt"
    output_hd_exclude_binary = "Data/output_hd_exclude_binary_herd.txt"
    filtered_data = "Data/filtered_data"

    common_cows = get_common_cows()

    # filter and sort raw_data
    print("filtering and sorting raw_data_exclude")
    filter_and_copy(raw_data_exclude, common_cows, filtered_data)
    sort_file_by_first_column(filtered_data)

    # remove first 6 cols from raw_data
    print("removing first 6 columns from raw_data_exclude")
    remove_first_n_values(filtered_data, raw_data_cleaned, 6)

    # compress raw_data to binary
    print("compressing raw_data_exclude to binary")
    to_binary(raw_data_cleaned, output_hd_exclude_binary)


# Gets binary file of top 200+ SNPs (w +/- 50 on either side)
def get_top_200_SNPs_expanded():
    # Top SNPs
    top_200_SNPs_expanded = "top_200_SNPs_expanded.txt"
    top_200_SNPs_expanded_binary = "top_200_SNPs_expanded_binary.txt"
    top_200_SNPs_indices = "top_200_SNPs_indices.txt"
    raw_data_cleaned = "raw_data_cleaned.raw"

    # Load important SNP indices
    important_SNPs_indices = load_1d_array_from_file(top_200_SNPs_indices)
    important_SNPs_expanded = expand_list_with_range(important_SNPs_indices, offset=50)

    # filter raw_data by significant SNPs
    print("filtering raw_data by significant SNPs")
    extract_specified_columns(
        raw_data_cleaned, top_200_SNPs_expanded, important_SNPs_expanded
    )

    # compress top_snps to binary
    print("compressing significant SNPs to binary")
    to_binary(top_200_SNPs_expanded, top_200_SNPs_expanded_binary)


def get_top_SNPs_chi2():
    # Top SNPs
    top500_SNPs_chi2 = "top500_SNPs_chi2.txt"
    top500_SNPs_chi2_binary = "top500_SNPs_chi2_binary.txt"
    top500_SNPs_chi2_indices_file = "ranked_snps_chi_2.csv"
    raw_data_cleaned = "Data/RawData/raw_data_cleaned.txt"

    # Load the first 500 rows of the first column as a list
    top500_SNPs_chi2_indices = pd.read_csv(top500_SNPs_chi2_indices_file, nrows=500)["Feature"].tolist()

    # filter raw_data by significant SNPs
    print("filtering raw_data by significant SNPs")
    extract_specified_columns(
        raw_data_cleaned, top500_SNPs_chi2, top500_SNPs_chi2_indices
    )

    # compress top_snps to binary
    print("compressing significant SNPs to binary")
    to_binary(top500_SNPs_chi2, top500_SNPs_chi2_binary)

def get_top_SNPs_mi():
    # Top SNPs
    top500_SNPs_mi = "Data/TopSNPs/MutualInfo/top500_SNPs_mi.txt"
    top500_SNPs_mi_binary = "Data/TopSNPs/MutualInfo/top500_SNPs_mi_binary.txt"
    top500_SNPs_mi_indices_file = "Data/TopSNPs/MutualInfo/ranked_snps_MI.csv"
    raw_data_cleaned = "Data/RawData/raw_data_cleaned.txt"

    # Load the first 500 rows of the first column as a list
    top500_SNPs_mi_indices = (
        pd.read_csv(top500_SNPs_mi_indices_file, nrows=500)["SNP"]
        .str.replace("SNP_", "", regex=False)  # Remove "SNP_"
        .astype(int)  # Convert to integers
        .tolist()  # Convert to list
    )
    # filter raw_data by significant SNPs
    print("filtering raw_data by significant SNPs")
    extract_specified_columns(
        raw_data_cleaned, top500_SNPs_mi, top500_SNPs_mi_indices
    )

    # compress top_snps to binary
    print("compressing significant SNPs to binary")
    to_binary(top500_SNPs_mi, top500_SNPs_mi_binary)

def get_top_SNPs_pca():
    # Top SNPs
    top500_SNPs_pca = "Data/TopSNPs/PCA/top500_SNPs_pca.txt"
    top500_SNPs_pca_binary = "Data/TopSNPs/PCA/top500_SNPs_pca_binary.txt"
    top500_SNPs_pca_indices_file = "Data/TopSNPs/PCA/ranked_snps_pca.csv"
    raw_data_cleaned = "Data/RawData/raw_data_cleaned.txt"

    # Load the first 500 rows of the first column as a list
    top500_SNPs_pca_indices = (
        pd.read_csv(top500_SNPs_pca_indices_file, nrows=500)["SNP"]
        .str.replace("SNP_", "", regex=False)  # Remove "SNP_"
        .astype(int)  # Convert to integers
        .tolist()  # Convert to list
    )
    # filter raw_data by significant SNPs
    print("filtering raw_data by significant SNPs")
    extract_specified_columns(
        raw_data_cleaned, top500_SNPs_pca, top500_SNPs_pca_indices
    )

    # compress top_snps to binary
    print("compressing significant SNPs to binary")
    to_binary(top500_SNPs_pca, top500_SNPs_pca_binary)

#clean_raw_data()
#get_top_SNPs_chi2()
#get_top_SNPs_mi()
get_top_SNPs_pca()