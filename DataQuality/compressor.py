import os
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

    return list1, list2, intersection_list


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
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            # Split the line into values using spaces as separators
            values = line.split()

            # Remove the first n values
            remaining_values = values[n:]

            # Join the remaining values and write them to the output file
            outfile.write(" ".join(remaining_values) + "\n")


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
    output_hd_exclude = "output_hd_exclude.raw"
    mast_lact1 = "mast_lact1.phen"

    # create intersection list of common cows
    return read_and_intersect(output_hd_exclude, mast_lact1)


# Filters mast_lact_1 by common cows, sorts by cow ID, removes cow ID
def clean_mastlact1():
    # Mast Lact1 versions
    mast_lact1 = "mast_lact1.phen"
    mast_lact1_temp = "mast_lact1_temp.txt"
    mast_lact1_sorted = "mast_lact1_sorted_herd.txt"

    common_cows = get_common_cows()

    print("filtering and sorting mast_lact1")
    filter_and_copy(mast_lact1, common_cows, mast_lact1_temp)
    sort_file_by_first_column(mast_lact1_temp)
    remove_first_n_values(mast_lact1_temp, mast_lact1_sorted, 2)

    # delete temp file
    delete_file(mast_lact1_temp)


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
    output_hd_exclude = "output_hd_exclude.raw"
    output_hd_exclude_cleaned = "output_hd_exclude_cleaned.txt"
    output_hd_exclude_binary = "output_hd_exclude_binary_herd.txt"

    common_cows = get_common_cows()

    # filter and sort output_hd_exclude
    print("filtering and sorting output_hd_exclude")
    filter_and_copy(output_hd_exclude, common_cows, output_hd_exclude_binary)
    sort_file_by_first_column(output_hd_exclude_binary)

    # remove first 6 cols from output_hd_exclude
    print("removing first 6 columns from output_hd_exclude")
    remove_first_n_values(output_hd_exclude_binary, output_hd_exclude_cleaned, 6)

    # compress output_hd_exclude to binary
    print("compressing output_hd_exclude to binary")
    to_binary(output_hd_exclude_cleaned, output_hd_exclude_binary)


# Gets binary file of top 200+ SNPs (w +/- 50 on either side)
def get_top_200_SNPs_expanded():
    # Top SNPs
    top_200_SNPs_expanded = "top_200_SNPs_expanded.txt"
    top_200_SNPs_expanded_binary = "top_200_SNPs_expanded_binary.txt"
    top_200_SNPs_indices = "top_200_SNPs_indices.txt"
    output_hd_exclude_cleaned = "output_hd_exclude_cleaned.raw"

    # Load important SNP indices
    important_SNPs_indices = load_1d_array_from_file(top_200_SNPs_indices)
    important_SNPs_expanded = expand_list_with_range(important_SNPs_indices, offset=50)

    # filter output_hd_exclude by significant SNPs
    print("filtering output_hd_exclude by significant SNPs")
    extract_specified_columns(
        output_hd_exclude_cleaned, top_200_SNPs_expanded, important_SNPs_expanded
    )

    # compress top_snps to binary
    print("compressing significant SNPs to binary")
    to_binary(top_200_SNPs_expanded, top_200_SNPs_expanded_binary)
