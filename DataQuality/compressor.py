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
    top500_SNPs_chi2 = "Data/TopSNPs/chi2/top500_SNPs_chi2.txt"
    top500_SNPs_chi2_binary = "Data/TopSNPs/chi2/top500_SNPs_chi2_binary.txt"
    top500_SNPs_chi2_indices_file = "Data/TopSNPs/chi2/ranked_snps_chi_2.csv"
    raw_data_cleaned = "Data/RawData/raw_data_cleaned.txt"

    # Load the first 500 rows of the first column as a list
    top500_SNPs_chi2_indices = pd.read_csv(top500_SNPs_chi2_indices_file, nrows=500)["Feature"].tolist()

    with open("chi2_indices.txt", "w") as file:
        for item in top500_SNPs_chi2_indices:
            file.write(f"{item}\n")

    return

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
        #.str.replace("SNP_", "", regex=False)  # Remove "SNP_"
        #.astype(int)  # Convert to integers
        .tolist()  # Convert to list
    )

    with open("mi_indices.txt", "w") as file:
        for item in top500_SNPs_mi_indices:
            file.write(f"{item}\n")

    return

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
        #.str.replace("SNP_", "", regex=False)  # Remove "SNP_"
        #.astype(int)  # Convert to integers
        .tolist()  # Convert to list
    )

    with open("pca_indices.txt", "w") as file:
        for item in top500_SNPs_pca_indices:
            file.write(f"{item}\n")

    return

    # filter raw_data by significant SNPs
    print("filtering raw_data by significant SNPs")
    extract_specified_columns(
        raw_data_cleaned, top500_SNPs_pca, top500_SNPs_pca_indices
    )

    # compress top_snps to binary
    print("compressing significant SNPs to binary")
    to_binary(top500_SNPs_pca, top500_SNPs_pca_binary)

def get_top_SNPs_rf():
    # Top SNPs
    top500_SNPs_rf = "Data/TopSNPs/rf/top500_SNPs_og_rf.txt"
    top500_SNPs_rf_binary = "Data/TopSNPs/rf/top500_SNPs_og_rf_binary.txt"
    top500_SNPs_rf_indices_file = "Data/TopSNPs/rf/ranked_snps_rf.csv"
    raw_data_cleaned = "Data/RawData/raw_data_cleaned.txt"

    #Load the first 500 rows of the first column as a list
    # top500_SNPs_rf_indices = (
    #     pd.read_csv(top500_SNPs_rf_indices_file, nrows=500)["Feature"]
    #     #.str.replace("Feature ", "", regex=False)  # Remove "Feature "
    #     #.astype(int)  # Convert to integers
    #     .tolist()  # Convert to list
    # )

    top500_SNPs_rf_indices = [5644, 7933, 7940, 7941, 7946, 15844, 17877, 18596, 20882, 20883, 20890, 20894, 21858, 33716, 33718, 33721, 33723, 33726, 33727, 33728, 33730, 33731, 33732, 33736, 33738, 33739, 33741, 35620, 35621, 35629, 35650, 35651, 41577, 41579, 44742, 44756, 44758, 44772, 44783, 44788, 44822, 44949, 45422, 45426, 45428, 45429, 45430, 46112, 46115, 46116, 46118, 46119, 46120, 46324, 54092, 61387, 61408, 61983, 62258, 62271, 62283, 62289, 62295, 62315, 62327, 62329, 62330, 62331, 62332, 62333, 62334, 62335, 62997, 63033, 67976, 69160, 69161, 69613, 69639, 69643, 69648, 69649, 69652, 69660, 69661, 70136, 70224, 71606, 73365, 73366, 73369, 74727, 74728, 74729, 75080, 76633, 79720, 86682, 86685, 98381, 98383, 100112, 100344, 104869, 107181, 108697, 115826, 115878, 115914, 115939, 115943, 115957, 122685, 124119, 126818, 128703, 128706, 132583, 132591, 133620, 140433, 140735, 140753, 140754, 140755, 140759, 140761, 140769, 140781, 141950, 143784, 148441, 150997, 151000, 156829, 160073, 163324, 167077, 167089, 167106, 178468, 179001, 179003, 179248, 179464, 182420, 182442, 182444, 182800, 182803, 185178, 185258, 185259, 185263, 187201, 187220, 187344, 187351, 187353, 187354, 187356, 187357, 187358, 187359, 187360, 187366, 187543, 187545, 187546, 187547, 187548, 187549, 187550, 187553, 187554, 187555, 187556, 187558, 187717, 187718, 188468, 188472, 189766, 215189, 215200, 215203, 215345, 215346, 215351, 215353, 215354, 215363, 215548, 215549, 221534, 221561, 224233, 232625, 236386, 240599, 241953, 245187, 245911, 246828, 246893, 249614, 261354, 261355, 262060, 262062, 263021, 268975, 273664, 274515, 274519, 283861, 283862, 286799, 292549, 292550, 304155, 304162, 304166, 304167, 304559, 304560, 304588, 304619, 319751, 328363, 328365, 330607, 330612, 330621, 334841, 335704, 335707, 345480, 352791, 355494, 355495, 355497, 355584, 356327, 356834, 362275, 367883, 369002, 369015, 369025, 369277, 369303, 370165, 370166, 377125, 379094, 379513, 386717, 386719, 386726, 386728, 391572, 399213, 399842, 403570, 403571, 403572, 403574, 403580, 403581, 403696, 403697, 407399, 407406, 407407, 414221, 416666, 418446, 418785, 418786, 418787, 418788, 418789, 418790, 418791, 418792, 418793, 418794, 418795, 418796, 418797, 418798, 418799, 418800, 418801, 418802, 418803, 418804, 418806, 418807, 418808, 418809, 418810, 418811, 418812, 420712, 422465, 422466, 422467, 422607, 422838, 423965, 423969, 424929, 426176, 428387, 429741, 429742, 433879, 433880, 436283, 436448, 436453, 437214, 438637, 438638, 438649, 440598, 440599, 440600, 442146, 442148, 442149, 442150, 442167, 442168, 442186, 442188, 442190, 442191, 442192, 442196, 442198, 444889, 445410, 446215, 447434, 447444, 448882, 449944, 450296, 456737, 457012, 459105, 462382, 462916, 462917, 464695, 466379, 473215, 496956, 498186, 498189, 498206, 498207, 498208, 498210, 498211, 498212, 498213, 498214, 498215, 498216, 498217, 498218, 498219, 498220, 498221, 498223, 498224, 498225, 498226, 498227, 498228, 498229, 498230, 498231, 498232, 498233, 498236, 506200, 506205, 506206, 506207, 517032, 517033, 517034, 517035, 517036, 517038, 517039, 517040, 517042, 517043, 517044, 517045, 517046, 517047, 517048, 517049, 517050, 517052, 517477, 517612, 520555, 522454, 522455, 522456, 522457, 522458, 522459, 522460, 522462, 522466, 522755, 523663, 525973, 529849, 534169, 535411, 547456, 550210, 571275, 571277, 571278, 571279, 571538, 571657, 571658, 571660, 572077, 572078, 572080, 572081, 572094, 572102, 572104, 572115, 572143, 572395, 572411, 572415, 572418, 572422, 572424, 572427, 586465, 594484, 594973, 595050, 600782, 601508, 601511, 601515, 601518, 601527, 601529, 601530, 601531, 601532, 601534, 601535, 601536, 601581, 601585, 601586, 601587, 601614, 601615, 603129, 604250, 604292, 604295, 609046, 609049, 609052, 609496, 614845, 615476, 615482, 615485, 615487, 615488, 615491, 620171, 620206, 620207, 620225, 620231, 621160]

    with open("og_rf_indices.txt", "w") as file:
        for item in top500_SNPs_rf_indices:
            file.write(f"{item}\n")

    return

    # filter raw_data by significant SNPs
    print("filtering raw_data by significant SNPs")
    extract_specified_columns(
        raw_data_cleaned, top500_SNPs_rf, top500_SNPs_rf_indices
    )

    # compress top_snps to binary
    print("compressing significant SNPs to binary")
    to_binary(top500_SNPs_rf, top500_SNPs_rf_binary)

def get_top_SNPs_xgb():
    # Top SNPs
    top500_SNPs_xgb = "Data/TopSNPs/xgboost/top500_SNPs_xgb.txt"
    top500_SNPs_xgb_binary = "Data/TopSNPs/xgboost/top500_SNPs_xgb_binary.txt"
    top500_SNPs_xgb_indices_file = "Data/TopSNPs/xgboost/ranked_snps_xgb.csv"
    raw_data_cleaned = "Data/RawData/raw_data_cleaned.txt"

    # Load the first 500 rows of the first column as a list
    top500_SNPs_xgb_indices = (
        pd.read_csv(top500_SNPs_xgb_indices_file, nrows=500)["SNP_Index"]
        #.str.replace("SNP_", "", regex=False)  # Remove "SNP_"
        #.astype(int)  # Convert to integers
        .tolist()  # Convert to list
    )

    with open("xgb_indices.txt", "w") as file:
        for item in top500_SNPs_xgb_indices:
            file.write(f"{item}\n")


    # filter raw_data by significant SNPs
    print("filtering raw_data by significant SNPs")
    extract_specified_columns(
        raw_data_cleaned, top500_SNPs_xgb, top500_SNPs_xgb_indices
    )

    # compress top_snps to binary
    print("compressing significant SNPs to binary")
    to_binary(top500_SNPs_xgb, top500_SNPs_xgb_binary)

#clean_raw_data()
#get_top_SNPs_chi2()
#get_top_SNPs_mi()
#get_top_SNPs_pca()
get_top_SNPs_xgb()