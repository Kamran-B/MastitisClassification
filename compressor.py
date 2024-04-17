import os
from tqdm import tqdm
import concurrent.futures


# Reads the first column from three input files, finds the intersection of values in those columns, and returns the lists and their intersection.
def read_and_intersect(file1_path, file2_path):
    def read_first_column(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            return [line.split()[0] for line in tqdm(lines, desc=f'Reading {file_path}')]

    list1 = read_first_column(file1_path)
    list2 = read_first_column(file2_path)

    intersection_list = list(set(list1) & set(list2))

    return list1, list2, intersection_list


# Filters lines from a file based on whether the first value of each line is present in the provided intersection list, then copies the filtered lines to an output file.
def filter_and_copy(file_path, intersection_list, output_path):
    with open(file_path, 'r') as input_file, open(output_path, 'w') as output_file:
        lines = input_file.readlines()
        for line in tqdm(lines, desc=f'Filtering {file_path} and copying to {output_path}'):
            first_value = line.split()[0]
            if first_value in intersection_list:
                output_file.write(line)


# Sorts the lines of a file based on the values in their first column.
def sort_file_by_first_column(file_path):
    with open(file_path, 'r') as file:
        # Read lines from the file
        lines = file.readlines()

    # Sort lines based on the values in the first column
    sorted_lines = sorted(lines, key=lambda line: line.split()[0])

    with open(file_path, 'w') as file:
        # Use tqdm to display a progress bar while writing the sorted lines back to the file
        for line in tqdm(sorted_lines, desc=f'Sorting and writing to {file_path}'):
            file.write(line)


# Removes the first n values from each line of the input file and writes the remaining values to an output file.
def remove_first_n_values(input_file, output_file, n):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Split the line into values using spaces as separators
            values = line.split()

            # Remove the first n values
            remaining_values = values[n:]

            # Join the remaining values and write them to the output file
            outfile.write(' '.join(remaining_values) + '\n')


# Removes all values except for those specified by the list of indexes from each line of the input file
# and writes the remaining values to an output file.
def extract_specified_columns(input_file, output_file, columns_to_keep):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Split the line into values using spaces as separators
            values = line.split()

            # Filter the values to keep only those specified by the list of indexes
            remaining_values = [values[i] for i in columns_to_keep]

            # Join the remaining values and write them to the output file
            outfile.write(' '.join(remaining_values) + '\n')


# Converts characters in a text file to binary and writes the binary representation to an output file with specified chunk size.
def to_binary(input_file, output_file, chunk_size=8192):
    try:
        with open(input_file, 'r') as file:
            with open(output_file, 'wb') as output:
                # Initialize tqdm for the entire file
                total_size = os.path.getsize(input_file)
                progress_bar = tqdm(desc="Compressing file", unit="char", total=total_size)

                while True:
                    # Read a chunk of data from the input file
                    chunk = file.read(chunk_size)
                    if not chunk:
                        break  # Break if no more data to read

                    # Process each character in the chunk
                    compressed_chunk = bytearray()
                    for char in chunk:
                        if char == '0':
                            compressed_chunk.extend(bytes([0b00]))
                        elif char == '1':
                            compressed_chunk.extend(bytes([0b01]))
                        elif char == '2':
                            compressed_chunk.extend(bytes([0b10]))
                        elif char == '\n':
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


# File definition:
output_hd_exclude = 'output_hd_exclude.raw'
output_hd_exclude_cleaned = 'output_hd_exclude_cleaned.txt'
output_hd_exclude_binary = 'output_hd_exclude_binary_herd.txt'
output_hd_exclude_top_SNPs = 'output_hd_exclude_top_SNPs.txt'
output_hd_exclude_top_SNPs_binary = 'output_hd_exclude_top_SNPs_binary.txt'

mast_lact1 = 'mast_lact1.phen'
mast_lact1_temp = 'mast_lact1_temp.txt'
mast_lact1_sorted = 'mast_lact1_sorted_herd.txt'

herdxyear_lact1 = 'herdxyear_lact1.covar'
herdxyear_lact1_temp = 'herdxyear_lact1_temp.txt'
herdxyear_lact1_sorted = 'herdxyear_lact1_sorted.txt'

important_SNPs = [1416, 5636, 5823, 7935, 7940, 19157, 20876, 20877, 20884, 20888, 27407, 33710, 45415, 45416, 45420, 46205, 61402, 62252, 62283, 66036, 68806, 69607, 69633, 69637, 69642, 69643, 73359, 73360, 73363, 96064, 96067, 96072, 111267, 116968, 142154, 142234, 142982, 143776, 143889, 153588, 153596, 153604, 153612, 159306, 160302, 185257, 187212, 187213, 187214, 187228, 187230, 187345, 187346, 187347, 187348, 187350, 187352, 187353, 187360, 204015, 211687, 215340, 223904, 224028, 224112, 238523, 238526, 238527, 238531, 238538, 243444, 245187, 256286, 262054, 262056, 264309, 264310, 268836, 268852, 268969, 268972, 268973, 277906, 278951, 283784, 283785, 284634, 304160, 304467, 319642, 319743, 319745, 347018, 355488, 361602, 369271, 373403, 373545, 377119, 388236, 393374, 398409, 399836, 403564, 414215, 417407, 418433, 418465, 418724, 418779, 418780, 418782, 418787, 418788, 418789, 418791, 418793, 418794, 418795, 418798, 418806, 422601, 424457, 425749, 438631, 438632, 438955, 438956, 440083, 440592, 440593, 440594, 440632, 440666, 446295, 446297, 456731, 459311, 462910, 462911, 466373, 466600, 482328, 494527, 494542, 498202, 498206, 498207, 498208, 498214, 498215, 498217, 498219, 498220, 498225, 498226, 498227, 506192, 506351, 507818, 507949, 508687, 511715, 515701, 517482, 522448, 522450, 522453, 522456, 522749, 524952, 537378, 537379, 537402, 547450, 572088, 572096, 572098, 572405, 583635, 585704, 594478, 600551, 600776, 601502, 601505, 601509, 601521, 601524, 601529, 601530, 601580, 601608, 601609, 602436, 602454, 605129, 615474, 615482, 620200, 620201, 620219]

# create intersection list of common cows
list1, list2, intersection_list = read_and_intersect(output_hd_exclude, mast_lact1)

# filter and sort mast_lact1
print("filtering and sorting mast_lact1")
filter_and_copy(mast_lact1, intersection_list, mast_lact1_temp)
sort_file_by_first_column(mast_lact1_temp)
remove_first_n_values(mast_lact1_temp, mast_lact1_sorted, 2)

# filter and sort herdxyear_lact1
print("filtering and sorting herdxyear_lact1")
filter_and_copy(herdxyear_lact1, intersection_list, herdxyear_lact1_temp)
sort_file_by_first_column(herdxyear_lact1_temp)
remove_first_n_values(herdxyear_lact1_temp, herdxyear_lact1_sorted, 2)

# filter and sort output_hd_exclude
print("filtering and sorting output_hd_exclude")
filter_and_copy(output_hd_exclude, intersection_list, output_hd_exclude_binary)
sort_file_by_first_column(output_hd_exclude_binary)

# remove first 6 cols from output_hd_exclude
print("removing first 6 columns from output_hd_exclude")
remove_first_n_values(output_hd_exclude_binary, output_hd_exclude_cleaned, 6)

# filter output_hd_exclude by significant SNPs
print("filtering output_hd_exclude by significant SNPs")
extract_specified_columns(output_hd_exclude_cleaned, output_hd_exclude_top_SNPs, important_SNPs)

# compress output_hd_exclude to binary
print("compressing output_hd_exclude to binary")
to_binary(output_hd_exclude_cleaned, output_hd_exclude_binary)

# compress output_hd_exclude_top_SNPs to binary
print("compressing output_hd_exclude_top_SNPs to binary")
to_binary(output_hd_exclude_top_SNPs, output_hd_exclude_top_SNPs_binary)

# delete temp files
print("deleting mast_lact1_temp & herdxyear_lact1_temp")
delete_file(mast_lact1_temp)
delete_file(herdxyear_lact1_temp)