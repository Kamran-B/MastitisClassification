import os
from tqdm import tqdm
import concurrent.futures


def read_and_intersect(file1_path, file2_path, file3_path):
    def read_first_column(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            return [line.split()[0] for line in tqdm(lines, desc=f'Reading {file_path}')]

    list1 = read_first_column(file1_path)
    list2 = read_first_column(file2_path)
    list3 = read_first_column(file3_path)

    intersection_list = list(set(list1) & set(list2) & set(list3))

    return list1, list2, list3, intersection_list


def filter_and_copy(file_path, intersection_list, output_path):
    with open(file_path, 'r') as input_file, open(output_path, 'w') as output_file:
        lines = input_file.readlines()
        for line in tqdm(lines, desc=f'Filtering {file_path} and copying to {output_path}'):
            first_value = line.split()[0]
            if first_value in intersection_list:
                output_file.write(line)


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


def remove_first_n_values(input_file, output_file, n):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Split the line into values using spaces as separators
            values = line.split()

            # Remove the first n values
            remaining_values = values[n:]

            # Join the remaining values and write them to the output file
            outfile.write(' '.join(remaining_values) + '\n')


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


def from_binary(input_file, output_file):
    try:
        with open(input_file, 'rb') as file:
            # Read the entire content of the binary file
            content = file.read()

            # Open the output file for writing in text mode
            with open(output_file, 'w') as output:
                # Iterate over each byte in the content
                for byte in content:
                    # Decode two bits at a time
                    first_bit = (byte & 0b10) >> 1
                    second_bit = byte & 0b01

                    # Process each pair of bits based on specified conditions
                    if first_bit == 0 and second_bit == 0:
                        # Write '0' to the output file
                        output.write('0')
                    elif first_bit == 0 and second_bit == 1:
                        # Write '1' to the output file
                        output.write('1')
                    elif first_bit == 1 and second_bit == 0:
                        # Write '2' to the output file
                        output.write('2')
                    elif first_bit == 1 and second_bit == 1:
                        # Write a newline character to the output file
                        output.write('\n')

    except FileNotFoundError:
        print(f"File not found: {input_file}")
    except Exception as e:
        print(f"An error occurred: {e}")


# File definition:
output_hd_exclude = 'output_hd_exclude.raw'
output_hd_exclude_temp = 'output_hd_exclude_temp.txt'
output_hd_exclude_binary = 'output_hd_exclude_binary_herd.txt'

mast_lact1 = 'mast_lact1.phen'
mast_lact1_temp = 'mast_lact1_temp.txt'
mast_lact1_sorted = 'mast_lact1_sorted_herd.txt'

herdxyear_lact1 = 'herdxyear_lact1.covar'
herdxyear_lact1_temp = 'herdxyear_lact1_temp1.txt'
herdxyear_lact1_sorted = 'herdxyear_lact1_sorted.txt'

# create intersection list of common cows
list1, list2, list3, intersection_list = read_and_intersect(output_hd_exclude, mast_lact1, herdxyear_lact1)

# filter and sort mast_lact1
filter_and_copy(mast_lact1, intersection_list, mast_lact1_temp)
sort_file_by_first_column(mast_lact1_temp)

# remove first 2 cols from mast_lact1
remove_first_n_values(mast_lact1_temp, mast_lact1_sorted, 2)

# filter and sort output_hd_exclude
filter_and_copy(output_hd_exclude, intersection_list, output_hd_exclude_binary)
sort_file_by_first_column(output_hd_exclude_binary)

print("removing first 6 columns")
# remove first 6 cols from output_hd_exclude
remove_first_n_values(output_hd_exclude_binary, output_hd_exclude_temp, 6)

filter_and_copy(herdxyear_lact1, intersection_list, herdxyear_lact1_temp)
sort_file_by_first_column(herdxyear_lact1_temp)

remove_first_n_values(herdxyear_lact1_temp, herdxyear_lact1_sorted, 2)

print("compressing to binary")
# compress output_hd_exclude to binary
to_binary(output_hd_exclude_temp, output_hd_exclude_binary)
