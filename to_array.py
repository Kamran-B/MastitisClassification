import os
from tqdm import tqdm


def bit_reader(input_file, chunk_size=1024):
    """
    Reads a binary file and processes its bits in chunks.

    Args:
        input_file (str): Path to the input binary file.
        chunk_size (int, optional): Size of each data chunk to read from the file. Defaults to 1024.

    Returns:
        list: A list of lists representing the bits read from the file. Each sublist contains
        a sequence of bits read from the file, represented as integers (0, 1, or 2).
        The integers represent the bit pairs decoded as follows:
        - (0, 0): Represented as 0
        - (0, 1): Represented as 1
        - (1, 0): Represented as 2
        - (1, 1): Starts a new sequence in the output list

    Note:
        This function assumes the input file contains binary data.
        Each byte in the file is processed to extract pairs of bits.
        The progress of reading the file is displayed using tqdm.
    """
    arr = []
    current = []

    try:
        file_size = os.path.getsize(input_file)
        with open(input_file, 'rb') as file, tqdm(total=file_size, unit='B', unit_scale=True,
                                                  desc="Reading bits") as pbar:
            while True:
                # Read a chunk of data from the binary file
                chunk = file.read(chunk_size)
                if not chunk:
                    break  # Break if no more data to read

                pbar.update(len(chunk))  # Update progress bar

                # Process each byte in the chunk
                for byte in chunk:
                    first_bit = (byte & 0b10) >> 1
                    second_bit = byte & 0b01

                    if first_bit == 0 and second_bit == 0:
                        current.append(0)
                    elif first_bit == 0 and second_bit == 1:
                        current.append(1)
                    elif first_bit == 1 and second_bit == 0:
                        current.append(2)
                    elif first_bit == 1 and second_bit == 1:
                        arr.append(current)
                        current = []

        if len(current) != 0:
            arr.append(current)

    except FileNotFoundError:
        print(f"File not found: {input_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return arr
