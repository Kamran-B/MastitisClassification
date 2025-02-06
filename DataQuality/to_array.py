import os

import numpy as np
from tqdm import tqdm

import os
import numpy as np
from tqdm import tqdm

def bit_reader_memory_efficient(input_file, chunk_size=65536):
    """
    Optimized bit reader that returns a 2D NumPy array with each row as a sequence.
    """
    try:
        file_size = os.path.getsize(input_file)
        output = []  # Use a list initially to collect data

        with open(input_file, "rb") as file, tqdm(
                total=file_size, unit="B", unit_scale=True, desc="Reading bits"
        ) as pbar:
            current_sequence = []  # Temporary list for current sequence

            while True:
                chunk = file.read(chunk_size)  # Read a chunk of binary data
                if not chunk:
                    break

                pbar.update(len(chunk))

                # Convert chunk into a NumPy array
                byte_array = np.frombuffer(chunk, dtype=np.uint8)

                # Extract the last two bits from each byte
                first_bits = (byte_array & 0b10) >> 1
                second_bits = byte_array & 0b01

                # Create encoded pairs of bits
                pairs = (first_bits << 1) | second_bits

                # Find transition indices (where pairs == 3)
                transition_indices = np.where(pairs == 3)[0]

                # Process sequences between transitions
                last_idx = 0
                for idx in transition_indices:
                    current_sequence.extend(pairs[last_idx:idx].tolist())  # Collect sequence
                    output.append(np.array(current_sequence, dtype=np.uint8))  # Append sequence as row
                    current_sequence = []  # Reset for next sequence
                    last_idx = idx + 1  # Move past transition

                # Append remaining pairs to the current sequence
                current_sequence.extend(pairs[last_idx:].tolist())

            # Append the last sequence if it's not empty
            if current_sequence:
                output.append(np.array(current_sequence, dtype=np.uint8))

        # Convert the list of sequences into a 2D NumPy array
        return np.vstack(output)  # Stack sequences as rows in a 2D array

    except FileNotFoundError:
        print(f"File not found: {input_file}")
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.array([], dtype=np.uint8)

def bit_reader(input_file, chunk_size=500000):
    """
    Optimized bit reader for faster processing of binary files.
    """
    try:
        file_size = os.path.getsize(input_file)
        output = []
        current_sequence = []

        with open(input_file, "rb") as file, tqdm(
                total=file_size, unit="B", unit_scale=True, desc="Reading bits"
        ) as pbar:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break

                pbar.update(len(chunk))

                # Process the chunk as a numpy array
                byte_array = np.frombuffer(chunk, dtype=np.uint8)

                # Decode pairs of bits: direct comparison for transitions
                first_bits = (byte_array & 0b10) >> 1
                second_bits = byte_array & 0b01

                # Combined representation for each pair of bits
                pairs = (first_bits << 1) | second_bits

                # Locate transitions (1, 1) represented as 3 in our encoding
                transition_indices = np.where(pairs == 3)[0]

                # Split the pairs at the transitions
                last_idx = 0
                for idx in transition_indices:
                    current_sequence.extend(pairs[last_idx:idx].tolist())
                    output.append(current_sequence)
                    current_sequence = []
                    last_idx = idx + 1

                # Append remaining pairs to the current sequence
                current_sequence.extend(pairs[last_idx:].tolist())

        # Add the last sequence if not empty
        if current_sequence:
            output.append(current_sequence)

        return output

    except FileNotFoundError:
        print(f"File not found: {input_file}")
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
