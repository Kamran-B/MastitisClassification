import random
import numpy as np
import pandas as pd


def load_2d_array_from_file(file_path):
    """
    Reads a file where each line contains space-separated integers
    and returns a 2D NumPy array of these integers.

    Args:
        file_path (str): The path to the input file.

    Returns:
        np.ndarray: A 2D NumPy array with the integers from the file.
    """
    numbers = []
    try:
        with open(file_path, "r") as file:
            for line in file:
                # Split each line by spaces, convert to integers, and append as a row to the list
                row = list(map(int, line.strip().split()))
                numbers.append(row)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return np.array(numbers)


def load_1d_array_from_file(file_path):
    """
    Reads a file where each line contains a single integer
    and returns a list of these integers.

    Args:
        file_path (str): The path to the input file.

    Returns:
        list: A list of integers from the file.
    """
    numbers = []
    try:
        with open(file_path, "r") as file:
            for line in file:
                # Convert each line to an integer and append to the list
                numbers.append(int(line.strip()))
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return numbers


def extract_columns(file_path, columns, delimiter='|'):
    """
    Extracts specified columns from a file and returns them as a pandas DataFrame.

    Args:
        file_path (str): The path to the input file.
        columns (list): A list of column names to extract.
        delimiter (str): The delimiter used in the file. Default is a pipe '|'.

    Returns:
        pd.DataFrame: A DataFrame containing the specified columns.
    """
    # Read the file into a pandas DataFrame with the specified delimiter
    df = pd.read_csv(file_path, delimiter=delimiter)

    # Extract the specified columns by name
    result = df[columns]

    return result


def duplicate_and_insert(
        original_list,
        target_list,
        original_target_labels,
        target_labels,
        label_value,
        num_duplicates,
        seed=None,
):
    """
    Duplicates elements from the original_list and inserts them into the target_list
    at random positions. Also updates the target_labels with the specified label_value.

    Args:
        original_list (list): The list of elements to duplicate.
        target_list (list): The list where duplicated elements will be inserted.
        original_target_labels (list): Labels corresponding to the elements in the original_list.
        target_labels (list): Labels corresponding to the elements in the target_list.
        label_value (any): The label value to assign to the duplicated elements in target_labels.
        num_duplicates (int): The number of duplicates to create for each element with the specified label_value.
        seed (int, optional): Seed for the random number generator to ensure reproducibility.

    Returns:
        None
    """
    # Initialize the random number generator with the specified seed for reproducibility
    random.seed(seed)

    # Iterate over each element in the original_list
    for d in range(len(original_list)):
        # Check if the current element's label matches the specified label_value
        if original_target_labels[d] == label_value:
            # Duplicate the current element num_duplicates times
            for j in range(num_duplicates):
                # Generate a random position in the target_list to insert the duplicate
                random_position = random.randint(0, len(target_list))
                # Insert a copy of the current element into the target_list at the random position
                target_list.insert(random_position, original_list[d].copy())
                # Insert the label_value into the target_labels at the same random position
                target_labels.insert(random_position, label_value)


def left_join(left_df, right_df, join_col, fill_col, default_value=0):
    """
    Perform a left join between two DataFrames and fill missing values in the joined column with a default value.

    Args:
        left_df (pd.DataFrame): The left DataFrame to join.
        right_df (pd.DataFrame): The right DataFrame to join.
        join_col (str): The column to join on.
        fill_col (str): The column from the right DataFrame that might have missing values after the join.
        default_value (any): The value to use when the `fill_col` has missing values after the join.

    Returns:
        pd.DataFrame: The resulting DataFrame after the left join with filled missing values.
    """
    # Perform a left join on the specified column
    merged_df = pd.merge(left_df, right_df, on=join_col, how='left')

    # Fill missing values in the specified column with the default value
    merged_df[fill_col] = merged_df[fill_col].fillna(default_value)

    return merged_df


def get_impact_scores():
    snp_names = extract_columns("../Data/FunctionalConsequences/top4000SNPS.txt", ["SNP_name0"], "|")
    impact_scores = extract_columns("../Data/FunctionalConsequences/4000_top_SNP_functional_consequences_impact_score_1.csv",
                                    ['SNP_name0', 'impact_score'], "|")
    combined = left_join(snp_names, impact_scores, "SNP_name0", "impact_score", 0.1)
    return combined
