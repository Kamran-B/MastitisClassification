import os
import json


def scan_directory(directory_path):
    """
    Scans a directory and records its file and folder structure.

    Args:
        directory_path (str): The path of the directory to scan.

    Returns:
        dict: A dictionary representation of the directory structure with file sizes.
    """
    directory_structure = {}

    for root, _, files in os.walk(directory_path):
        relative_root = os.path.relpath(root, directory_path)
        directory_structure[relative_root] = {
            file: os.path.getsize(os.path.join(root, file)) for file in files
        }

    return directory_structure


def compare_directories(recorded_structure, current_directory):
    """
    Compares a recorded directory structure to the current state of a directory.

    Args:
        recorded_structure (dict): The recorded directory structure.
        current_directory (str): The path of the current directory to compare.

    Prints:
        Differences in missing, extra, or files of different sizes.
    """
    current_structure = scan_directory(current_directory)

    # Convert keys to sets for easy comparison
    recorded_dirs = set(recorded_structure.keys())
    current_dirs = set(current_structure.keys())

    # Compare directories
    missing_dirs = recorded_dirs - current_dirs
    extra_dirs = current_dirs - recorded_dirs

    for missing_dir in missing_dirs:
        print(f"Missing directory: {missing_dir}")

    for extra_dir in extra_dirs:
        print(f"Extra directory: {extra_dir}")

    # Compare files within directories
    for dir_key in recorded_dirs & current_dirs:
        recorded_files = recorded_structure[dir_key]
        current_files = current_structure[dir_key]

        recorded_files_set = set(recorded_files.keys())
        current_files_set = set(current_files.keys())

        missing_files = recorded_files_set - current_files_set
        extra_files = current_files_set - recorded_files_set
        common_files = recorded_files_set & current_files_set

        for missing_file in missing_files:
            print(f"Missing file: {os.path.join(dir_key, missing_file)}")

        for extra_file in extra_files:
            print(f"Extra file: {os.path.join(dir_key, extra_file)}")

        for common_file in common_files:
            recorded_size = recorded_files[common_file]
            current_size = current_files[common_file]

            if recorded_size != current_size:
                print(f"File size mismatch: {os.path.join(dir_key, common_file)}")
                print(f"  Recorded size: {recorded_size}")
                print(f"  Current size: {current_size}")


if __name__ == "__main__":

    # Step 2: Load the structure and compare
    with open("data_structure.json", "r") as f:
        recorded_structure = json.load(f)

    compare_directories(recorded_structure, "Data")
