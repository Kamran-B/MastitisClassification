import random


def read_numbers_from_file(file_path):
    numbers = []
    try:
        with open(file_path, "r") as file:
            for line in file:
                # Convert each line to a number and append to the list
                numbers.append(int(line.strip()))
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return numbers


# Function to duplicate rows and insert at random positions
def duplicate_and_insert(
        original_list,
        target_list,
        original_target_labels,
        target_labels,
        label_value,
        num_duplicates,
        seed=None,
):
    random.seed(seed)
    for d in range(len(original_list)):
        if original_target_labels[d] == label_value:
            for j in range(num_duplicates):
                random_position = random.randint(0, len(target_list))
                target_list.insert(random_position, original_list[d].copy())
                target_labels.insert(random_position, label_value)
