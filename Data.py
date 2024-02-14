import csv

import pandas as pd

# File paths
raw_file1_path = '/Users/gavinlynch04/Desktop/CSC487_Project_Data/data/output_hd_exclude.raw'
raw_file2_path = '/Users/gavinlynch04/Desktop/CSC487_Project_Data/data/mast_lact1.phen'
output_csv_path = '/Users/gavinlynch04/Desktop/CSC487_Project_Data/data/merged.csv'

# Load the first raw file
file1 = '/Users/gavinlynch04/Desktop/CSC487_Project_Data/data/output_hd_exclude.raw'
with open(file1, 'r') as f:
    content1 = f.read().split('\n')[1:]  # Skip the header and split by newline
    data1 = {row.split()[0]: row.split()[1:] for row in content1 if row}

# Load the second raw file
file2 = '/Users/gavinlynch04/Desktop/CSC487_Project_Data/data/mast_lact1.phen'
with open(file2, 'r') as f:
    content2 = f.read().split('\n') # Split by newline
    data2 = {row.split()[0]: row.split()[1:] for row in content2 if row}

# Merge the two datasets based on the first column
merged_data = {}
for id_value, values1 in data1.items():
    if id_value in data2:
        merged_data[id_value] = values1 + data2[id_value][1:]

# Save the merged data to a CSV file
output_csv = '/Users/gavinlynch04/Desktop/CSC487_Project_Data/data/merged.csv'
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    for id_value, values in merged_data.items():
        writer.writerow([id_value] + values)

'''
# Load genomic data without headers
genomic_data_path = '/Users/gavinlynch04/Desktop/CSC487_Project_Data/data/output.csv'
genomic_df = pd.read_csv(genomic_data_path, header=0)

# Load disease labels without headers
disease_labels_path = '/Users/gavinlynch04/Desktop/CSC487_Project_Data/data/outputPhen.csv'
disease_df = pd.read_csv(disease_labels_path, header=None)

# Merge based on the values in the first column of each DataFrame
merged_df = pd.merge(genomic_df, disease_df, left_on=0, right_on=0)

# Save the merged DataFrame to a CSV file
merged_df.to_csv('/Users/gavinlynch04/Desktop/CSC487_Project_Data/data/mergedOutput.csv', index=False)

# Load the first raw file without headers
raw_file1_path = '/Users/gavinlynch04/Desktop/CSC487_Project_Data/data/output_hd_exclude.raw'
raw_df1 = pd.read_csv(raw_file1_path, header=0, delimiter=' ', lineterminator='\n')

# Load the second raw file without headers
raw_file2_path = '/Users/gavinlynch04/Desktop/CSC487_Project_Data/data/mast_lact1.phen'
raw_df2 = pd.read_csv(raw_file2_path, header=None, delimiter=' ', lineterminator='\n')

# Merge based on the values in the first column of each DataFrame
merged_df = pd.merge(raw_df1, raw_df2, left_on=0, right_on=0)

# Save the merged DataFrame to a CSV file
merged_df.to_csv('/Users/gavinlynch04/Desktop/CSC487_Project_Data/data/merged.csv', index=False)

# Display the first few rows of the merged DataFrame
print(merged_df.head())
'''