import pandas as pd

df = pd.read_csv('../Data/RawData/raw_data.raw', delimiter=' ', header=0, nrows=1)

# Step 2: Load the SNP file and extract the first 500 SNP IDs
with open('../SNPLists/top4000SNPS.txt', 'r') as f:
    snp_ids = [line.split(' : ')[0] for line in f.readlines()[:500]]

# Step 3: Get the column indices matching the SNP IDs
column_indices = [df.columns.get_loc(snp_id) for snp_id in snp_ids if snp_id in df.columns]
column_indices.sort()

print(column_indices)
