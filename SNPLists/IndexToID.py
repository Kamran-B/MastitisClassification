import pandas as pd
'''Use below line for a list of indicies'''

df = pd.read_csv('./ranked_snps_MI.csv')  # Replace with actual file name

# Take the first 500 rows
df = df.head(500)

# Extract the index (the part after SNP_)
df['Index'] = df['SNP'].str.replace('SNP_', '')

# Convert to integers (optional, in case you want numeric indices)
df['Index'] = df['Index'].astype(int)

# Show the result
print(df[['SNP', 'Index']])

# If you just want the list of indices
index_list = df['Index'].tolist()
print(index_list)

listOfIndexes = index_list

'''Use below line for file containing indicies on newlines'''
'''listOfIndexes = pd.read_csv('SNPLists/SNPLists/indexes.txt', sep="\n", header=None).tolist()
'''
# Load in header of 600k data file
raw = pd.read_csv("../Data/RawData/raw_data.raw", sep=" ", nrows=1, header=None)

# Get rid of everything besides SNP IDs in the dataframe, then convert to list
raw = raw.iloc[:, 6:]
selected_items = [raw[i] for i in listOfIndexes]

print(selected_items)
