import pandas as pd
'''Use below line for a list of indicies'''
#listOfIndexes = []

'''Use below line for file containing indicies on newlines'''
listOfIndexes = pd.read_csv('SNPLists/SNPLists/indexes.txt', sep="\n", header=None).tolist()

# Load in header of 600k data file
raw = pd.read_csv("../Data/RawData/raw_data.raw", sep=" ", nrows=1, header=None)

# Get rid of everything besides SNP IDs in the dataframe, then convert to list
raw = raw.iloc[:, 6:].tolist()
selected_items = [raw[i] for i in listOfIndexes]

print(selected_items)
