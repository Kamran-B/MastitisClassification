import pandas as pd

kevin = pd.read_excel('./mlma_Kevin.xlsx')
kevin = kevin.iloc[:, 0:2]
print(kevin)

raw = pd.read_csv("../Data/RawData/raw_data.raw", sep=" ", nrows=1, header=None)
'''def strip_suffix(value):
    if isinstance(value, str) and (value.endswith('_A') or value.endswith('_B')):
        return value[:-2]
    return value

raw = raw.applymap(strip_suffix)'''
raw = raw.iloc[:, 6:].reset_index(drop=True)


kevin_ids = kevin.iloc[:, 1]

raw_entries = raw.values.flatten()

raw_entries_cleaned = [entry[:-2] if entry.endswith('_A') or entry.endswith('_B') else entry for entry in raw_entries]
print(raw_entries_cleaned)
print(kevin)

entry_to_index = {entry: i for i, entry in enumerate(raw_entries_cleaned)}

# Find the indices in the order of kevin_list
indices = [entry_to_index[entry] for entry in kevin_ids if entry in entry_to_index]

df = pd.DataFrame(indices, columns=['SNP'])
# Output the indices
df.to_csv("../Data/TopSNPs/Kevin/ranked_snps_Kevin.csv", index=False)
print(df)
print(len(indices))