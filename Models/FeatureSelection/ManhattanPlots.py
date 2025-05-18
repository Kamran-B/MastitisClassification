import pandas as pd
import matplotlib.pyplot as plt

kevin = pd.read_excel('../../SNPLists/mlma_Kevin.xlsx')
kevin = kevin.iloc[:, 0:2]
print(kevin)


raw = pd.read_csv("../../Data/RawData/raw_data.raw", sep=" ", nrows=1, header=None)

def strip_suffix(value):
    if isinstance(value, str) and (value.endswith('_A') or value.endswith('_B')):
        return value[:-2]
    return value

raw = raw.applymap(strip_suffix)
raw = raw.iloc[:, 6:].reset_index(drop=True)

raw_entries = raw.values.flatten()
sorted_kevin = kevin.set_index('SNP').reindex(raw_entries).reset_index()

# Display the sorted Kevin DataFrame
print("Sorted Kevin DataFrame:")
print(sorted_kevin)


df = pd.read_csv("../../SNPLists/ranked_snps_pca.csv")

# Sort by SNP ID
df['SNP_numeric'] = df['SNP'].str.extract(r'(\d+)$').astype(int)
df = df.sort_values(by='SNP_numeric').reset_index()

df["Chromosome"] = sorted_kevin["Chr"]
print(df.iloc[:10, :])


df["Index"] = range(len(df))

# Define colors for the chromosomes (using a repeating pattern of colors)
chromosome_colors = {1: 'red', 2: 'blue', 3: 'green', 4: 'purple', 5: 'orange', 6: 'cyan', 7: 'magenta'}
# Use modulo to cycle colors if there are more chromosomes than the length of the color list
df['Color'] = df["Chromosome"].apply(lambda x: chromosome_colors.get(x % len(chromosome_colors), 'black'))

# Plotting the Manhattan plot with color based on chromosome
plt.figure(figsize=(12, 6))
plt.scatter(df["Index"], df["Total Contribution"], color=df["Color"], alpha=0.7)

# Add labels and titles
plt.xlabel("SNP Index (Sorted by SNP ID)", fontsize=12)
plt.ylabel("PCA Score", fontsize=12)
plt.title("Manhattan Plot of SNP Importance for PCA", fontsize=14)
plt.grid(alpha=0.3)

# Show the plot
plt.show()