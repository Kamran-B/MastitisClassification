from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Load CSV with first column as index
df = pd.read_csv('acc.csv', index_col=0)

# Multiply all float values by 100
df = df.map(lambda x: x * 100 if isinstance(x, (float, int)) else x)


# Define custom pastel green to yellow colormap
custom_cmap = LinearSegmentedColormap.from_list(
    "white_green",
    ["#ffffff", "#138736"]
)

# Plot the customized heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(
    df,
    annot=True,
    cmap=custom_cmap,
    linewidths=0.0,
    fmt=".2f",
    cbar_kws={'label': 'Accuracy (%)'},
    center=80,
    annot_kws={"fontsize": 12, "color": "black"}
)

# Customize labels and layout
plt.title("Peak Accuracy Across Feature Selection Methods and SNP Counts", fontsize=14, weight='bold', pad=20)
plt.xlabel("")
plt.ylabel("")

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

sns.despine()
plt.tight_layout()
plt.show()
