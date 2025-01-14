import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the CSV file with the adjusted p-values
def load_and_plot_manhattan(csv_file):
    # Load the CSV file containing feature indices and adjusted p-values
    data = pd.read_csv(csv_file)

    # Check if the CSV contains the expected columns
    if 'Feature' not in data.columns or 'Adjusted_P_Value' not in data.columns:
        print("The CSV file must contain 'Feature' and 'Adjusted_P_Value' columns.")
        return

    # Extract feature indices and adjusted p-values
    feature_indices = data['Feature'].values
    adjusted_p_values = data['Adjusted_P_Value'].values

    # Compute -log10 of the adjusted p-values for the Manhattan plot
    log_p_values = -np.log10(adjusted_p_values)

    # Create the Manhattan plot
    plt.figure(figsize=(12, 6))

    # Plotting with Seaborn
    sns.scatterplot(x=feature_indices, y=log_p_values, color='blue', s=10)

    # Add a horizontal line for significance threshold (e.g., -log10(0.05) ≈ 1.3)
    plt.axhline(y=-np.log10(0.05), color='red', linestyle='--', label="FDR threshold (p = 0.05)")

    # Set labels and title
    plt.xlabel('Feature Index')
    plt.ylabel('-log10(Adjusted P-Value)')
    plt.title('Manhattan Plot of Features Based on FDR Adjusted P-Values')
    plt.legend()

    # Display the plot
    plt.show()


# Call the function with your CSV file
csv_file = "top_features.csv"  # Replace with the path to your generated CSV file
load_and_plot_manhattan(csv_file)
