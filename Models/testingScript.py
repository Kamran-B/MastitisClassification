import random
import numpy as np
import json
from datetime import datetime
from Transformer import main

# Configuration
iterations = 10  # Number of runs
epochs = 5  # Epochs per run
random_seed = True  # Randomize seed

description = "Running base transformer with SNPs, herd year, and breed embedded together, no func coseq and with overlapping window (also changed hidden dim to 64 from 128)"

# Prepare to store all results
results = []

# Run experiment
for run_num in range(1, iterations + 1):
    # Set seed
    seed = random.randint(1, 1000) if random_seed else 42
    print(f"Running with seed: {seed}")

    # Run main function and collect accuracies
    accuracies = main(seed, epochs, False, True)
    fullAcc = np.array(accuracies)

    # Calculate stats
    avg_accuracy = fullAcc.mean()
    max_accuracy = fullAcc.max()

    # Save results for this run
    results.append({
        "run_number": run_num,
        "seed": seed,
        "accuracies_per_epoch": accuracies,
        "average_accuracy": avg_accuracy,
        "max_accuracy": max_accuracy
    })

    # Display results for this run
    print(f"Average accuracy for run {run_num}: {avg_accuracy}")
    print(f"Highest accuracy for run {run_num}: {max_accuracy}")

# Calculate overall stats
all_accuracies = np.concatenate([np.array(result["accuracies_per_epoch"]) for result in results])
overall_avg_accuracy = all_accuracies.mean()
overall_max_accuracy = max(result["max_accuracy"] for result in results)

# Display overall stats
print(f"Overall average accuracy across all runs and epochs: {overall_avg_accuracy}")
print(f"Overall highest accuracy across all runs: {overall_max_accuracy}")

# Save results to JSON
output = {
    "experiment_date": datetime.now().isoformat(),
    "description": description,
    "total_runs": iterations,
    "epochs_per_run": epochs,
    "random_seed_enabled": random_seed,
    "overall_average_accuracy": overall_avg_accuracy,
    "overall_max_accuracy": overall_max_accuracy,
    "runs": results
}

# Save to file
with open("experiment_notes2.json", "w") as f:
    json.dump(output, f, indent=4)

print("Results saved to experiment_notes.json")