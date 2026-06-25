import re
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_ga_fitness(log_file_path):
    """
    Reads a log file, extracts fitness values and generation numbers for "Best block solution",
    and plots the fitness over time, color-coding by the block label.

    Args:
        log_file_path (str): The path to the log file.
    """
    # Define regex patterns for extracting data
    # Matches the block label (e.g., 'od', 'cfm')
    label_pattern = r"GA for (\w+) finished"

    # Matches the fitness value
    fitness_pattern = r"Fitness: (\d+\.\d+)"

    # Lists to store the extracted data
    all_fitness_values = []
    all_labels = []

    # Read the log file line by line
    with open(log_file_path, 'r') as f:
        for line in f:
            # Check if the line contains the required phrase
            if "Best block solution this iteration" in line:
                # Extract the block label from the current line
                label_match = re.search(label_pattern, line)

                # Extract the fitness value from the current line
                fitness_match = re.search(fitness_pattern, line)

                if label_match and fitness_match:
                    label = label_match.group(1)
                    fitness = float(fitness_match.group(1))

                    # Append the extracted data to our lists
                    all_fitness_values.append(fitness)
                    all_labels.append(label)

    # If no data was found, print a message and exit
    if not all_fitness_values:
        print("No lines with 'Best block solution this iteration' were found.")
        return

    # Create a unique list of all labels found
    unique_labels = sorted(list(set(all_labels)))

    # Generate a color map for the labels
    colors = list(mcolors.TABLEAU_COLORS.values())
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot data for each label separately to create the legend
    for label in unique_labels:
        # Filter the data for the current label
        label_fitness_values = [all_fitness_values[i] for i, l in enumerate(all_labels) if l == label]
        
        # Create a sequential index for the x-axis for this label's data
        x_index = [i for i, l in enumerate(all_labels) if l == label]
        
        # Plot the data as a scatter plot
        plt.scatter(
            x_index,
            label_fitness_values,
            color=color_map[label],
            label=f'Block: {label}'
        )

    # Customize the plot
    plt.title('Best Block Solution Fitness Over Time (by Index)', fontsize=16)
    plt.xlabel('Block Solution Index', fontsize=12)
    plt.ylabel('Fitness Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Block Labels')
    plt.tight_layout()
    plt.savefig('./analysis/block_fitness_optimizer_output_contsteplen.png')
    plt.show()

# --- Example Usage ---
# Save your log file as `your_log_file.log` and provide the path here.
# For example, if the log file is in the same directory, use 'your_log_file.log'
# log_file_path = 'path/to/your/log/file.log'
log_file_path = 'optimizer_output.log' # Using a placeholder name for the example

# You can save the provided sample log data into a file named 'sample_log.log'
# to test the script directly.

# Run the plotting function
# You will need to have a file named `sample_log.log` with the provided data
# in the same directory as this script.
plot_ga_fitness(log_file_path)








