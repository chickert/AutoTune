import re
import matplotlib.pyplot as plt
import pandas as pd

def analyze_and_plot_best_fitness(log_file_path="optimizer_output.log"):
    """
    Reads a log file, extracts 'best fitness' values and their associated
    designations (from parentheses), and then plots these values in a
    color-coded scatterplot. This version does not require 'GA' in the line.

    Args:
        log_file_path (str): The path to the optimizer_output.log file.
    """
    fitness_data = []

    # Regex pattern to extract designation from parentheses and the fitness value.
    # It looks for '(designation)' and 'best fitness: value' on the same line.
    # Group 1: captures the designation (e.g., 'od', 'cfm')
    # Group 2: captures the fitness value (e.g., '0.6166')
    # The order of finding (designation) and best fitness is important if they can appear out of order.
    # Assuming (designation) typically appears before best fitness: value
    best_fitness_pattern = re.compile(r"\((\w+)\).*?best fitness: ([\d.]+)")

    try:
        with open(log_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Optimize by checking for key phrases before applying regex
                if "best fitness:" in line and "(" in line and ")" in line:
                    match = best_fitness_pattern.search(line)
                    if match:
                        try:
                            designation = match.group(1)
                            fitness_value = float(match.group(2))
                            fitness_data.append({'designation': designation, 'fitness': fitness_value})
                        except ValueError:
                            print(f"Warning: Could not convert fitness to float on line {line_num}: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: The file '{log_file_path}' was not found. Please ensure the log file exists.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return

    if not fitness_data:
        print("No 'best fitness' lines with extractable data were found. Please check your log file format.")
        return

    df = pd.DataFrame(fitness_data)
    df['observation_index'] = df.index

    # --- Plotting the Data ---
    plt.figure(figsize=(12, 7))

    for designation_name, group in df.groupby('designation'):
        plt.scatter(group['observation_index'], group['fitness'], label=designation_name, s=50, alpha=0.3)

    plt.title('Best Fitness Values Over Time, Color-Coded by Parameter Block')
    plt.xlabel('Observation Index')
    plt.ylabel('Best Fitness Value')
    plt.grid(True)
    plt.legend(title='Optimizer Block')
    plt.tight_layout()
    plt.savefig('./analysis/scatter_best_fitness_optimizer_output_contsteplen.png')
    plt.show()

if __name__ == "__main__":
    analyze_and_plot_best_fitness('optimizer_output_contsteplen.log')