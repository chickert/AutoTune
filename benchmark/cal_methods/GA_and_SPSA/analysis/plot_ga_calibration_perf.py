import re
import matplotlib.pyplot as plt

def analyze_and_plot_metrics(log_file_path="optimizer_output.log"):
    """
    Reads a log file, extracts Metric1 and Metric2 from specific lines,
    and then plots these metrics.

    Args:
        log_file_path (str): The path to the optimizer_output.log file.
    """
    metric1_values = []
    metric2_values = []
    
    # Define the pattern to search for in the log lines
    search_string = "INFO - error_functions_custom - Average error metrics over"
    
    # Define the regex pattern to extract Metric1 and Metric2 values
    # It looks for 'Metric1=' followed by a float, and 'Metric2=' followed by a float
    # We use non-greedy matching (.*?) to ensure it doesn't span across multiple metrics if format changes
    metric_pattern = re.compile(r"Metric1=([\d.]+), Metric2=([\d.]+)")

    try:
        with open(log_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if search_string in line:
                    match = metric_pattern.search(line)
                    if match:
                        try:
                            metric1 = float(match.group(1))
                            metric2 = float(match.group(2))
                            metric1_values.append(metric1)
                            metric2_values.append(metric2)
                        except ValueError:
                            print(f"Warning: Could not convert metrics to float on line {line_num}: {line.strip()}")
                    else:
                        print(f"Warning: Line {line_num} contained '{search_string}' but metrics could not be extracted: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: The file '{log_file_path}' was not found.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return

    if not metric1_values:
        print("No 'Average error metrics' lines found with Metric1 and Metric2 data.")
        return

    # Plotting the extracted metrics
    plt.figure(figsize=(10, 6))

    # Plot Metric1
    # plt.scatter(range(len(metric1_values)), metric1_values, marker='o', linestyle='-', label='Count RMSE')
    # Plot Metric2
    plt.scatter(range(len(metric2_values)), metric2_values, marker='x', linestyle='--', label='Speed IOU')

    plt.title('Data from all simulations')
    plt.xlabel('Observation Index')
    plt.ylabel('Metric Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./analysis/scatter_all_optimizer_output_5it5gen.png') 
    plt.show()

# To run the analysis, call the function:
# Make sure 'optimizer_output.log' is in the same directory as your script
# or provide the full path to the file.
analyze_and_plot_metrics('optimizer_output_5it5gen.log')

