import re
import matplotlib.pyplot as plt
import pandas as pd
import io


def plot_optimizer_metrics(filename, metric_to_plot):
    """
    Reads an SPSA optimizer log file, extracts iteration numbers, metric values,
    and parameter categories, then generates a scatter plot.

    Args:
        filename (str): The name of the log file to read.
        metric_to_plot (str): The metric to plot on the y-axis ('Metric1' or 'Metric2').
    """
    iterations = []
    metric_values = []
    categories = []
    
    current_iteration = None
    current_category = None

    prior_block_iterations = 0
    running_iterations_tot = 0

    # Regex patterns to find the required information in the log file
    # This pattern captures the parameter category (e.g., 'od', 'lc') and the iteration number.
    iteration_pattern = re.compile(r"SPSA \((od|sim|cfm|lc|heterogeneity)\) Iteration (\d+)/(\d+)")
    # This pattern captures the values for Metric1 and Metric2.
    metric_pattern = re.compile(r"Metric1=([\d.]+), Metric2=([\d.]+)")

    try:
        with open(filename, 'r') as f:
            for line in f:
                # Search for lines indicating a new SPSA iteration and its category
                iteration_match = iteration_pattern.search(line)
                if iteration_match:
                    new_category = iteration_match.group(1) # Extract the category (e.g., 'od')
                    current_iteration = int(iteration_match.group(2)) # Extract the iteration number

                    # If we've moved to a new block, update the running_iterations_tot
                    if new_category != current_category:
                        running_iterations_tot += prior_block_iterations # Extract the total iterations for this block
                        current_category = new_category
                        prior_block_iterations = int(iteration_match.group(3))

                    # Update currrent_iterations w/ the number of iterations from previous blocks
                    current_iteration += running_iterations_tot

                    continue # Move to the next line, expecting the metric values soon

                # if current_iteration is not None:
                #     import ipdb; ipdb.set_trace()

                # Search for lines containing the average error metrics
                metric_match = metric_pattern.search(line)
                # Ensure we have both iteration and category before processing metric values
                if metric_match and current_iteration is not None and current_category is not None:
                    metric1_value = float(metric_match.group(1)) # Extract Metric1 value
                    metric2_value = float(metric_match.group(2)) # Extract Metric2 value

                    # Append the extracted data to our lists
                    iterations.append(current_iteration)
                    categories.append(current_category)
                    
                    # Choose which metric to plot based on the 'metric_to_plot' argument
                    if metric_to_plot == "Metric1":
                        metric_values.append(metric1_value)
                    elif metric_to_plot == "Metric2":
                        metric_values.append(metric2_value)
                    else:
                        # Fallback if an invalid metric name is provided
                        print(f"Warning: Invalid metric_to_plot '{metric_to_plot}'. Plotting Metric1 by default.")
                        metric_values.append(metric1_value)


    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return

    # Check if any data was extracted
    if not iterations:
        print("No relevant data found to plot. Please ensure the log file format matches expectations.")
        return

    # Create a Pandas DataFrame from the collected data for easy manipulation and plotting
    df = pd.DataFrame({
        'Iteration': iterations,
        'Metric Value': metric_values,
        'Category': categories
    })

    # Color map for the different parameter categories
    color_map = {
        'od': 'red',
        'sim': 'purple',
        'cfm': 'blue',
        'lc': 'green',
        'heterogeneity': 'orange',
    }

    plt.figure(figsize=(12, 7)) # Adjust figure size for better readability

    # Iterate through unique categories in the DataFrame and plot each one separately.
    # (Allows for a distinct legend entry for each category)
    for category in df['Category'].unique():
        subset = df[df['Category'] == category] # Get data for the current category
        plt.scatter(
            subset['Iteration'],
            subset['Metric Value'],
            color=color_map.get(category, 'gray'), # Use defined color, default to gray if not found
            label=category, # Label for the legend
            s=50, # Marker size for better visibility
            alpha=0.7 # Transparency of the points
        )

    plt.title(f'Optimizer Metric ({metric_to_plot}) over Iterations', fontsize=16)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel(f'{metric_to_plot} Value', fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Parameter Category')
    plt.tight_layout()
    plt.savefig('./analysis/scatter_SPSA.png') 

# Generate the plot for Metric1
# plot_optimizer_metrics("optimizer_output.log", "Metric1")

# Generate the plot for Metric2
plot_optimizer_metrics("optimizer_output.log", "Metric2")
