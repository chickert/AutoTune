"""
File to infer lane boundaries from vehicle y-positions at detectors using kernel density estimation (KDE) and plot the results.
The script fetches I-24 data, infers lane boundaries, and generates plots of the KDE distributions of vehicle y-positions at detectors.
Note that the process of fetching I-24 data can be time-consuming, so the script allows for reusing existing data. Similarly (and for reproducibility), inferred
lane boundaries may be reused if they have already been generated.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BENCHMARK_DIR))

from utils import (
    get_array,
    get_time_of_day_in_minutes,
    get_valid_trajectories,
    key_func,
)


### Parameters ###

# Parameters to determine which portions of processing pipeline to run
fetch_i24_data = False                                  # Whether to fetch I-24 data or to use existing data
find_lane_boundaries = False                             # Whether to infer lane boundaries or reuse existing ones
plot_indiv_detector_kdes = True                         # Whether to plot individual detector KDEs or not
plot_all_detector_kdes = True                            # Whether to plot all detector KDEs together or not

# Parameters shared across functions
detector_data_dir = str(BENCHMARK_DIR / "build_data") + "/"  # Path to the directory containing the detector data files
start_time = '2022-11-30 06:00:00'                      # Start time for data collection in 'YYYY-MM-DD HH:MM:SS' format
end_time = '2022-11-30 10:00:00'                        # End time for data collection in 'YYYY-MM-DD HH:MM:SS' format
day = start_time.split(' ')[0].split('-')[1] + '-' + start_time.split(' ')[0].split('-')[2]     # Get date in 'MM-DD' format
date_str_mmdd = day.split('-')[0] + day.split('-')[1]   # Get date in MMDD format
y_positions_at_detectors_path = detector_data_dir + f"y_positions_at_detectors_{date_str_mmdd}_{get_time_of_day_in_minutes(start_time)}-{get_time_of_day_in_minutes(end_time)}.json"  # Path for where to save/load the y-positions at detectors
detector_dict_path = detector_data_dir + "sumo_net_detector_dict.json"  # Path for where to save/load the detector dictionary
lane_boundaries_path = detector_data_dir + f"lane_boundaries_{date_str_mmdd}_{get_time_of_day_in_minutes(start_time)}-{get_time_of_day_in_minutes(end_time)}.json"  # Path for where to save/load the lane boundaries
output_plot_dir = detector_data_dir + "plots/" + day + "/"  # Path for where to save the output plots
bw_adjust = 0.75                                        # Bandwidth adjustment parameter for KDE -- affects smoothness of the KDE plots
alpha = 0.2                                             # Transparency level for the lanes' shaded areas

# Parameters for get_veh_positions_at_detectors() only -- to fetch I-24 detection data
i24_data_dir = str(BENCHMARK_DIR / "i24_data") + "/"    # Path to the directory containing the I-24 data files
i24_traj_data_filename = f"{i24_data_dir}{day}.json"    # Source data file for determining lane boundaries
verbose = True

# Parameters for infer_lane_boundaries() only -- to infer lane boundaries
skip_detectors_list = []                                # List of detectors to skip (e.g., those with no detections)
right_lane_low_detections_list_i = ['559-eastbound']    # Manual modification to handle detector with low detections in rightmost lane
right_lane_low_detections_list_ii = ['556-westbound']   # Additional manual modification to handle detector with low detections in rightmost lane
ignore_ramp_detectors_list = ['560-eastbound']          # List of detectors with ramps nearby (e.g., those that should be ignored for lane boundary inference)
right_lane_splitting_list = ['561-westbound']           # Manual modification to handle detector with right lane splitting (otherwise will find 2 lanes where there is only one)
edge_threshold_reg=0.001                                # Threshold for regular detectors to identify outermost lane edges (left and right)
left_edge_threshold_low_i = 0.001                       # Left threshold for detectors in right_lane_low_detections_list_i
right_edge_threshold_low_i = 0.00001                    # Right threshold for detectors in right_lane_low_detections_list_i
right_edge_threshold_low_ii = 0.0002                    # Additional right edge threshold for detectors in right_lane_low_detections_list_ii

# Parameters for plot_single_detector_kde() only -- to plot the KDE distribution of vehicle y-positions at individual detector(s)
plot_individual_datapoints = True                       # Whether to plot individual datapoints as well
vertical_jitter_spread = 0.004                          # Spread of vertical jitter for individual datapoints to improve visualization
lane_colors = ['#355070', '#6D597A', '#B56576', '#E56B6F', '#EAAC8B', '#F0CE9E']    # List of colors for the lanes
ind_kde_figsize = (16, 3)                               # Figure size for the plot(s)  
xticks_ind_kde = np.arange(0, 90, 2)                    # List of x-ticks for the plot
vert_offset = 0.08                                     # Vertical offset -- how far plot of individual datapoints should hover above KDE (relative to the horizontal axis)
point_size = 1                                          # Size of individual datapoints

# Parameters for plot_all_kdes() only -- to plot the KDE distribution of vehicle y-positions at all detectors
num_cols = 5                                            # Number of columns for the subplot grid. Number of rows is calculated automatically.
all_kde_figsize=(16, 8)                                 # Figure size for the plot  
xlims = (0, 90)                                         # X-axis limits for the plot
ylims = (0, 0.09)                                       # Y-axis limits for the plot
xticks_all_kde = np.arange(0, 90, 10)                   # List of x-ticks for the plot
fontsize_title = 18                                     # Font size for the title of the plot
fontsize_x = 14                                         # Font size for the x-axis label

##################


def load_dictionaries(
    detector_dict_path=None,
    y_positions_at_detectors_path=None,
    lane_boundaries_path=None,
):
    """
    Function to load the detector dictionary, y-positions at detectors, and/or lane boundaries from JSON files.

    Args:
        detector_dict_path (str): Path to the detector dictionary file.
        y_positions_at_detectors_path (str): Path to the JSON file containing vehicle y-positions at detectors.
        lane_boundaries_path (str): Path to the JSON file containing lane boundaries.

    Returns:
        tuple: A tuple containing the loaded dictionaries (detector_dict, y_positions_at_detectors_dict, lane_boundaries_dict).
    """

    if detector_dict_path:
        with open(detector_dict_path, "r") as f:
            sumo_net_detector_dict = json.load(f)
            print(f"\nLoaded sumo_net_detector_dict from {detector_dict_path}.")

    if y_positions_at_detectors_path:
        with open(y_positions_at_detectors_path, "r") as f:
            y_positions_at_detectors_dict = json.load(f)
            print(
                f"\nLoaded y-positions at detectors from {y_positions_at_detectors_path}."
            )

    if lane_boundaries_path:
        with open(lane_boundaries_path, "r") as f:
            lane_boundaries_dict = json.load(f)
            print(f"\nLoaded lane boundaries from {lane_boundaries_path}.")

    return sumo_net_detector_dict, y_positions_at_detectors_dict, lane_boundaries_dict


def get_y_positions_at_detector(valid_trajectories, detector_loc_milemarker):
    """
    Function to extract the y-positions at the closest index to the detector, useful for identifying lane numbers and boundaries.

    Args:
        valid_trajectories (list): List of trajectories that are within the desired time range, pass by the loop location, and are in the correct direction.
        detector_loc_milemarker (float): Milemarker of the 'detector' location for which we want the data.

    Returns:
        y_positions_at_closest_index (list): List of each trajectory's y-position at the point closest to the detector.
            Note: Uses absolute value of y-position, since y-pos is negative for eastbound traffic.
    """

    # Sanity check to see how close the trajectory is to the detector
    max_meters_from_detector = 0

    # Initialize a list to store y-positions
    y_positions_at_closest_index = []

    for trajectory in valid_trajectories:
        x_position = (
            get_array(trajectory.get("x_position", None)) / 5280
        )  # Converting feet to miles
        y_position = get_array(trajectory.get("y_position", None))

        # Build dataframe for the current trajectory
        data_index = pd.DataFrame(np.column_stack((x_position, y_position)))
        data_index.columns = ["x", "y"]

        ## Find the lane location at the point where the trajectory crosses the loop detector
        # Find the row where 'x' is closest to the detector location
        closest_index = abs(data_index["x"] - detector_loc_milemarker).idxmin()
        dist_at_closest_index = abs(
            data_index.loc[closest_index, "x"] - detector_loc_milemarker
        )
        # Store the maximum distance from the detector for a later check (if it's too far, may indicate data quality issues)
        if dist_at_closest_index > max_meters_from_detector:
            max_meters_from_detector = (
                dist_at_closest_index * 1609.34
            )  # Converting miles to meters

        # Get the y-position at the closest index (abs val, since eastbound traffic has negative y-pos)
        y_positions_at_closest_index.append(abs(data_index.loc[closest_index, "y"]))

    print(
        f"\tOf the {len(valid_trajectories)} trajectories, the maximum x-dist from detector is: {max_meters_from_detector:.4f} meters"
    )
    print(f"\tdetector_loc_milemarker: {detector_loc_milemarker}")

    return y_positions_at_closest_index


def get_veh_positions_at_detectors(
    detector_dict_path,
    start_time,
    end_time,
    i24_traj_data_filename,
    y_positions_at_detectors_path,
    verbose,
):
    """
    Function to get all vehicle latitudinal positions (y-positions) at each detector (in feet) in a given time increment.
    Saves the data to a JSON file, where keys are detector IDs and values are lists of y-positions for all vehicles that crossed the detector in the given time frame.

    Args:
        detector_dict_path (str): Path to the detector dictionary file -- it should have detector IDs as keys and milemarker locations as values.
            Note: IDs should be of format "556-eastbound" or "556-westbound", since direction is derived from the ID.
        start_time (str): Start time for data collection in 'YYYY-MM-DD HH:MM:SS' format.
        end_time (str): End time for data collection in 'YYYY-MM-DD HH:MM:SS' format.
        i24_traj_data_filename (str): Source data file for determining lane boundaries.
        y_positions_at_detectors_path (str): Path for where to save the output JSON file.
        verbose (bool): If True, print additional information during processing.

    Returns:
        None: The function saves the y-positions (in feet) at detectors to a JSON file, where keys are detector IDs
                and values are lists of y-positions (in feet) for all vehicles that crossed the detector in the given time frame.
    """

    # Read detector dictionary from JSON file
    with open(detector_dict_path, "r") as f:
        sumo_net_detector_dict = json.load(f)
        print(f"\nLoaded sumo_net_detector_dict from {detector_dict_path}.")

    y_positions_at_detectors_dict = {}

    # Iterate over all detectors (first by eastbound, then by westbound)
    for detector in sorted(sumo_net_detector_dict.keys(), key=key_func):

        detector_loc_milemarker = sumo_net_detector_dict[detector]["milemarker"]
        direction_string = detector.split("-")[1]

        # Get relevant I-24 trajectories for the current time increment, detector, and direction
        valid_trajectories = get_valid_trajectories(
            i24_traj_data_filename,
            start_time,
            end_time,
            direction_string,
            detector_loc_milemarker,
            verbose,
        )

        # Get the y-positions at each detector across all the trajectories
        y_positions_at_closest_index = get_y_positions_at_detector(
            valid_trajectories, detector_loc_milemarker
        )

        # Add the data to the output dictionary
        y_positions_at_detectors_dict[detector] = y_positions_at_closest_index

    # Save resulting dictionary to JSON
    with open(y_positions_at_detectors_path, "w") as f:
        json.dump(y_positions_at_detectors_dict, f, indent=4)
        print(f"\nSaved y-positions at detectors to {y_positions_at_detectors_path}")


def infer_lane_boundaries(
    y_positions_at_detectors_path,
    lane_boundaries_path,
    bw_adjust,
    skip_detectors_list,
    right_lane_low_detections_list_i,
    right_lane_low_detections_list_ii,
    ignore_ramp_detectors_list,
    right_lane_splitting_list,
    edge_threshold_reg,
    left_edge_threshold_low_i,
    right_edge_threshold_low_i,
    right_edge_threshold_low_ii,
):
    """
    Function to infer lane boundaries using the y-positions at detectors data.

    Args:
        y_positions_at_detectors_path (str): Path to the JSON file containing vehicle y-positions (latitudinal positions) as they cross detectors over some time period.
        lane_boundaries_path (str): Path for where to save the output JSON file with lane boundaries.
        bw_adjust (float): Bandwidth adjustment parameter for KDE.
        skip_detectors_list (list): List of detectors to skip (e.g., those with no detections).
        right_lane_low_detections_list_i (list): Manual modification to handle detector(s) with few detections in rightmost lane (to avoid accidentally ignoring the lane).
        right_lane_low_detections_list_ii (list): Additional manual modification to handle detecto(s) with few detections in rightmost lane (to avoid accidentally ignoring the lane).
        ignore_ramp_detectors_list (list): List of detectors with ramps nearby (e.g., those that should be ignored for lane boundary inference).
        right_lane_splitting_list (list): Manual modification to handle detector with right lane splitting (otherwise will find 2 lanes where there is only one)
        edge_threshold_reg (float): Threshold (both left and right) for regular detectors to identify outermost lane edges.
        left_edge_threshold_low_i (float): Left threshold for detectors in right_lane_low_detections_list_i to identify outermost lane edges.
        right_edge_threshold_low_i (float): Right threshold for detectors in right_lane_low_detections_list_i to identify outermost lane edges.
        right_edge_threshold_low_ii (float): Additional right edge threshold for detectors in right_lane_low_detections_list_ii to add one more lane edge.

    Returns:
        None: The function saves the lane boundaries to a JSON file, where keys are detector IDs and values are lists of lane boundaries (in feet)
    """

    # Load the y-positions at detectors from the JSON file
    with open(y_positions_at_detectors_path, "r") as f:
        y_positions_at_detectors_dict = json.load(f)
        print(
            f"\nLoaded y-positions at detectors from {y_positions_at_detectors_path}."
        )

    # Initialize dictionary to store lane boundaries
    lane_boundaries_dict = {}

    # Iterate over all detectors
    for detector in y_positions_at_detectors_dict.keys():

        # Skip detectors that are in the skip list (for example, those with no detections due to detector error)
        if detector in skip_detectors_list:
            continue

        kde = sns.kdeplot(
            data=y_positions_at_detectors_dict[detector], bw_adjust=bw_adjust
        )

        x, y = kde.lines[-1].get_data()

        # Handle detectors with ramp nearby that should be ignored
        if detector in ignore_ramp_detectors_list:
            x = x[:125]
            y = y[:125]

        # Find local minima using scipy.signal.find_peaks
        valleys, _ = find_peaks(-y)  # Invert y for finding minima

        # Handle detectors with right lane splitting (otherwise will infer 2 lanes where there is only one)
        if detector in right_lane_splitting_list:
            # Omit the rightmost local minima
            valleys = valleys[:-1]

        # Get x-coordinates of local minima
        local_minima_x = x[valleys]

        # To avoid identifying lanes where none exist, drop minima corresponding to spurious peaks
        # These may be due to lanes splitting, detector errors, etc.
        # Note, however, that this will fail for legitimate lanes with few detections, so we omit those
        if detector not in right_lane_low_detections_list_i:

            peaks, _ = find_peaks(y)
            peak_idx = 0
            while y[peaks][peak_idx] < np.mean(y[valleys]):
                # Omit the leftmost local minima
                local_minima_x = local_minima_x[1:]
                peak_idx += 1

            peak_idx = 1
            while y[peaks][-peak_idx] < np.mean(y[valleys]):
                # Omit the rightmost local minima
                local_minima_x = local_minima_x[:-1]
                peak_idx += 1

            # Find leftmost lane edge: first x-coord at which kde plot crosses threshold
            local_minima_x = np.insert(
                local_minima_x, 0, x[np.where(y > edge_threshold_reg)[0][0]]
            )
            # Find rightmost lane edge last x-coord at which kde plot crosses threshold
            local_minima_x = np.append(
                local_minima_x, x[np.where(y > edge_threshold_reg)[0][-1]]
            )

            if detector in right_lane_low_detections_list_ii:
                # Add one more local minima to the right to compensate for low detections
                local_minima_x = np.append(
                    local_minima_x, x[np.where(y > right_edge_threshold_low_ii)[0][-1]]
                )

        else:
            # May still need to omit leftmost minimum due to false minimum
            local_minima_x = local_minima_x[1:]
            # Find leftmost lane edge: first x-coord at which kde plot crosses threshold
            local_minima_x = np.insert(
                local_minima_x, 0, x[np.where(y > left_edge_threshold_low_i)[0][0]]
            )
            # Find rightmost lane edge last x-coord at which kde plot crosses threshold
            local_minima_x = np.append(
                local_minima_x, x[np.where(y > right_edge_threshold_low_i)[0][-1]]
            )

        # Round to 2 decimal places
        local_minima_x = np.round(local_minima_x, 1)

        lane_boundaries_dict[detector] = list(local_minima_x)

    # Save the lane boundaries to a JSON file
    with open(lane_boundaries_path, "w") as f:
        json.dump(lane_boundaries_dict, f, indent=4)
        print(f"\nSaved lane boundaries to {lane_boundaries_path}")


def plot_single_detector_kde(
    lane_boundaries_dict,
    y_positions_at_detectors_dict,
    sumo_net_detector_dict,
    detector,
    start_time,
    end_time,
    output_plot_dir,
    plot_individual_datapoints,
    vertical_jitter_spread,
    bw_adjust,
    lane_colors,
    ind_kde_figsize,
    alpha,
    vert_offset,
    point_size,
    xticks_ind_kde,
):
    """
    Function to plot the kernel density estimate (KDE) distribution of vehicle y-positions at a single detector, with the inferred lane boundaries overlaid.
        Optionally plots the individual datapoints used for the KDE as well.
    Args:
        lane_boundaries_dict (dict): Dictionary containing inferred lane boundaries for each detector.
        y_positions_at_detectors_dict (dict): Dictionary containing vehicle y-positions at detectors.
        sumo_net_detector_dict (dict): Dictionary containing detector information as present in the network. Used to compare inferred lane number with network lane number in figure title.
        detector (str): The ID of the detector to plot.
        start_time (str): Start time for data collection in 'YYYY-MM-DD HH:MM:SS' format.
        end_time (str): End time for data collection in 'YYYY-MM-DD HH:MM:SS' format.
        output_plot_dir (str): Directory to save the output plot.
        plot_individual_datapoints (bool): If True, plot individual datapoints as well.
        vertical_jitter_spread (float): Spread of vertical jitter for individual datapoints to improve visualization
        bw_adjust (float): Bandwidth adjustment parameter for KDE.
        lane_colors (list): List of colors for the lanes.
        ind_kde_figsize (tuple): Figure size for the plot.
        alpha (float): Transparency level for the lanes' shaded areas.
        vert_offset (float): Vertical offset -- how far plot of individual datapoints should hover above KDE (relative to the horizontal axis)
        point_size (int): Size of individual datapoints.
        xticks_ind_kde (list): List of x-ticks for the plot.

    Returns:
        None: The function generates a plot and saves it to the specified directory.
    """

    lane_boundaries = lane_boundaries_dict[detector]

    # Adjust size
    plt.figure(figsize=ind_kde_figsize)

    # Plot the distribution of y-positions at the detector
    sns.kdeplot(data=y_positions_at_detectors_dict[detector], bw_adjust=bw_adjust)

    # If desired, plot individual datapoints as hovering above the kde plot
    if plot_individual_datapoints:

        # Create some vertical offset to improve visualization
        num_points = len(y_positions_at_detectors_dict[detector])
        random_y_offset = np.random.uniform(
            -vertical_jitter_spread / 2, vertical_jitter_spread / 2, num_points
        )

        plt.scatter(
            y_positions_at_detectors_dict[detector],
            np.zeros_like(y_positions_at_detectors_dict[detector])
            + vert_offset
            + random_y_offset,  # jitter for better visibility, with offset to raise points above kde plot
            s=point_size,  # size of the points
            color="steelblue",
            alpha=alpha,
        )

    # Plot the lane boundaries as vertical lines
    for i, lower_lane_boundary in enumerate(lane_boundaries):

        upper_lane_boundary = (
            lane_boundaries[i + 1] if i + 1 < len(lane_boundaries) else None
        )

        _ = plt.axvline(x=lower_lane_boundary, color="black", linestyle=":")

        # Visualize lane boundary and shading between the current lane_boundary and the previous one
        if upper_lane_boundary:
            plt.axvspan(
                xmin=lower_lane_boundary,
                xmax=upper_lane_boundary,
                color=lane_colors[i],
                alpha=alpha,
                label=f"Lane {i}",
            )

    plt.xlabel("y-position (feet)")
    plt.title(
        f"Distribution of vehicle y-positions at detector {detector} from {start_time} to {end_time}\nData: {len(y_positions_at_detectors_dict[detector])} trajectories across {len(lane_boundaries) - 1} lanes | Network: {sumo_net_detector_dict[detector]['num_lanes']} lanes"
    )
    # plt.xlabel("Latitudinal position (feet)")
    # plt.ylabel("KDE density")
    # plt.title(f"Lane boundaries inferred at detector {detector} | {bw_adjust:.2f} KDE bandwidth smoothing factor")

    # Set x-ticks granularity
    plt.xticks(xticks_ind_kde)
    plt.xlim(xticks_ind_kde[0], xticks_ind_kde[-1])
    plt.ylim(0, 0.09)
    plt.legend()

    # tight layout
    plt.tight_layout()

    # Get date in MMDD format
    date_str = (
        start_time.split(" ")[0].split("-")[1] + start_time.split(" ")[0].split("-")[2]
    )

    # Save file with name of detector, time, and bin width parameter
    save_path = (
        output_plot_dir
        + f"lane_boundaries_{detector}_bw-{str(bw_adjust).replace('.', '')}_{date_str}_{get_time_of_day_in_minutes(start_time)}-{get_time_of_day_in_minutes(end_time)}.pdf"
    )
    plt.savefig(save_path)
    print(f"Generated plot and saved to {save_path}")
    plt.close()


def plot_all_kdes(
    lane_boundaries_dict,
    y_positions_at_detectors_dict,
    bw_adjust,
    start_time,
    end_time,
    output_plot_dir,
    num_cols,
    all_kde_figsize,
    xlims,
    ylims,
    xticks_all_kde,
    fontsize_title,
    fontsize_x,
):
    """
    Function to plot the kernel density estimate (KDE) distribution of vehicle y-positions at all detectors, with the inferred lane boundaries overlaid.

    Args:
        lane_boundaries_dict (dict): Dictionary containing inferred lane boundaries for each detector.
        y_positions_at_detectors_dict (dict): Dictionary containing vehicle y-positions at detectors.
        bw_adjust (float): Bandwidth adjustment parameter for KDE.
        start_time (str): Start time for data collection in 'YYYY-MM-DD HH:MM:SS' format.
        end_time (str): End time for data collection in 'YYYY-MM-DD HH:MM:SS' format.
        output_plot_dir (str): Directory to save the output plot.
        num_cols (int): Number of columns for the subplot grid. num_rows is calculated automatically.
        all_kde_figsize (tuple): Figure size for the plot.
        xlims (tuple): X-axis limits for the plot.
        ylims (tuple): Y-axis limits for the plot.
        xticks_all_kde (list): List of x-ticks for the plot.
        fontsize_title (int): Font size for the title of the plot.
        fontsize_x (int): Font size for the x-axis label.

    Returns:
        None: The function generates a plot and saves it to the specified directory.
    """

    # Get the number of rows for the subplots automatically from the number of detectors and specified number of columns
    num_rows = (
        len(lane_boundaries_dict) + num_cols - 1
    ) // num_cols  # Ceiling division to get the number of rows needed

    # Create a figure with the desired grid size
    fig, axes = plt.subplots(num_rows, num_cols, figsize=all_kde_figsize)
    axes_flat = axes.flatten()

    # Counter for current subplot index
    plot_index = 0
    longest_legend_handles = None
    longest_legend_labels = None

    # Iterate through sorted detectors
    for detector in sorted(lane_boundaries_dict.keys(), key=key_func):
        # Calculate row and column indices for the current subplot
        row_index = plot_index // num_cols
        col_index = plot_index % num_cols

        # Get the current detector's data
        y_positions = y_positions_at_detectors_dict[detector]
        lane_boundaries = lane_boundaries_dict[detector]

        # Create the KDE plot on the current subplot
        ax = axes_flat[plot_index]
        sns.kdeplot(ax=ax, data=y_positions, bw_adjust=bw_adjust)
        ax.set_title(f"{detector}: {len(y_positions)} traj.")

        # Plot lane boundaries and shaded areas
        for i, lower_lane_boundary in enumerate(lane_boundaries):

            upper_lane_boundary = (
                lane_boundaries[i + 1] if i + 1 < len(lane_boundaries) else None
            )

            ax.axvline(x=lower_lane_boundary, color="black", linestyle=":")

            # Visualize lane boundary and shading between the current lane_boundary and the previous one
            if upper_lane_boundary:
                ax.axvspan(
                    xmin=lower_lane_boundary,
                    xmax=upper_lane_boundary,
                    color=lane_colors[i],
                    alpha=alpha,
                    label=f"Lane {i}",
                )

        # Adjust axis limits
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

        # Hide x and y labels for all subplots except the bottom row and leftmost column
        # To do this, determine if this subplot is the last one *actually plotted* in its column
        # (when total number of detectors is not a multiple of num_cols)
        is_effectively_last_in_column = (plot_index + num_cols) >= len(
            lane_boundaries_dict
        )
        if is_effectively_last_in_column:
            ax.set_xticks(xticks_all_kde)
        else:
            # Hide x-ticks because there's another plotted item below it in this column
            ax.set_xticks([])

        if col_index != 0:
            ax.set_yticks([])
            ax.set_ylabel("")

        # Get legend handles and labels from the current subplot
        handles, labels = ax.get_legend_handles_labels()

        # Check if it's the longest legend so far
        if not longest_legend_handles or len(handles) > len(longest_legend_handles):
            longest_legend_handles = handles
            longest_legend_labels = labels

        # Update the plot index
        plot_index += 1

    total_subplots_created = num_rows * num_cols
    for i in range(len(lane_boundaries_dict), total_subplots_created):
        axes_flat[i].axis("off")  # Turn off the axis for unused subplots

    # Adjust layout and display the figure
    fig.suptitle(
        f"Distribution (KDE plots) of vehicle latitudinal positions and inferred lane boundaries at detectors\nfrom {start_time} to {end_time}\n",
        fontsize=fontsize_title,
    )  # Add a main title

    # Set X label for the entire figure
    fig.text(
        0.5,
        0.05,
        "y-positions (in feet)",
        ha="center",
        va="center",
        fontsize=fontsize_x,
    )

    # fig.align_ylabels(left=True)  # Align y-axis labels
    plt.legend(
        longest_legend_handles,
        longest_legend_labels,
        loc="upper left",
        bbox_to_anchor=(1.05, 4.75),
        fontsize=fontsize_x,
    )  # Place legend outside the subplots

    # Get date in MMDD format
    date_str = (
        start_time.split(" ")[0].split("-")[1] + start_time.split(" ")[0].split("-")[2]
    )

    # Save file with time and bin width parameter
    save_path = (
        output_plot_dir
        + f"all_lane_boundaries_bw-{str(bw_adjust).replace('.', '')}_{date_str}_{get_time_of_day_in_minutes(start_time)}-{get_time_of_day_in_minutes(end_time)}.pdf"
    )
    plt.savefig(save_path)
    print(f"\nGenerated all-detectors KDE plot and saved to {save_path}\n")
    plt.close()


if __name__ == "__main__":

    # Fetch new data if specified, otherwise use existing data (this step can be time-consuming)
    if fetch_i24_data:
        # Call the function to get vehicle positions at detectors
        get_veh_positions_at_detectors(
            detector_dict_path,
            start_time,
            end_time,
            i24_traj_data_filename,
            y_positions_at_detectors_path,
            verbose,
        )

    # Infer lane boundaries if specified, otherwise use existing boundaries
    if find_lane_boundaries:
        # Call the function to infer lane boundaries
        infer_lane_boundaries(
            y_positions_at_detectors_path,
            lane_boundaries_path,
            bw_adjust,
            skip_detectors_list,
            right_lane_low_detections_list_i,
            right_lane_low_detections_list_ii,
            ignore_ramp_detectors_list,
            right_lane_splitting_list,
            edge_threshold_reg,
            left_edge_threshold_low_i,
            right_edge_threshold_low_i,
            right_edge_threshold_low_ii,
        )

    # Load the dictionaries from JSON files
    sumo_net_detector_dict, y_positions_at_detectors_dict, lane_boundaries_dict = (
        load_dictionaries(
            detector_dict_path=detector_dict_path,
            y_positions_at_detectors_path=y_positions_at_detectors_path,
            lane_boundaries_path=lane_boundaries_path,
        )
    )

    # Plot the KDE distributions and lane boundaries for each detector separately
    if plot_indiv_detector_kdes:
        for detector in sorted(sumo_net_detector_dict.keys(), key=key_func):
            plot_single_detector_kde(
                lane_boundaries_dict,
                y_positions_at_detectors_dict,
                sumo_net_detector_dict,
                detector,
                start_time,
                end_time,
                output_plot_dir,
                plot_individual_datapoints,
                vertical_jitter_spread,
                bw_adjust,
                lane_colors,
                ind_kde_figsize,
                alpha,
                vert_offset,
                point_size,
                xticks_ind_kde,
            )

    # Plot the KDE distributions and lane boundaries for all detectors together
    if plot_all_detector_kdes:
        plot_all_kdes(
            lane_boundaries_dict,
            y_positions_at_detectors_dict,
            bw_adjust,
            start_time,
            end_time,
            output_plot_dir,
            num_cols,
            all_kde_figsize,
            xlims,
            ylims,
            xticks_all_kde,
            fontsize_title,
            fontsize_x,
        )
