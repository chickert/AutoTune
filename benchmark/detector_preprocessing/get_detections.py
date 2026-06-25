"""
File to convert I-24 INCEPTION data into SUMO E1 detector measurements for a desired time range.
It uses network detectors, locations, and lane numbers -- along with corresponding inferred lane boundaries -- to determine the lane-level counts and speeds from INCEPTION data
for the specified time range. It uses the get_detections() method to produce a CSV file with the counts and speeds for each detector and lane.
Plots are produced for data quality checks, including the number of trajectories, excluded data points, and vehicles' maximum distance from the detector at their detection point.
Also, the plot_detection_data() method produces a histogram of the counts and speeds aggregated across all detectors, as well as per-detector bar charts of lane-level counts and mean speeds.
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BENCHMARK_DIR))

from utils import (
    convert_to_cst_unix,
    get_array,
    get_valid_trajectories,
    key_func,
    get_time_of_day_in_minutes,
    cache_trajectories,
)


### Parameters ###

run_get_detections = True                                                 # Whether to run the get_detections() method
run_plot_detection_data = True                                            # Whether to run the plot_detection_data() method

start_time = "2022-11-25 09:00:00"                                        # Start time of the time range. (YYYY-MM-DD HH:MM:SS) (inclusive)
end_time = "2022-11-25 09:10:00"                                          # End time of the time range. (YYYY-MM-DD HH:MM:SS) (exclusive)

detector_dict_path = str(BENCHMARK_DIR / "build_data" / "sumo_net_detector_dict.json")  # Path to the JSON file containing the SUMO network detector dictionary
lane_boundaries_path = str(BENCHMARK_DIR / "build_data" / "lane_boundaries_1130_0360-0600.json")  # Path to the JSON file containing lane boundaries
inception_data_dir = str(BENCHMARK_DIR / "i24_data") + "/"                # Path to the directory containing the I-24 data files
speed_calc_time_window = 1.0                                              # Time window (in seconds) over which speed is calculated  
output_dir = str(BENCHMARK_DIR / "detector_measurements") + "/"           # Directory to save the output CSV files
increment_duration = 30.0                                                 # Detector data aggregation period / reporting frequency (in seconds) 
verbose = True                                                            # Whether to print additional information
plots_subdir = "plots/"                                                   # Subdirectory for saving plots
lane_colors = ['#355070', '#6D597A', '#B56576', '#E56B6F', '#EAAC8B', '#F0CE9E'] # List of colors corresponding to lanes for plotting
disable_tqdm = False                                                       # Whether to disable the tqdm progress bar
qpkw_ylim = (0, 275)                                                      # Y-axis limits for the qPKW plots
vpkw_ylim = (0, 175)                                                      # Y-axis limits for the vPKW plots

##################


class DetectionBuilder:
    """
    Class to convert I-24 INCEPTION data into SUMO E1 detector measurements for a desired time range.
    It uses SUMO network detectors, locations, and lane numbers -- along with corresponding inferred lane boundaries -- to determine the lane-level counts and speeds from INCEPTION data
    for the specified time range.
    """

    def __init__(
        self,
        detector_dict_path,
        lane_boundaries_path,
        inception_data_dir,
        speed_calc_time_window,
        output_dir,
        plots_subdir,
        lane_colors,
        disable_tqdm,
    ):
        """
        Initialize the DetectionBuilder class.

        Args:
            detector_dict_path (str): Path to the JSON file containing the SUMO network detector dictionary.
            lane_boundaries_path (str): Path to the JSON file containing lane boundaries, where each key is a detector ID and the value is a list of lane boundaries (floats)
            inception_data_dir (str): Path to the directory containing the I-24 data files.
            speed_calc_time_window (float): Time window (in seconds) over which speed is calculated.
            output_dir (str): Directory to save the output CSV files.
            plots_subdir (str): Subdirectory for saving plots.
            lane_colors (list): List of colors corresponding to lanes for plotting
            disable_tqdm (bool): Whether to disable the tqdm progress bar. Best to disable when logging output as part of larger pipeline. 

        Returns:
            None
        """

        with open(detector_dict_path, "r") as f:
            self.sumo_net_detector_dict = json.load(f)
            print(f"\nLoaded sumo_net_detector_dict from {detector_dict_path}.")

        with open(lane_boundaries_path, "r") as f:
            self.lane_boundaries_dict = json.load(f)
            print(f"\nLoaded lane boundaries from {lane_boundaries_path}.")

        self.inception_data_dir = inception_data_dir
        self.speed_calc_time_window = speed_calc_time_window
        self.output_dir = output_dir
        self.plots_subdir = plots_subdir
        self.lane_colors = lane_colors
        self.disable_tqdm = disable_tqdm
        self.df = None
        self.cached_trajectories = None

        # Create dictionaries to store valid trajectories and excluded data points
        #   (Useful for debugging and understanding data quality)
        self.valid_trajectories_num_dict = {}
        self.time_excluded_data_points_dict = (
            {}
        )  # Note these aren't bad; they are a result of time-verification in get_lane_counts_and_speeds()
        self.lane_excluded_data_points_dict = {}
        self.max_meters_from_detector_per_increment_dict = {}
        # Initialize these dictionaries with for each detector
        for detector in self.sumo_net_detector_dict.keys():
            self.valid_trajectories_num_dict[detector] = 0
            self.time_excluded_data_points_dict[detector] = 0
            self.lane_excluded_data_points_dict[detector] = 0
            self.max_meters_from_detector_per_increment_dict[detector] = []

    def get_detections(
        self, start_time, end_time, increment_duration=30.0, verbose=False
    ):
        """
        Top-level method to get detection data (counts & speeds) for a specified time range.

        Args:
            start_time (str): Start time of the time range. (YYYY-MM-DD HH:MM:SS) (inclusive)
            end_time (str): End time of the time range. (YYYY-MM-DD HH:MM:SS) (exclusive)
            increment_duration (float): Duration of each time increment (in seconds) for which we want to get detection data.
                Corresponds to detector data aggregation period / reporting frequency (in seconds)
            verbose (bool): Whether to print additional information.

        Returns:
            None: The method modifies the instance's self.df in place and saves it to a CSV file with a timestamped filename.
        """

        # Check that start and end times are for the same day (important since data may not be available for all hours)
        start_day = (
            start_time.split(" ")[0].split("-")[1]
            + "-"
            + start_time.split(" ")[0].split("-")[2]
        )  # Get date in 'MM-DD' format
        end_day = (
            end_time.split(" ")[0].split("-")[1]
            + "-"
            + end_time.split(" ")[0].split("-")[2]
        )  # Get date in 'MM-DD' format
        assert (
            start_day == end_day
        ), f"Start and end days do not match: {start_day} != {end_day}"

        # Construct the output_filename (and appropriate directory, if needed)
        date_str_mmdd = (
            start_day.split("-")[0] + start_day.split("-")[1]
        )  # Get date in MMDD format
        # Create dir for this day's output files if it doesn't exist
        os.makedirs(self.output_dir + f"{date_str_mmdd}/", exist_ok=True)
        # Create the output filename
        output_filename = (
            self.output_dir
            + f"{date_str_mmdd}/"
            + f"detections_{get_time_of_day_in_minutes(start_time)}-{get_time_of_day_in_minutes(end_time)}.csv"
        )

        # Get the I-24 trajectory data filename for the specified day
        i24_traj_data_filename = f"{self.inception_data_dir}{start_day}.json"

        print(f"\nBuilding detections for {start_time} to {end_time}...\n")

        # Create an empty DataFrame with the desired column names
        self.df = pd.DataFrame(columns=["Detector", "Time", "qPKW", "vPKW"])

        # Cache INCEPTION trajectories for the specified time range, to avoid re-reading the file multiple times later
        self.cached_trajectories = cache_trajectories(
            i24_traj_data_filename, start_time, end_time, verbose, disable_tqdm=self.disable_tqdm
        )

        increment_start_time = start_time
        # Iterate over time window in increments
        for increment_end_time in tqdm(
            self.iterate_time_increments(start_time, end_time, increment_duration), disable=self.disable_tqdm
        ):
            # Print the current time increment
            print(f"\nProcessing {increment_start_time} to {increment_end_time}...")

            # Produce count & speed data for the current time increment
            self.get_detections_for_time_increment(
                increment_start_time,
                increment_end_time,
                i24_traj_data_filename,
                verbose,
            )
            increment_start_time = increment_end_time

        # Save the DataFrame to a CSV file and use ; delimiter for compatibility w/ SUMO's routing methods
        self.df.to_csv(output_filename, index=False, sep=";")

        print(f"\nSaved detection data to {output_filename}.\n")

        # Generate plots for the data quality checks
        self.plot_data_quality_checks(
            date_str_mmdd,
            start_time,
            end_time,
            increment_duration,
        )

        # Empty dictionaries in case we want to run this method again
        self.valid_trajectories_num_dict = {}
        self.time_excluded_data_points_dict = {}
        self.lane_excluded_data_points_dict = {}
        self.max_meters_from_detector_per_increment_dict = {}
        # Initialize these dictionaries with for each detector
        for detector in self.sumo_net_detector_dict.keys():
            self.valid_trajectories_num_dict[detector] = 0
            self.time_excluded_data_points_dict[detector] = 0
            self.lane_excluded_data_points_dict[detector] = 0
            self.max_meters_from_detector_per_increment_dict[detector] = []

    def get_detections_for_time_increment(
        self,
        increment_start_time,
        increment_end_time,
        i24_traj_data_filename,
        verbose=False,
    ):
        """
        Method to get detection data for a specified time increment.

        Args:
            increment_start_time (str): Start time of the time increment. (YYYY-MM-DD HH:MM:SS)
            increment_end_time (str): End time of the time increment. (YYYY-MM-DD HH:MM:SS)
            i24_traj_data_filename (str): Path to the I-24 trajectory data file.
            verbose (bool): Whether to print additional information.

        Returns:
            None: The method modifies the instance's self.df in place.
        """

        # At each time increment, iterate over all detectors (first by eastbound, then by westbound)
        for detector in sorted(self.sumo_net_detector_dict.keys(), key=key_func):

            detector_loc_milemarker = self.sumo_net_detector_dict[detector][
                "milemarker"
            ]
            direction_string = detector.split("-")[1]

            # Get relevant I-24 trajectories for the current time increment, detector, and direction
            valid_trajectories = get_valid_trajectories(
                i24_traj_data_filename,  # Since using cached trajectories, this is not used
                increment_start_time,
                increment_end_time,
                direction_string,
                detector_loc_milemarker,
                verbose,
                cached_trajectories=self.cached_trajectories,  # Use cached trajectories to save time
                disable_tqdm=self.disable_tqdm,
            )

            # Get detection data for the trajectories and sort into lane-level counts and speeds
            lane_counts_and_speeds = self.get_lane_counts_and_speeds(
                valid_trajectories,
                increment_start_time,
                increment_end_time,
                detector_loc_milemarker,
                detector,
                verbose,
            )

            # Once we have the lane-level data, map it to network lanes and format appropriately
            # Note 1: SUMO network lanes (net_lane_idx) are indexed from the outside in, while I-24 data (data_lane_idx) is indexed from the road median out
            # Note 2: We iterate over num_lanes rather than the lane_counts_and_speeds keys to ensure that we include lanes with 0 detections
            for net_lane_idx, data_lane_idx in enumerate(
                reversed(range(self.sumo_net_detector_dict[detector]["num_lanes"]))
            ):
                # Build the ID of the lane-level detector
                net_lane_detector_id = detector + "_" + str(net_lane_idx)
                # Get the key for the lane in the I-24 data lane_counts_and_speeds dictionary (inverse of the former, due to the indexing difference)
                lane_count_key = f"lane_{data_lane_idx}_count"
                lane_speed_key = f"lane_{data_lane_idx}_speed"
                qPKW = (
                    lane_counts_and_speeds[lane_count_key]
                    if lane_count_key in lane_counts_and_speeds
                    else 0
                )
                vPKW = (
                    lane_counts_and_speeds[lane_speed_key]
                    if lane_speed_key in lane_counts_and_speeds
                    else 0.0
                )

                time_of_day_in_minutes = get_time_of_day_in_minutes(increment_end_time, return_float=True)

                # Append the data to the DataFrame
                new_data = {
                    "Detector": net_lane_detector_id,
                    "Time": time_of_day_in_minutes,
                    "qPKW": qPKW,
                    "vPKW": vPKW,
                }
                self.df = pd.concat(
                    [self.df if not self.df.empty else None, pd.DataFrame([new_data])],
                    ignore_index=True,
                )

                if verbose:
                    print(f"\t{new_data}")
            if verbose:
                print(f"\n\n")

    def get_lane_counts_and_speeds(
        self,
        valid_trajectories,
        start_time,
        end_time,
        detector_loc_milemarker,
        detector,
        verbose,
    ):
        """
        Method to extract the lane-level trajectory counts and speeds at the point where I-24 trajectories cross the simulated loop detector for specified time range.
        Also verifies that each included trajectory crosses the detector within the specified time range.

        Args:
            valid_trajectories (list): List of trajectories that are within the desired time range, pass by the loop location, and are in the correct direction.
            start_time (str): Start time of the time range. (YYYY-MM-DD HH:MM:SS)
            end_time (str): End time of the time range. (YYYY-MM-DD HH:MM:SS)
            detector_loc_milemarker (float): Milemarker of the 'detector' location for which we want the data.
            detector (str): The detector ID for which we are extracting data.
            verbose (bool): Whether to print additional information.

        Returns:
            lane_counts_and_speeds (dict): Dictionary containing the counts and speeds of trajectories in each lane.
        """

        start_time_unix = convert_to_cst_unix(start_time)
        end_time_unix = convert_to_cst_unix(end_time)

        # Sanity check to see how close the trajectory is to the detector
        max_meters_from_detector = 0

        # Initialize dictionary to store counts
        lane_counts_and_speeds = {}

        # Initialize a count of excluded data points
        time_excluded_data_points = 0
        lane_excluded_data_points = 0

        for trajectory in valid_trajectories:
            timestamp = get_array(
                trajectory.get("timestamp", None)
            )  # in seconds (unix time)
            x_position = (
                get_array(trajectory.get("x_position", None)) / 5280
            )  # Converting feet to miles
            y_position = get_array(trajectory.get("y_position", None))  # in feet

            # Build dataframe for the current trajectory
            data_index = pd.DataFrame(
                np.column_stack((timestamp, x_position, y_position))
            )
            data_index.columns = ["time", "x", "y"]

            # y-position is negative for eastbound traffic, so take absolute value
            data_index["y"] = abs(data_index["y"])

            # Assign lane number to each point in trajectory
            lane_boundaries = self.lane_boundaries_dict[detector]
            data_index["lane"] = data_index["y"].apply(
                lambda x: np.searchsorted(lane_boundaries, x) - 1
            )

            ## Find the lane location at the point where the trajectory crosses the loop detector
            # Find the row where 'x' is closest to the detector location
            closest_index = abs(data_index["x"] - detector_loc_milemarker).idxmin()

            # Get the distance and time at the closest index
            dist_at_closest_index = abs(
                data_index.loc[closest_index, "x"] - detector_loc_milemarker
            )
            time_at_closest_index = data_index.loc[closest_index, "time"]

            # Ignore trajectory if it passes the detector beyond the selected time range [start_time_unix, end_time_unix) (note the inclusive/exclusive start/end time)
            if (time_at_closest_index < start_time_unix) or (
                time_at_closest_index >= end_time_unix
            ):
                time_excluded_data_points += 1
                continue

            # Get the corresponding 'lane' value
            result_lane = data_index.loc[closest_index, "lane"]
            # Ignore if result_lane is beyond the outermost lane boundaries (remembering that lanes are 0-indexed)
            if (result_lane < 0) or (result_lane >= len(lane_boundaries) - 1):
                lane_excluded_data_points += 1
                continue

            # Find the speed at the at the detector over self.speed_calc_time_window seconds
            #   (Look forward speed_calc_time_window / 2 seconds & look back speed_calc_time_window / 2 seconds)
            speed_at_closest_index = self.calc_speed(data_index, closest_index)

            # Store the maximum distance from the detector for a later check (if it's too far, may indicate data quality issues)
            if dist_at_closest_index > max_meters_from_detector:
                max_meters_from_detector = (
                    dist_at_closest_index * 1609.34
                )  # Converting miles to meters

            # Make labels for dict keys
            result_lane_count_key = "lane_" + str(int(result_lane)) + "_count"
            result_lane_speed_key = "lane_" + str(int(result_lane)) + "_speed"

            # Increment appropriate entry in the lane_counts_and_speeds dictionary with the trajectory
            if result_lane_count_key not in lane_counts_and_speeds.keys():
                lane_counts_and_speeds[result_lane_count_key] = 0
            lane_counts_and_speeds[result_lane_count_key] += 1
            # Create or add to the list of lane speeds
            if result_lane_speed_key not in lane_counts_and_speeds.keys():
                lane_counts_and_speeds[result_lane_speed_key] = []
            lane_counts_and_speeds[result_lane_speed_key].append(speed_at_closest_index)

        # Now iterate over the lists of lane speeds and replace them with the mean speed of the respective list
        for key in lane_counts_and_speeds.keys():
            if "speed" in key:
                lane_counts_and_speeds[key] = round(
                    np.mean(lane_counts_and_speeds[key]), 2
                )

        if verbose:
            print(
                f"\tOf the {len(valid_trajectories)} trajectories, the maximum x-dist from detector is: {max_meters_from_detector:.4f} meters"
            )
            print(
                f"\tExcluded data from {time_excluded_data_points} trajectories that were outside the specified time range."
            )
            print(
                f"\tExcluded data from {lane_excluded_data_points} trajectories that were beyond the outermost lane boundaries at the detector point."
            )
            print(f"\tdetector: {detector}")
            print(f"\tdetector_loc_milemarker: {detector_loc_milemarker}")
            print(f"\tLane counts: {lane_counts_and_speeds}\n")

        # Update the relevant dictionaries
        self.valid_trajectories_num_dict[detector] += len(valid_trajectories)
        self.time_excluded_data_points_dict[detector] += time_excluded_data_points
        self.lane_excluded_data_points_dict[detector] += lane_excluded_data_points
        self.max_meters_from_detector_per_increment_dict[
            detector
        ].append(  # For every (likely 30-sec) time increment, store the max distance from the detector
            round(max_meters_from_detector, 4)
        )

        return lane_counts_and_speeds

    @staticmethod
    def iterate_time_increments(start_time_str, end_time_str, increment_duration):
        """
        Iterates over time increments between two given timestamps, excluding the start time.

        Args:
            start_time_str (str): The start time in the format 'YYYY-MM-DD HH:MM:SS'.
            end_time_str (str): The end time in the format 'YYYY-MM-DD HH:MM:SS'.
            increment_duration (int, optional): The interval between time increments in seconds.

        Yields:
            datetime.datetime: The current time in the iteration, excluding the start time.
        """

        start_time = datetime.datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
        end_time = datetime.datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")

        current_time = start_time + datetime.timedelta(seconds=increment_duration)
        while current_time <= end_time:
            yield current_time.strftime("%Y-%m-%d %H:%M:%S")
            current_time += datetime.timedelta(seconds=increment_duration)

    def calc_speed(self, data_index, closest_index):
        """
        Method to calculate the speed of a vehicle at the point where it crosses the loop detector.

        Args:
            data_index (pd.DataFrame): DataFrame containing the trajectory data, with 'time' col in seconds and 'x' col in miles.
            closest_index (int): Index of the point in the trajectory closest to the detector.

        Returns:
            speed (float): Speed of the vehicle in km/hr.
        """

        # Divide by 2 to get the time look-ahead/-back time (i.e., look forward (time_window / 2) sec, look back (time_window / 2) sec)
        time_window = self.speed_calc_time_window / 2
        # Convert time_window to an index offset
        index_offset = int(time_window / data_index["time"].diff().mean())

        # Handle case when closest_index is at/near the beginning of the data_index
        if closest_index - index_offset < 0:
            start_time, start_loc = data_index.loc[0, "time"], data_index.loc[0, "x"]
        # Otherwise, compute normally
        else:
            start_time, start_loc = (
                data_index.loc[closest_index - index_offset, "time"],
                data_index.loc[closest_index - index_offset, "x"],
            )
        # Handle case when closest_index is at/near the end of the data_index
        if closest_index + index_offset >= len(data_index):
            end_time, end_loc = (
                data_index.loc[len(data_index) - 1, "time"],
                data_index.loc[len(data_index) - 1, "x"],
            )
        # Otherwise, compute normally
        else:
            end_time, end_loc = (
                data_index.loc[closest_index + index_offset, "time"],
                data_index.loc[closest_index + index_offset, "x"],
            )

        time_split = end_time - start_time  # in seconds
        dist_split = abs(
            end_loc - start_loc
        )  # in miles - need abs() since traffic can be eastbound or westbound

        assert time_split > 0, "Time split is negative or zero."
        assert dist_split >= 0, "Distance split is negative."

        # Calculate speed in km/hr
        speed = dist_split / time_split * 3600 * 1.60934

        return speed

    def plot_data_quality_checks(
        self,
        date_str_mmdd,
        start_time,
        end_time,
        increment_duration,
        max_dist_ylim=(0, 3),
        cols=5,
        plot_figsize=(10, 6),
        panel_figsize=(15, 12),
        rotation=75,
        offset = 0.2, 
        width = 0.25,
    ):
        """
        Method to plot data quality checks for the detector data. 
        Focuses on the number of trajectories, excluded data points, and vehicles' maximum distance from the detector at their detection point.

        Args:
            date_str_mmdd (str): Date string in MMDD format.
            start_time (str): Start time of the time range. (YYYY-MM-DD HH:MM:SS)
            end_time (str): End time of the time range. (YYYY-MM-DD HH:MM:SS)
            increment_duration (float): Duration of each time increment (in seconds) for which we get detection data.
            max_dist_ylim (tuple, optional): Y-axis limits (in meters) for the maximum distance from the detector plot. If None, auto-scales y-axes. Default is (0, 3).
            cols (int, optional): Number of columns for the per-detector plots. Default is 5.
            plot_figsize (tuple, optional): Figure size for the bar plots. Default is (10, 6).
            panel_figsize (tuple, optional): Figure size for the panel plots. Default is (15, 12).
            rotation (int, optional): Rotation angle for x-tick labels. Default is 75 degrees.
            offset (float, optional): Offset for the trajectory numbers bar plot. Default is 0.2.
            width (float, optional): Width of each individual trajectory bar in trajectory numbers bar plot. Default is 0.25.

        Returns:
            None: The method saves the plots to the specified output directory.
        """

        # Create dir for this day's plot output files if it doesn't exist
        subdir_path = self.output_dir + f"{date_str_mmdd}/" + self.plots_subdir
        os.makedirs(subdir_path, exist_ok=True)

        # Plot the number of time-valid and time-and-lane-boundary-filtered trajectories per detector
        plt.figure(figsize=(plot_figsize))

        detector_labels = list(self.valid_trajectories_num_dict.keys())
        x_pos = np.arange(len(detector_labels))

        # First plot the unfiltered trajectories
        plt.bar(
            x_pos - offset,
            self.valid_trajectories_num_dict.values(),
            width,
            label="Unfiltered trajectories",
        )

        # Then plot the time-filtered trajectories (slightly offset)
        # Note time-filtration isn't bad; these are a result of time-verification in get_lane_counts_and_speeds()
        time_filtered_values = [
            self.valid_trajectories_num_dict[detector]
            - self.time_excluded_data_points_dict[detector]
            for detector in detector_labels
        ]
        plt.bar(
            x_pos,
            time_filtered_values,
            width,
            label="Time-filtered trajectories\n(Generally time-filtration is required & benign)",
        )

        # Plot the time- & lane-boundary-filtered trajectories (slightly offset in the other direction)
        time_lane_filtered_values = [
            self.valid_trajectories_num_dict[detector]
            - self.time_excluded_data_points_dict[detector]
            - self.lane_excluded_data_points_dict[detector]
            for detector in detector_labels
        ]
        plt.bar(
            x_pos + offset,
            time_lane_filtered_values,
            width,
            label="Time- & lane-boundary-filtered\n(What is used in detection data)",
            # alpha=0.5,  # More transparency for the third bar
        )

        plt.xticks(x_pos, detector_labels, rotation=rotation)

        plt.xlabel("Detector ID")
        plt.ylabel("# of trajectories")
        plt.title(
            "Trajectories per detector (unfiltered & filtered by get_lane_counts_and_speeds())"
        )
        plt.legend()
        # plt.xticks(rotation=rotation)
        plt.tight_layout()
        plt.savefig(
            subdir_path
            + f"{get_time_of_day_in_minutes(start_time)}-{get_time_of_day_in_minutes(end_time)}_num_trajectories.pdf"
        )
        plt.close()

        # Plot the number of lane-excluded data points per detector.
        # These may indicate issues with lane boundaries if they are high relative to trajectories.
        plt.figure(figsize=(plot_figsize))
        plt.bar(
            self.lane_excluded_data_points_dict.keys(),
            self.lane_excluded_data_points_dict.values(),
        )
        plt.xlabel("Detector ID")
        plt.ylabel("# of excluded data points")
        plt.title(
            "Number of trajectories excluded due to passing detector beyond outermost lane boundaries"
        )
        plt.xticks(rotation=rotation)
        plt.tight_layout()
        plt.savefig(
            subdir_path
            + f"{get_time_of_day_in_minutes(start_time)}-{get_time_of_day_in_minutes(end_time)}_lane_excluded_points.pdf"
        )
        plt.close()

        # Plot the fraction of lane-excluded data points per detector
        # These may indicate issues with lane boundaries if they are high.
        plt.figure(figsize=(plot_figsize))
        plt.bar(
            self.lane_excluded_data_points_dict.keys(),
            [
                self.lane_excluded_data_points_dict[detector]
                / (
                    self.valid_trajectories_num_dict[detector]
                    - self.time_excluded_data_points_dict[detector]
                )
                for detector in self.lane_excluded_data_points_dict.keys()
            ],
        )
        plt.ylim(0, 1.0)
        plt.xlabel("Detector ID")
        plt.ylabel("Fraction excluded")
        plt.title("Lane-excluded data points as a fraction of trajectories")
        plt.xticks(rotation=rotation)
        plt.tight_layout()
        plt.savefig(
            subdir_path
            + f"{get_time_of_day_in_minutes(start_time)}-{get_time_of_day_in_minutes(end_time)}_lane_excluded_fraction.pdf"
        )
        plt.close()

        # Plot the maximum distance from the detector per increment as a bar chart subplot
        rows = (
            len(self.sumo_net_detector_dict) + cols - 1
        ) // cols  # Ceiling division to get the number of rows needed

        # Create subplots for max distance from detector
        fig, axes = plt.subplots(rows, cols, figsize=panel_figsize)

        plot_index = 0

        for detector in sorted(self.sumo_net_detector_dict.keys(), key=key_func):
            # Access the current axis in the grid
            ax = axes.flat[plot_index]

            # Plot the data for max distance from detector
            ax.bar(
                list(
                    range(
                        len(self.max_meters_from_detector_per_increment_dict[detector])
                    )
                ),
                self.max_meters_from_detector_per_increment_dict[detector],
            )
            ax.set_title(
                f"{detector}\n({self.valid_trajectories_num_dict[detector] - self.lane_excluded_data_points_dict[detector] - self.lane_excluded_data_points_dict[detector]} filtered trajectories)"
            )
            ax.set_ylabel("Max. distance (m)")
            ax.set_xlabel("Time increment")
            if max_dist_ylim:
                ax.set_ylim(max_dist_ylim)
                # When y-limits are constant across subplots, y-labels are redundant and can be hidden for all but the first column
                if plot_index % cols != 0:
                    ax.yaxis.set_visible(False)

            plot_index += 1

        # Hide any unused subplots
        for i in range(plot_index, rows * cols):
            fig.delaxes(axes.flat[i])

        fig.suptitle(
            f"Maximum detection distance from detector across vehicles per {increment_duration:.0f}-sec. increment\nfrom {start_time} to {end_time}",
            fontsize=16,
        )
        plt.tight_layout()
        plt.savefig(
            subdir_path
            + f"{get_time_of_day_in_minutes(start_time)}-{get_time_of_day_in_minutes(end_time)}_max_distances_from_detectors.pdf"
        )
        plt.close()

        print(f"\nSaved data quality check plots to {subdir_path}.\n")

    def plot_detection_data(
        self, start_time, end_time, qpkw_ylim=None, vpkw_ylim=None, cols=5
    ):
        """
        Method to plot (1) histogram of the counts across all detectors,
            (2) histogram of the speeds across all detectors, and
            (3) per-detector bar charts of lane-level counts and speeds.

        Args:
            start_time (str): Start time of the time range. (YYYY-MM-DD HH:MM:SS). Must correspond to a file that was produced by get_detections() method.
            end_time (str): End time of the time range. (YYYY-MM-DD HH:MM:SS). Must correspond to a file that was produced by get_detections() method.
            qpkw_ylim (tuple, optional): Y-axis limits for the qPKW plots. Default is None, which allows automatic scaling.
            vpkw_ylim (tuple, optional): Y-axis limits for the vPKW plots. Default is None, which allows automatic scaling.
            cols (int, optional): Number of columns for the per-detector plots. Default is 5.

        Returns:
            None: The method saves the plots to the specified output directory.
        """

        # Get date in 'MM-DD' format
        start_day = (
            start_time.split(" ")[0].split("-")[1]
            + "-"
            + start_time.split(" ")[0].split("-")[2]
        )

        # Construct the output_filename (and appropriate directory, if needed)
        date_str_mmdd = (
            start_day.split("-")[0] + start_day.split("-")[1]
        )  # Get date in MMDD format

        # Create the input filename
        detections_filename = (
            self.output_dir
            + f"{date_str_mmdd}/"
            + f"detections_{get_time_of_day_in_minutes(start_time)}-{get_time_of_day_in_minutes(end_time)}.csv"
        )

        # Check if the file exists
        if not os.path.exists(detections_filename):
            raise FileNotFoundError(
                f"File {detections_filename} does not exist. Have you run get_detections() for this time period?"
            )

        # Read the CSV file into a DataFrame. Recall we use ';' as the delimiter for compatibility with SUMO's routing methods
        self.df = pd.read_csv(detections_filename, sep=";")

        # Create dir for this day's plot output files if it doesn't exist
        subdir_path = self.output_dir + f"{date_str_mmdd}/" + self.plots_subdir
        os.makedirs(subdir_path, exist_ok=True)

        # Plot histogram of counts
        plt.hist(self.df["qPKW"], bins=50)
        plt.xlabel("Lane-level counts")
        plt.title(
            f"Distribution of qPKW for all detectors\nfrom {start_time} to {end_time}"
        )
        plt.savefig(
            subdir_path
            + f"{get_time_of_day_in_minutes(start_time)}-{get_time_of_day_in_minutes(end_time)}_qPKW_hist.pdf"
        )
        plt.close()

        # Plot histogram of speeds
        plt.hist(self.df["vPKW"], bins=50)
        plt.xlabel("km/hr")
        plt.title(
            f"Distribution of vPKW for all detectors\nfrom {start_time} to {end_time}"
        )
        plt.savefig(
            subdir_path
            + f"{get_time_of_day_in_minutes(start_time)}-{get_time_of_day_in_minutes(end_time)}_vPKW_hist.pdf"
        )
        plt.close()

        # Plot per-detector lane-level counts and speeds separately
        rows = (
            len(self.sumo_net_detector_dict) + cols - 1
        ) // cols  # Ceiling division to get the number of rows needed

        # Create subplots for counts and speeds
        fig_qpkw, axes_qpkw = plt.subplots(rows, cols, figsize=(15, 12))
        fig_vpkw, axes_vpkw = plt.subplots(rows, cols, figsize=(15, 12))

        # Counter to track current plot position in the grid
        plot_index = 0

        for detector in sorted(self.sumo_net_detector_dict.keys(), key=key_func):
            # Downsize to a DF with only the current detector
            detector_df = self.df[self.df["Detector"].str.contains(detector)]

            # Extract the lane number from the detector ID
            detector_df["orig_lane_number"] = (
                detector_df["Detector"].str.split("_").str[1]
            )

            # Group by lane number and sum the qPKW values and save as new df
            qpkw_df = (
                detector_df.groupby("orig_lane_number")["qPKW"].sum().reset_index()
            )
            # Group by lane number and average the vPKW values and save as new df
            vpkw_df = (
                detector_df.groupby("orig_lane_number")["vPKW"].mean().reset_index()
            )

            # Get the number of lanes found for that detector in the network
            num_lanes_network = self.sumo_net_detector_dict[detector]["num_lanes"]
            assert len(qpkw_df) == len(
                vpkw_df
            ), f"Number of lanes in qPKW and vPKW dataframes do not match for {detector}."
            num_lanes_found = len(qpkw_df)

            # Access the current axis in the grid
            ax_qpkw = axes_qpkw.flat[plot_index]
            ax_vpkw = axes_vpkw.flat[plot_index]

            # Plot the data for counts
            # Reverse the order to match network lane order from drivers' POV
            ax_qpkw.bar(
                list(reversed(qpkw_df["orig_lane_number"])),
                list(reversed(qpkw_df["qPKW"])),
                color=self.lane_colors,
                alpha=1.0,
            )
            ax_qpkw.set_title(
                f"{detector}\n{num_lanes_found} inferred vs. {num_lanes_network} network lanes"
            )
            ax_qpkw.set_ylabel("# of vehicles")
            ax_qpkw.set_xlabel("Network Lane # (0 is outermost lane)")
            if qpkw_ylim:
                ax_qpkw.set_ylim(qpkw_ylim)
                # When y-limits are constant across subplots, y-labels are redundant and can be hidden for all but the first column
                if plot_index % cols != 0:
                    ax_qpkw.yaxis.set_visible(False)

            # Plot the data as above, but for speeds
            # Reverse the order to match network lane order from drivers' POV
            ax_vpkw.bar(
                list(reversed(vpkw_df["orig_lane_number"])),
                list(reversed(vpkw_df["vPKW"])),
                color=self.lane_colors,
                alpha=1.0,
            )
            ax_vpkw.set_title(
                f"{detector}\n{num_lanes_found} inferred vs. {num_lanes_network} network lanes"
            )
            ax_vpkw.set_ylabel("km/hr")
            ax_vpkw.set_xlabel("Network Lane # (0 is outermost lane)")
            if vpkw_ylim:
                ax_vpkw.set_ylim(vpkw_ylim)
                # When y-limits are constant across subplots, y-labels are redundant and can be hidden for all but the first column
                if plot_index % cols != 0:
                    ax_vpkw.yaxis.set_visible(False)

            # Update plot index for the next detector
            plot_index += 1

            # If all plots are created, break the loop
            if plot_index >= rows * cols:
                break

        # Hide any unused subplots
        for i in range(plot_index, rows * cols):
            axes_qpkw.flat[i].axis("off")
            axes_vpkw.flat[i].axis("off")

        # Set the title for the entire figure
        fig_qpkw.suptitle(
            f"Lane-level counts for all detectors from {start_time} to {end_time}",
            fontsize=16,
        )
        fig_vpkw.suptitle(
            f"Lane-level mean speeds for all detectors from {start_time} to {end_time}",
            fontsize=16,
        )

        # Tight layout to avoid overlapping labels
        fig_qpkw.tight_layout()
        fig_vpkw.tight_layout()

        # Save the figures
        fig_qpkw.savefig(
            subdir_path
            + f"{get_time_of_day_in_minutes(start_time)}-{get_time_of_day_in_minutes(end_time)}_detector_lane_counts.pdf"
        )
        fig_vpkw.savefig(
            subdir_path
            + f"{get_time_of_day_in_minutes(start_time)}-{get_time_of_day_in_minutes(end_time)}_detector_lane_speeds.pdf"
        )

        print(f"\nSaved plots to {subdir_path}.\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Process and plot INCEPTION trajectories as aggregate detector data."
    )
    parser.add_argument(
        "--start_time",
        default=start_time,
        help="Start time for processing (YYYY-MM-DD HH:MM:SS).",
    )
    parser.add_argument(
        "--end_time",
        default=end_time,
        help="Start time for processing (YYYY-MM-DD HH:MM:SS).",
    )
    parser.add_argument(
        "--disable_tqdm",
        action="store_true",
        default=disable_tqdm,
        help="Disable tqdm progress bars.",
    )

    args = parser.parse_args()

    # Instantiate the DetectionBuilder class
    detection_builder = DetectionBuilder(
        detector_dict_path,
        lane_boundaries_path,
        inception_data_dir,
        speed_calc_time_window,
        output_dir,
        plots_subdir,
        lane_colors,
        args.disable_tqdm,
    )

    if run_get_detections:
        # Run the detection builder to get detections for the specified time range
        detection_builder.get_detections(
            args.start_time,
            args.end_time,
            increment_duration=increment_duration,
            verbose=verbose,
        )

    if run_plot_detection_data:
        # Plot the data for the specified time range
        detection_builder.plot_detection_data(
            args.start_time, args.end_time, qpkw_ylim=qpkw_ylim, vpkw_ylim=vpkw_ylim
        )
