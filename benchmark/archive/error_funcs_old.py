import os
import csv
import pickle
import sumolib
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from scipy.stats import wasserstein_distance
from datetime import datetime 
from matplotlib import pyplot as plt

# TODO: may ultimately not need this
from utils import get_array, cache_trajectories, get_time_of_day_in_minutes, convert_to_cst_unix


class ErrorCalculator:
    def __init__(
        self,
        cache_traj_start_time="2022-11-25 09:00:00" ,
        cache_traj_end_time="2022-11-25 09:10:00",
        micro_obj_fn_timestamps=[
            "2022-11-25 09:03:00",
            "2022-11-25 09:06:00",
            "2022-11-25 09:09:00",
        ],
        mm_latlon_mapping_path="build_data/mile_marker_layer.csv",
        net_file_path="sim_files/sumo_test.net.xml",
        segment_dict=None,
    ):
        """
        # TODO Clean up and review this entire class.
        mm_latlon_mapping_path: Path to the file used to translate between milemarkers and lat/lon coordinates
        segment_dict: Example {'eastbound': [(59, 60), (60, 61), (61, 62)], 'westbound': [(59, 60), (60, 61), (61, 62)]}

        """

        # Anchor points may or may not be on starting edges (e.g., if segment of interest is interior), so define separately
        self.eb_anchor_mm = 58.8
        self.wb_anchor_mm = 62.8
        self.segment_len = 1.0  # Length of each segment in miles

        if segment_dict is None:
            self.segment_dict = self.gen_segments_by_length(self.eb_anchor_mm, self.wb_anchor_mm, self.segment_len)
        else:
            self.segment_dict = segment_dict

        self.mm_latlon_df = pd.read_csv(mm_latlon_mapping_path)
        self.use_internal_edges = True  # Whether to use internal edges in the network
        self.net = sumolib.net.readNet(net_file_path, withInternal=self.use_internal_edges)

        self.inception_data_dir = "./i24_data/"
        self.cached_traj_dir = "./obj_func/cached_trajectories/"
        self.cached_trajectories = None

        self.min_acceptable_traj_length = (
            10  # at current inception data freq of 25Hz, this is 0.4s
        )
        self.smoothing_length = self.min_acceptable_traj_length

        self.features = ["speed", "y_position", "headway"]
        self.inception_features = [
            "timestamp",
            "x_position",
            "y_position",
            # "length",
            # "coarse_vehicle_class",
        ]

        self.inception_headway_nan_fill_val = (
            -1.0
        )  # -1.0 is what is used in SUMO's --fcd-output
        self.lane_width = 12  # in feet, used for computing headway and lateral position
        self.max_leader_distance = (
            100 * 3.28084
        )  # (m * ft/m) -- in feet, used for computing headway
        self.num_bins = 30
        self.dist_plot_dir = "./obj_func/dist_plots/"
        os.makedirs(self.dist_plot_dir, exist_ok=True)

        self.search_radius = 50 # in meters, used to find neighboring edges

        # Get the full extent of the eb and wb routes
        self.eb_route_start_edge_id = '108162303'
        self.eb_route_end_edge_id = '988591740#1'
        self.wb_route_start_edge_id = '27828382'
        self.wb_route_end_edge_id = '634155175'
        self.eb_route_edge_ids = self.get_route_edge_ids(
            start_edge_id=self.eb_route_start_edge_id,
            end_edge_id=self.eb_route_end_edge_id,
        )
        self.wb_route_edge_ids = self.get_route_edge_ids(
            start_edge_id=self.wb_route_start_edge_id,
            end_edge_id=self.wb_route_end_edge_id,
        )

        eb_anchor_sumo_x, eb_anchor_sumo_y = self.convert_mm_to_sumo_xy(self.eb_anchor_mm)
        self.eb_anchor_edge, self.eb_anchor_pos_on_edge = self.convert_sumo_xy_to_edge_and_pos(eb_anchor_sumo_x, eb_anchor_sumo_y, "eastbound")
        wb_anchor_sumo_x, wb_anchor_sumo_y = self.convert_mm_to_sumo_xy(self.wb_anchor_mm)
        self.wb_anchor_edge, self.wb_anchor_pos_on_edge = self.convert_sumo_xy_to_edge_and_pos(wb_anchor_sumo_x, wb_anchor_sumo_y, "westbound")

        self.fcd_numeric_cols = ['x', 'y', 'angle', 'speed', 'pos', 'slope', 'time']
        self.xml_file_path = "./sim_files/fcd_output.xml"  # Path to the FCD XML file

        # Convert segments to SUMO distances from anchor point
        self.segment_dict_sumo = {}
        for direction in self.segment_dict.keys():
            self.segment_dict_sumo[direction] = [None] * len(self.segment_dict[direction])
            for i, segment in enumerate(self.segment_dict[direction]):
                self.segment_dict_sumo[direction][i] = self.convert_segment_to_sumo_cutoffs(direction, segment)

        ## Inputs to self.obj_fn_micro_component()
        self.cache_traj_start_time = cache_traj_start_time
        self.cache_traj_end_time = cache_traj_end_time
        # TODO: Automate this & Filter out warm-up data
        self.micro_obj_fn_timestamps = micro_obj_fn_timestamps  # Timestamps at which to compute the objective function's microscopic term
        self.figsize = (10, 6)  # Set the figure size for INCEPTION/SUMO feature distribution histograms
        self.alpha = 0.6  # Set the transparency for INCEPTION/SUMO feature distribution histograms
        # TODO: Need to work this w/ macro term and calibrate via hyperband
        self.weight_mapping = {
            "speed": 1.0,
            "headway": 1.0,
            "y_position": 1.0,
        }
        # Get SUMO sim data from the FCD XML file - only load once for entire sim since this can be time-intensive
        # TODO: Consider moving this to the method(s) that need it so only call when needed (Takes about 40 sec)
        # self.fcd_df = self.fcd_xml_to_df(self.xml_file_path)

    def get_route_edge_ids(self, start_edge_id, end_edge_id):
        """
        Function to get an ordered list of edge IDs along each route.
        
        Args:
            start_edge_id (str): The ID of the starting edge.
            end_edge_id (str): The ID of the ending edge.
        
        Returns:
            list: An ordered list of edge IDs representing the route from start to end.
        """
        
        route, _ = self.net.getShortestPath(
            self.net.getEdge(start_edge_id), 
            self.net.getEdge(end_edge_id)
        )

        edge_ids = [edge.getID() for edge in route]

        return edge_ids

    def gen_segments_by_length(
        self, min_val, max_val, segment_len, directions=["eastbound", "westbound"]
    ):
        """
        Generate segments of a given length between min_val and max_val for the specified directions.
        Automatically reverses the order of segments for 'westbound' direction.

        Args:
            min_val (float): The minimum value (MM) of the segment.
            max_val (float): The maximum value (MM) of the segment.
            segment_len (float): The length of each segment.
            directions (list): A list of directions for which to generate segments. Default is ['eastbound', 'westbound'].

        Returns:
            dict: A dictionary with directions as keys and lists of tuples (start, end) as values.
        """

        # Check if max - min is divisible by segment_len
        if (max_val - min_val) % segment_len != 0:
            print(
                "Warning: The range is not evenly divisible by the segment length. Final segment(s) will be shorter than others."
            )

        segment_dict = {}
        # Generate segments for each direction
        for direction in directions:
            segments = []
            current = min_val
            while current < max_val:
                next_val = current + segment_len
                segments.append((current, min(next_val, max_val)))
                current = next_val

            segment_dict[direction] = segments

        # Reverse order of segments for westbound
        if "westbound" in segment_dict:
            segment_dict["westbound"] = [
                (end, start) for start, end in reversed(segment_dict["westbound"])
            ]

        return segment_dict

    def get_inception_feature_dists_df(
        self, timestamp, direction_string, segment, plot_feature_dists=True
    ):
        """
        # TODO: Figure out if want to handle plotting here or when have both SUMO & INCEPTION data
        # TODO: Handle case when no data found (just a len() check should do)
        Function to get the INCEPTION feature distributions (e.g., speed, headway) for a specific segment at a given timestamp.
        Optionally plots them.

        Args:
            timestamp (str): The timestamp 'snapshot' for which to get the inception data. (e.g., "2022-11-23 06:23:00")
            direction_string (str): The direction of the segment ("westbound" or "eastbound")
            segment (tuple): The MM segment of interest (e.g., (59, 60) or (61.1, 61.6)).
            plot_feature_dists (bool): Whether to plot the feature distributions. Default is True.

        Returns:
            pd.DataFrame: A DataFrame with the computed features for the specified segment and timestamp.
                            Each row corresponds to a single vehicle 'snapshot' at the timestamp, with columns for the features of interest.
                            The number of rows corresponds to the number of vehicles in the segment at that time.
        """

        # Filter the trajectories based on the segment, direction, timestamp, and length
        # TODO: This get_valid_inception_trajs call ends up being VERY slow (20 sec to 5 min?) -- find way to speed up or compute in advance! 
        valid_trajectories = self.get_valid_inception_trajs(
            timestamp, direction_string, segment
        )
        output_df, num_excluded_short_trajs = (
            self.compute_features_for_inception_segment(
                valid_trajectories, timestamp, direction_string, segment
            )
        )

        print("Data loaded, now computing features...")

        if plot_feature_dists:
            for feature in self.features:
                if feature in output_df.columns:
                    # Plot the distribution of the feature and save it to a file
                    output_df[feature].plot(
                        kind="hist",
                        title=f"{feature} distribution at {timestamp} for\n{direction_string} {segment}({num_excluded_short_trajs} short trajectories excluded)",
                        xlabel=feature,
                        ylabel="Frequency",
                        bins=self.num_bins,
                    )
                    # Get (or make) date-based directory
                    date_str = timestamp.split()[0].replace("-", "")
                    date_path = os.path.join(self.dist_plot_dir, date_str)
                    os.makedirs(date_path, exist_ok=True)

                    # Create filename
                    time_str = timestamp.split()[1].replace(":", "")
                    direction_abbrv = "wb" if direction_string == "westbound" else "eb"
                    plot_path = os.path.join(
                        date_path,
                        f"{time_str}_{direction_abbrv}-{segment[0]}-{segment[1]}_{feature}-dist_inception.pdf",
                    )

                    plt.tight_layout()
                    plt.savefig(plot_path)
                    plt.close()

        return output_df
    
    def get_cached_trajectories(self, cache_traj_start_time, cache_traj_end_time):
        """
        Function to cache trajectories from INCEPTION for a given time period.
        Saves time by storing trajectories for a given period and saves to a pickle file for future use.

        Args:
            cache_traj_start_time (str): The start time of the cache period in 'YYYY-MM-DD HH:MM:SS' format.
            cache_traj_end_time (str): The end time of the cache period in 'YYYY-MM-DD HH:MM:SS' format.

        Returns:
            list: A list of cached trajectories for the specified time period. Also saves them to a pickle file for future use.
        """

        # Ensure the cache start and end times are on the same day
        assert cache_traj_start_time.split(" ")[0] == cache_traj_end_time.split(" ")[0], (
            "Cache start and end times must be on the same day."
        )
        # Ensure the timestamp occurs between the cache start and end times
        assert (
            cache_traj_start_time < cache_traj_end_time
        ), (
            f"Cache start time ({cache_traj_start_time}) must be before end time ({cache_traj_end_time})."
        )

        # Get date in 'MM-DD' format
        date_str_mmdd = (
            cache_traj_start_time.split(" ")[0].split("-")[1]
            + "-"
            + cache_traj_start_time.split(" ")[0].split("-")[2]
        )

        # Create dir for this day's output files if it doesn't exist
        cache_traj_dir_name = os.path.join(self.cached_traj_dir, date_str_mmdd)
        os.makedirs(cache_traj_dir_name, exist_ok=True)

        # Get the full path for the cached trajectories pickle file
        self.cached_traj_file_path = os.path.join(
            cache_traj_dir_name,
            f"{get_time_of_day_in_minutes(cache_traj_start_time)}_{get_time_of_day_in_minutes(cache_traj_end_time)}.pkl",
        )

        # If the trajectories have already been cached, load them from the file
        if os.path.exists(self.cached_traj_file_path):
            with open(self.cached_traj_file_path, 'rb') as file:
                cached_trajectories = pickle.load(file)
                print(f"Loaded cached trajectories from {self.cached_traj_file_path}")
        # If not, load the trajectories from INCEPTION and cache them
        else:
            # Create full path for the inception trajectory data file
            inception_traj_data_filename = os.path.join(
                self.inception_data_dir,
                f"{date_str_mmdd}.json",
            )
            # Get the trajectories from INCEPTION and cache them
            cached_trajectories = cache_trajectories(
                inception_traj_data_filename,
                cache_traj_start_time,
                cache_traj_end_time,
                verbose=False,
                disable_tqdm=False,
            )
            # Save to a pkl file so we don't have to load them in next time
            with open(self.cached_traj_file_path, 'wb') as f:
                pickle.dump(cached_trajectories, f)
                print(f"Cached trajectories saved to {self.cached_traj_file_path}")

        return cached_trajectories

    def get_valid_inception_trajs(self, timestamp, direction_string, segment):
        """
        Get the valid INCEPTION trajectories for a given timestamp, direction, and segment.
        NOTE: This function only checks trajectory start- and endpoints, so it may include trajectories occur at the correct time and have some overlap with the segment,
            BUT do not actually drive ON the detector AT the correct time (that is, they meet the time and space thresholds independently).
            To address this, self.compute_features_for_inception_segment() method filters the trajectories further to ensure the point(s) of interest is within the time and space thresholds.

        Args:
            timestamp (str): The timestamp to filter the trajectories.
            direction_string (str): segment's direction ("westbound" or "eastbound")
            segment (tuple): The segment to filter the trajectories. (e.g., (59, 60) or (61.1, 61.6))

        Returns:
            list: A list of valid trajectories that meet the criteria.
        """

        # Convert the timestamp to Unix timestamp in CST timezone
        timestamp_unix = convert_to_cst_unix(timestamp)

        # Cache trajectories 
        self.cached_trajectories = self.get_cached_trajectories(self.cache_traj_start_time, self.cache_traj_end_time)

        # Check direction
        assert direction_string in [
            "westbound",
            "eastbound",
        ], f"Invalid direction_string: {direction_string}"

        # -1 for westbound, 1 for eastbound
        direction = -1 if direction_string == "westbound" else 1

        total_trajectories = 0
        valid_trajectories = []
        for trajectory in self.cached_trajectories:

            # Need to get min & max milemarkers for segments & trajs as follows, since they are inverted for eastbound vs westbound
            segment_min_mm = min(segment[0], segment[1])
            segment_max_mm = max(segment[0], segment[1])
            traj_min_mm = min(
                float(trajectory["starting_x"]) / 5280,
                float(trajectory["ending_x"]) / 5280,
            )
            traj_max_mm = max(
                float(trajectory["starting_x"]) / 5280,
                float(trajectory["ending_x"]) / 5280,
            )

            if (
                (float(trajectory["first_timestamp"]) <= timestamp_unix)
                & (float(trajectory["last_timestamp"]) >= timestamp_unix)
                & (int(trajectory["direction"]) == direction)
                & (traj_min_mm <= segment_max_mm)
                & (traj_max_mm >= segment_min_mm)
            ):
                total_trajectories += 1
                valid_trajectories.append(trajectory)

        return valid_trajectories

    def compute_features_for_inception_segment(
        self, valid_trajectories, target_timestamp, direction_string, segment
    ):
        """
        Compute features for a given segment at a specific timestamp using valid trajectories.
        This also applies 2 exclusionary filters:
            1. If the x-position at the timestamp is beyond the segment, exclude the trajectory (veh must be in segment AT timestamp).
            2. If the trajectory has fewer than self.min_acceptable_traj_length points, exclude it. The # of trajs excluded via this condition are tracked.

        Args:
            valid_trajectories (list): List of valid trajectories to compute features from.
            target_timestamp (str): The timestamp at which to compute the features.
            direction_string (str): The direction of the segment ("westbound" or "eastbound")
            segment (tuple): The segment of interest. This function excludes trajectories as described in criteria (1) above

        Returns:
            pd.DataFrame: A DataFrame where each row corresponds to a vehicle 'snapshot' at the timestamp, with columns for the features of interest.
                            The number of rows corresponds to the number of vehicles in the segment at that time.
            num_excluded_short_trajs (int): The number of trajectories excluded for being too short
        """
        # Convert the timestamp to Unix timestamp in CST timezone
        target_timestamp_unix = convert_to_cst_unix(target_timestamp)

        segment_min_mm = min(segment[0], segment[1])
        segment_max_mm = max(segment[0], segment[1])

        # Create a new dataframe to store outputs
        output_df = pd.DataFrame(columns=self.inception_features)

        num_excluded_short_trajs = 0
        for trajectory in valid_trajectories:

            timestamps = get_array(
                trajectory.get("timestamp", None)
            )  # in seconds (unix time)
            x_positions = (
                get_array(trajectory.get("x_position", None)) / 5280
            )  # Converting feet to miles
            y_positions = get_array(trajectory.get("y_position", None))  # in feet
            # veh_length = get_array(trajectory.get("length", None))  # in feet     # TODO: Omitted these since get_array() threw error since not iterable
            # veh_class = get_array(trajectory.get("coarse_vehicle_class", None))   # TODO: Omitted these since get_array() threw error since not decimal
            # Build dataframe for the current trajectory
            traj_df = pd.DataFrame(
                np.column_stack(
                    (timestamps, x_positions, y_positions,)    # TODO: omitted veh_length and veh_class for now
                ),
            )
            traj_df.columns = self.inception_features

            # Get the row where 'time' is closest to the timestamp
            timestamp_index = abs(traj_df["timestamp"] - target_timestamp_unix).idxmin()

            # Filter 1: If at that time, the x-position is beyond the segment, then we exclude it
            x_pos_at_timestamp = traj_df.loc[timestamp_index, "x_position"]
            if (x_pos_at_timestamp < segment_min_mm) or (
                x_pos_at_timestamp > segment_max_mm
            ):
                continue
            # Filter 2: Exclude trajectories with fewer than desired points
            if len(traj_df) < self.min_acceptable_traj_length:
                num_excluded_short_trajs += 1
                continue

            # Calculate the feature(s) of interest
            for feature in self.features:
                if feature == "speed":
                    # Get the instantaneous speed at the timestamp
                    traj_df["inst_speed"] = abs(
                        traj_df["x_position"].diff() / traj_df["timestamp"].diff()
                    )
                    # Backfill for the first row
                    traj_df["inst_speed"] = traj_df["inst_speed"].bfill()
                    # Convert from miles/second to meters/second
                    traj_df["inst_speed"] = traj_df["inst_speed"] * 1609.34
                    # Smooth speed to smooth out the noise
                    traj_df["speed"] = (
                        traj_df["inst_speed"]
                        .rolling(
                            window=self.smoothing_length, center=True, min_periods=1
                        )
                        .mean()
                    )
                if feature == "y_position":
                    # y_position is negative for eastbound traffic, so take absolute value
                    traj_df["y_position"] = abs(traj_df["y_position"])
                    # Avg over the self.min_acceptable_traj_length to smooth out noise
                    traj_df["y_position"] = (
                        traj_df["y_position"]
                        .rolling(
                            window=self.smoothing_length, center=True, min_periods=1
                        )
                        .mean()
                    )

            # Append the single row of the timestamp_index to the output dataframe
            output_df = pd.concat(
                [output_df, traj_df.iloc[[timestamp_index]]], axis=0, ignore_index=True
            )

        # Compute headway feature if requested (couldn't do before, since needed info from all vehicles in the segment)
        if "headway" in self.features:
            output_df = self.compute_inception_headway_feature(
                output_df, direction_string
            )

        return output_df, num_excluded_short_trajs

    def compute_inception_headway_feature(self, df, direction_string):
        """
        Compute the headway feature for the given DataFrame of vehicles in a segment.
        TODO: Handle case where leader is beyond the segment

        Args:
            df (pd.DataFrame): DataFrame containing vehicle data with 'x_position' and 'y_position' columns.
            direction_string (str): Direction of the segment ("westbound" or "eastbound").

        Returns:
            pd.DataFrame: The input DataFrame with an additional 'headway' column containing the computed headway values.
        """

        # Initialize the 'headway' column w/ default val (will be used when leader is nonexistent or too far)
        df["headway"] = self.inception_headway_nan_fill_val

        for ego_idx, ego_row in df.iterrows():
            ego_x = ego_row["x_position"]
            ego_y = ego_row["y_position"]

            min_found_headway = float("inf")  # Initialize with large number
            found_leader_for_ego = False

            # Iterate through all other vehicles to find potential leaders
            for leader_idx, leader_row in df.iterrows():
                if ego_idx == leader_idx:  # A vehicle cannot be its own leader
                    continue

                leader_x = leader_row["x_position"]
                leader_y = leader_row["y_position"]

                # Condition 1: Leader's y_position is within +/- 2 units of the ego vehicle
                y_condition_met = abs(leader_y - ego_y) <= (self.lane_width / 2)

                # Condition 2: Leader is in front of the ego vehicle (greater x_position for eastbound, less for westbound)
                x_front_condition_met = (
                    leader_x > ego_x
                    if direction_string == "eastbound"
                    else leader_x < ego_x
                )

                # Condition 3: Leader's x_position is no more than self.max_leader_distance away from the ego vehicle
                x_close_condition_met = (
                    abs(leader_x - ego_x) <= self.max_leader_distance
                )

                if y_condition_met and x_front_condition_met and x_close_condition_met:
                    dist_to_this_leader = abs(leader_x - ego_x)
                    # If this leader is closer than previously found valid leaders
                    if dist_to_this_leader < min_found_headway:
                        min_found_headway = dist_to_this_leader
                        found_leader_for_ego = True

            # If a leader was found, update the headway for the ego vehicle. Otherwise, it remains as initialized
            if found_leader_for_ego:
                df.loc[ego_idx, "headway"] = min_found_headway

        return df

    def get_sumo_feature_dists_df(self, timestamp, direction_string, segment):
        """
        # TODO: Note that we throw out vehicles on internal edges since we can't get their positions from SUMO (shouldn't be that many)
        
        Function to get the SUMO feature distributions (e.g., speed, headway) for a specific segment at a given timestamp.

        Args:
            timestamp (str): The timestamp 'snapshot' for which to get the SUMO data. (e.g., "2022-11-23 06:23:00")
            direction_string (str): The direction of the segment ("westbound" or "eastbound")
            segment (tuple): The MM segment of interest (e.g., (59, 60) or (61.1, 61.6)).

        Returns:
            pd.DataFrame: A DataFrame with the computed features for the specified segment and timestamp.
        """
        
        # Filter FCD data down to the 'time' closest to the timestamp 
        # Convert the timestamp to seconds in the day (used by SUMO)
        timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        timestamp = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second

        # Find the rows with the 'time' column closest to the timestamp
        closest_time_diff = (self.fcd_df['time'] - timestamp).abs().min()
        snapshot_df = self.fcd_df[(self.fcd_df['time'] - timestamp).abs() == closest_time_diff]
        # Handle the case where multiple times have the same closest_time_diff
        # by simply grabbing the first row with the closest time and use that to filter
        snapshot_df = snapshot_df[snapshot_df['time'] == snapshot_df['time'].iloc[0]]

        # Add a column with the edge IDs
        snapshot_df['edge_id'] = snapshot_df['lane'].apply(lambda x: x.split('_')[0])

        # Only keep vehicles that are on mainline I-24 edges in the correct direction.
        # In effect, this applies 3 filters: (1) only vehicles in correct direction, (2) excludes vehicles on ramps, and (3) excludes vehicles on internal edges
        # Note: We use exclude vehicles on internal edges since sumolib cannot retrieve their positions and thus we can't assign to segments
        valid_edge_ids = self.eb_route_edge_ids if direction_string == "eastbound" else self.wb_route_edge_ids
        snapshot_df = snapshot_df[snapshot_df['edge_id'].isin(valid_edge_ids)]

        # Get the anchor edge and position based on the direction string
        anchor_edge = self.eb_anchor_edge if direction_string == "eastbound" else self.wb_anchor_edge
        anchor_pos_on_edge = self.eb_anchor_pos_on_edge if direction_string == "eastbound" else self.wb_anchor_pos_on_edge
        
        # For each remaining detection, get the driving distance from the anchor point
        snapshot_df['dist_from_anchor'] = snapshot_df.apply(
            lambda row: self.get_dist_between_sumo_positions(
                start_edge=anchor_edge,
                start_pos=anchor_pos_on_edge,
                end_edge=row['edge_id'],
                end_pos=row['pos'],
                use_start_id=False,
                use_end_id=True,
            ),
            axis=1
        )

        # Get the distances for the segment
        segment_start_dist_from_anchor, segment_end_dist_from_anchor = segment
        # Filter the snapshot_df to only include vehicles within the segment distances
        snapshot_df = snapshot_df[snapshot_df['dist_from_anchor'].between(segment_start_dist_from_anchor, segment_end_dist_from_anchor)]

        # TODO: Decide if want to (a) use --lateral-position flag to get lateral position, or (b) compute it ourselves (noting that default sumo lane width != self.lane_width)
        # TODO: How to handle shifting of lanes that we handle via infer_lane_boundaries?
        if 'y_position' in self.features:
            # Extract the lateral position of the vehicle in the lane
            # First, extract the lane index from the lane ID (e.g., 'edge_1_lane_0' -> 0)
            snapshot_df['lane_idx'] = snapshot_df['lane'].apply(lambda x: int(x.split('_')[1]))
            # Get the number of lanes for each edge, which we will need to reverse the lane idx
            snapshot_df['num_lanes'] = snapshot_df['edge_id'].apply(lambda x: self.net.getEdge(x).getLaneNumber())
            # Now reverse the lane index so that the innermost lane is 0 and the outermost lane is num_lanes - 1
            snapshot_df['lane_idx'] = snapshot_df.apply(
                lambda row: row['num_lanes'] - 1 - row['lane_idx'],
                axis=1
            )
            # Now convert the updated lane index to lateral position in feet
            snapshot_df['lat_pos'] = snapshot_df['lane_idx'] * self.lane_width + (self.lane_width / 2)  # Center of the lane in feet
        

        # Keep only the columns of interest
        snapshot_df = snapshot_df[['id', 'time', 'x', 'y', 'dist_from_anchor', 'type', 'speed', 'lat_pos', 'leaderGap']]

        return  snapshot_df

    def convert_segment_to_sumo_cutoffs(self, direction_string, segment):
        """
        Converts a segment defined by mile markers to SUMO distances from an anchor point.

        Args:
            direction_string (str): The direction of the segment ("westbound" or "eastbound").
            segment (tuple): The segment defined by two mile markers (e.g., (59, 60) or (61.1, 61.6)).

        Returns:
            tuple: A tuple containing the distances from the anchor point to the start and end of the segment in meters.
                   The distances are relative to a predefined anchor point (e.g., a mile marker).
        """

        # Get the SUMO coordinates for the start and end of the segment
        segment_start_sumo_x, segment_start_sumo_y = self.convert_mm_to_sumo_xy(segment[0])
        segment_end_sumo_x, segment_end_sumo_y = self.convert_mm_to_sumo_xy(segment[1])

        # Get the edges and positions along those edges for the start and end of the segment
        segment_start_edge, segment_start_pos_on_edge = self.convert_sumo_xy_to_edge_and_pos(
           segment_start_sumo_x, segment_start_sumo_y, direction_string
        )
        segment_end_edge, segment_end_pos_on_edge = self.convert_sumo_xy_to_edge_and_pos(
            segment_end_sumo_x, segment_end_sumo_y, direction_string
        )

        # Get the anchor edge and position based on the direction string
        anchor_edge = self.eb_anchor_edge if direction_string == "eastbound" else self.wb_anchor_edge
        anchor_pos_on_edge = self.eb_anchor_pos_on_edge if direction_string == "eastbound" else self.wb_anchor_pos_on_edge

        # Get distance (in meters) from the anchor MM to segment's starting MM and ending MM
        dist_from_anchor_to_segstart = self.get_dist_between_sumo_positions(start_edge=anchor_edge, start_pos=anchor_pos_on_edge, end_edge=segment_start_edge, end_pos=segment_start_pos_on_edge)
        dist_from_anchor_to_segend = self.get_dist_between_sumo_positions(start_edge=anchor_edge, start_pos=anchor_pos_on_edge, end_edge=segment_end_edge, end_pos=segment_end_pos_on_edge)

        assert dist_from_anchor_to_segstart < dist_from_anchor_to_segend, (
            f"Distance from anchor to segment start ({dist_from_anchor_to_segstart}) should be less than distance to segment end ({dist_from_anchor_to_segend})."
        )

        return (dist_from_anchor_to_segstart, dist_from_anchor_to_segend)

    def convert_sumo_xy_to_edge_and_pos(self, sumo_x, sumo_y, direction_string):
        """
        Gets the nearest edge in the specified direction to the given SUMO coordinates (sumo_x, sumo_y), as well as 
        the position along that edge where the coordinates would be located.

        Args:
            sumo_x (float): The x-coordinate in SUMO coordinates.
            sumo_y (float): The y-coordinate in SUMO coordinates.
            direction_string (str): The direction of the segment ("westbound" or "eastbound").

        Returns:
            tuple: A tuple containing the nearest edge object and the position along that edge.
                   The position is a float representing the offset along the edge in meters.
        """


        edges = self.net.getNeighboringEdges(sumo_x, sumo_y, self.search_radius)
        assert (
            len(edges) > 0
        ), f"\tCould not find any edges within specified search_radius of {self.search_radius}m for ({sumo_x, sumo_y})."
        # Get the nearest edge that is also going the correct direction
        # Iterate over edges, starting with the closest
        nearest_edge = None
        for edge, dist_to_edge in sorted(edges, key=lambda x: x[1]):

            # Only include edges that are part of the mainline I-24 (i.e., exclude ramp edges)
            if (edge.getID() not in self.eb_route_edge_ids) and (edge.getID() not in self.wb_route_edge_ids):
                continue

            # Determine the direction of the edge
            edge_xcoord_delta = (
                edge.getShape()[-1][0] - edge.getShape()[0][0]
            )  # x-coordinate of endpoint minus x-coordinate of start point
            # Check that the edge has a length and is not vertical
            assert (
                edge_xcoord_delta != 0
            ), f"\tEdge {edge.getID()} has no length or is vertical."
            # Extract direction
            edge_direction = "eastbound" if edge_xcoord_delta > 0 else "westbound"

            # If the edge direction is correct, then we have found the nearest edge going the correct direction
            # (Since we are iterating over edges in order of distance, this will be the nearest edge going the correct direction)
            if edge_direction == direction_string:
                nearest_edge = edge
                break

        # If no edge was found, raise an error
        if nearest_edge is None:
            raise ValueError(
                f"No edge found in the specified search radius of {self.search_radius}m for ({sumo_x, sumo_y})."
            )
        
        # Now that we have the correct edge, get the position along that edge
        offset_pos_along_edge, residual_dist = (
            sumolib.geomhelper.polygonOffsetAndDistanceToPoint(
                (sumo_x, sumo_y), nearest_edge.getShape()
            )
        )

        return nearest_edge, offset_pos_along_edge

    def convert_mm_to_sumo_xy(self, mm):
        """
        Converts a mile marker (MM) to SUMO XY coordinates using the mm_latlon_df mapping.
        Note that the mm must be in the mm_latlon_df dataframe, otherwise an error will be raised.
        Args:
            mm (float): The mile marker to convert.
        Returns:
            tuple: A tuple containing the SUMO x and y coordinates (in meters).
        """
        # Check if the mm is in the mm_latlon_df dataframe
        if mm not in self.mm_latlon_df['MM'].values:
            raise ValueError(f"Mile marker {mm} not found in the mm_latlon_df dataframe.")

        (lon, lat) = self.mm_latlon_df[self.mm_latlon_df['MM'] == mm][['X_WGS84', 'Y_WGS84']].values[0]

        sumo_x, sumo_y = self.net.convertLonLat2XY(lon, lat)

        return (sumo_x, sumo_y)
    
    def get_dist_between_sumo_positions(self, start_edge, start_pos, end_edge, end_pos, use_start_id=False, use_end_id=False):
        """
        Function to get the distance between two SUMO positions defined by their edges and positions along those edges.
        Args:
            start_edge (sumolib.net.edge.Edge): The starting edge object.
            start_pos (float): The position along the starting edge in meters.
            end_edge (sumolib.net.edge.Edge): The ending edge object
            end_pos (float): The position along the ending edge in meters.
            use_start_id (bool): Whether to use the start edge ID instead of the edge object. Default is False.
            use_end_id (bool): Whether to use the end edge ID instead of the edge object. Default is False.

        Returns:
            float: The driving distance between the two positions in meters.
        """

        if use_start_id:
            # If using the start edge ID, get the start edge object from the net
            start_edge = self.net.getEdge(start_edge)

        if use_end_id:
            # If using the end edge ID, get the end edge object from the net
            end_edge = self.net.getEdge(end_edge)

        # First get distance (in meters) from start of first edge to the end of the last edge in the route
        route, route_dist = self.net.getShortestPath(start_edge, end_edge)
        # Now subtract (1) length of last edge and (2) position along first edge, then (3) add on the position along last edge.
        dist = route_dist - end_edge.getLength() - start_pos + end_pos

        return dist

    def fcd_xml_to_df(self, xml_file_path):
        """
        Reads an FCD (Floating Car Data) XML file in the SUMO format
        and stores the vehicle data as a Pandas DataFrame.

        Args:
            xml_file_path (str): The path to the FCD XML file.

        Returns:
            pandas.DataFrame: A DataFrame containing the vehicle data from the XML file.
        """
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        all_vehicle_data = []

        for timestep in root.findall('timestep'):
            time = timestep.get('time')
            for vehicle in timestep.findall('vehicle'):
                data = {'time': time}
                data.update(vehicle.attrib)
                all_vehicle_data.append(data)

        df = pd.DataFrame(all_vehicle_data)

        # Convert columns to appropriate data types
        numeric_cols = self.fcd_numeric_cols
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def obj_fn_micro_component(
        self,
        plot_distributions=True,
        print_safety_stat=False, # TODO: Remove this and and substitute w/ separate safety validation metric
        verbose=True,
    ):
        """
        TODO: Akin to Waymo Sim Agents paper, likely need to normalize distributions before computing Wasserstein distances!!!
                WOSAC: "we standardize the NLL computation by fitting histograms to the 32 submitted samples of agent futures, 
                and compute NLLs under the categorical distribution induced by normalizing the histograms."
        Function to compute the microscopic error component of the objective function.
        This function computes the 1-Wasserstein distance (i.e., Earth Mover's Distance) b/w distributions 
        of INCEPTION and SUMO features of interest for each segment at each timestamp.

        Args:
            plot_distributions (bool): Whether to plot the feature distributions for each segment at each timestamp.
                                       Default is True, which will plot the distributions and save them to files.
        Returns:
            float: The total microscopic error, which is the sum of the weighted Wasserstein distances for all features across all segments and timestamps.
        """

        total_micro_error = 0 
        
        for timestamp in self.micro_obj_fn_timestamps:
            for direction in self.segment_dict.keys():
                for segment_idx, i24_segment in enumerate(self.segment_dict[direction]):

                    print(f"Computing microscopic error for timestamp {timestamp}, direction {direction}, segment {i24_segment}...")

                    # Get INCEPTION feature distributions at the timestamp & segment of interest
                    feature_dist_inception = self.get_inception_feature_dists_df(
                        timestamp, direction, i24_segment
                    )

                    # Get SUMO segment corresponding to the I-24 segment
                    sumo_segment = self.segment_dict_sumo[direction][segment_idx]
                    # Get SUMO feature distributions at the timestamp & segment of interest
                    feature_dist_sumo = self.get_sumo_feature_dists_df(
                        timestamp, direction, sumo_segment
                    )

                    for feature in self.features:

                        inception_feature_name = feature
                        sumo_feature_name = "leaderGap" if feature == "headway" else ("lat_pos" if feature == "y_position" else feature)
                        weight_term = self.weight_mapping[feature]

                        # Get 1-Wasserstein distance -- aka Earth Mover's Distance -- between the INCEPTION and SUMO feature distributions
                        wass_dist = wasserstein_distance(
                            feature_dist_inception[inception_feature_name],
                            feature_dist_sumo[sumo_feature_name],
                        )
                        total_micro_error += weight_term * wass_dist

                        if feature == 'headway':
                            if print_safety_stat:
                                # Print the safety statistic for the headway feature
                                print(f"\tSafety statistic for headway at {timestamp}, {direction}:\n{i24_segment} & SUMO ({sumo_segment[0]:.2f}, {sumo_segment[1]:.2f}) | Wasserstein Distance: {wass_dist:.2f}")

                        if plot_distributions:
                            # Plot histograms of headway feature to compare INCEPTION and SUMO
                            plt.figure(figsize=self.figsize)
                            plt.hist(
                                feature_dist_inception[inception_feature_name],
                                bins=self.num_bins, 
                                alpha=self.alpha,
                                label='INCEPTION',
                                color='blue',
                                density=False,
                            )
                            plt.hist(
                                feature_dist_sumo[sumo_feature_name],
                                bins=self.num_bins, 
                                alpha=self.alpha,
                                label='SUMO',
                                color='red',
                                density=False,
                            )
                            plt.xlabel(feature)
                            plt.ylabel("Frequency") # Or "Probability Density" if density=True
                            plt.title(f"Comparison of {feature} distributions at {timestamp}, {direction}:\nINCEPTION {i24_segment} & SUMO ({sumo_segment[0]:.2f}, {sumo_segment[1]:.2f}) | Wasserstein Distance: {wass_dist:.2f}")
                            plt.legend() 

                            # Get (or make) date-based directory
                            date_str = timestamp.split()[0].replace("-", "")
                            date_path = os.path.join(self.dist_plot_dir, date_str)
                            os.makedirs(date_path, exist_ok=True)
                            # Create filename
                            time_str = timestamp.split()[1].replace(":", "")
                            direction_abbrv = "wb" if direction == "westbound" else "eb"
                            plot_path = os.path.join(
                                date_path,
                                f"{time_str}_{direction_abbrv}-{i24_segment[0]}-{i24_segment[1]}_{feature}-dist_both.pdf",
                            )
                            plt.tight_layout()
                            plt.savefig(plot_path)
                            plt.close()
                            
                    if verbose:
                        print(f"Computed microscopic error for timestamp {timestamp}, direction {direction}, segment {i24_segment} (SUMO {sumo_segment})")

        return total_micro_error

    def compute_speed_count_mae_rmse(self, xml_filepath, csv_filepath):
        """
        Computes RMSE and Mean Absolute Error (MAE) for speed and vehicle counts
        by comparing simulated output (XML) with ground truth data (CSV).

        Args:
            xml_filepath (str): Path to the SUMO XML output file.
            csv_filepath (str): Path to the ground truth CSV file.

        Returns:
            dict: A dictionary containing the error metrics:
                {'speed_mae': float, 'speed_rmse': float,
                'count_mae': float, 'count_rmse': float}.
                Returns None if critical file errors occur.
                Metrics will be float('nan') if no valid data points
                are available for comparison for that specific metric.
        """
        
        # Parse CSV data (ground truth)
        # Store in a dictionary for quick lookup: {(detector_id, time_min): {speed, count}}
        csv_data = {}
        with open(csv_filepath, mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            for row in reader:
                csv_data[(row['Detector'], float(row['Time']))] = {'speed': float(row['vPKW']), 'count': int(row['qPKW'])}

        assert len(csv_data) > 0, "No valid data found in CSV file."

        # Parse XML data (simulated) and compare
        speed_differences = []
        count_differences = []
        
        xml_intervals_found = False

        tree = ET.parse(xml_filepath)
        root = tree.getroot()
        
        for interval_element in root.findall('interval'):
            xml_intervals_found = True

            xml_detector_id = interval_element.get('id')
            xml_end_time_sec_str = interval_element.get('end')
            xml_speed_str = interval_element.get('speed')
            xml_counts_str = interval_element.get('nVehContrib')

            if not all([xml_detector_id, xml_end_time_sec_str, xml_speed_str, xml_counts_str]):
                print(f"Warning: Skipping XML interval due to missing attributes: {ET.tostring(interval_element, encoding='unicode').strip()}")
                continue

            xml_end_time_minutes = float(xml_end_time_sec_str) / 60 # Convert from minutes to seconds
            # XML speed can be -1.00 if no vehicles contributed, so we handle that case by setting to 0.0
            xml_speed_val = float(xml_speed_str) if float(xml_speed_str) >= 0 else 0.0
            xml_speed_val = xml_speed_val * 3.6  # Convert from m/s to km/h
            xml_counts_val = int(xml_counts_str)

            # Check if the XML interval matches any entry in the CSV data
            match_key = (xml_detector_id, xml_end_time_minutes)
            if match_key in csv_data:
                save_match_key = match_key
                csv_entry = csv_data[match_key]
                csv_speed_val = csv_entry['speed']
                csv_counts_val = csv_entry['count']

                count_diff = int(xml_counts_val) - int(csv_counts_val)
                count_differences.append(count_diff)

                speed_diff = xml_speed_val - float(csv_speed_val)
                speed_differences.append(speed_diff)

        if not xml_intervals_found or not csv_data:
            print("Warning: No data found in either XML or CSV file.")
            return {'speed_mae': float('nan'), 'speed_rmse': float('nan'),
                    'count_mae': float('nan'), 'count_rmse': float('nan')}
        
        # Calculate and return MAE and RMSE metrics
        results = {
            'speed_mae': float('nan'), 'speed_rmse': float('nan'),
            'count_mae': float('nan'), 'count_rmse': float('nan')
        }
        
        num_speed_points = len(speed_differences)
        assert num_speed_points > 0, "No valid speed data points found for comparison."
        results['speed_mae'] = sum(abs(d) for d in speed_differences) / num_speed_points
        results['speed_rmse'] = np.sqrt(sum(d**2 for d in speed_differences) / num_speed_points)


        num_count_points = len(count_differences)
        assert num_count_points > 0, "No valid count data points found for comparison."
        results['count_mae'] = sum(abs(d) for d in count_differences) / num_count_points
        results['count_rmse'] = np.sqrt(sum(d**2 for d in count_differences) / num_count_points)

        return results

    def obj_fn_macro_component(self, xml_filepath, csv_filepath):
        return self.compute_speed_count_mae_rmse(xml_filepath, csv_filepath)

    def objective_function(self, xml_filepath, csv_filepath, verbose=False):
        # micro_component = self.obj_fn_micro_component()
        macro_component = self.obj_fn_macro_component(xml_filepath, csv_filepath)

        if verbose:
            # print(f"Microscopic Error: {micro_component}")
            print(f"Macroscopic Error: {macro_component}")
        # return micro_component + macro_component
        return macro_component

    def eval_function(self, xml_filepath, csv_filepath, sim_duration, metrics=['GEH', 'throughput', 'speed_iou']):
        if 'GEH' in metrics:
            geh_stat = self.compute_geh_statistic(xml_filepath, csv_filepath, sim_duration)
            print(f"GEH Statistic: {geh_stat:.2f}")
        if 'throughput' in metrics:
            throughput_error = self.compute_throughput_error(xml_filepath, csv_filepath)
            print(f"Throughput Error: {throughput_error:.2f}")
        if 'speed_iou' in metrics:
            iou = self.compute_speed_iou(xml_filepath, csv_filepath)
            print(f"Speed IoU: {iou:.2f}")
        if 'safety' in metrics:
            self.compute_safety_statistic(xml_filepath, csv_filepath)

    def compute_geh_statistic(self, xml_filepath, csv_filepath, sim_duration):
        """
        Function to compute the GEH statistic from a SUMO XML output file and an INCEPTION-derived CSV file.
        Note that it uses self.sim_duration to determine the extrapolation factor for the counts (since GEH is an hourly statistic).

        Args:
            xml_filepath (str): Path to the SUMO XML output file.
            csv_filepath (str): Path to the INCEPTION-derived CSV file.
            sim_duration (int): Duration of the simulation in minutes. This is used to extrapolate the counts to an hourly basis.
                TODO: Move this to make self.configurable

        Returns:
            float: The GEH statistic computed from the merged data.
                Returns NaN if no valid data points are available for comparison.
        """

        # Read in xml as DF
        # Parse the XML file
        tree = ET.parse(xml_filepath)
        root = tree.getroot()

        # List to hold dictionaries, where each dictionary represents a row
        data_list = []

        # Iterate over each 'interval' element in the XML
        for interval_element in root.findall('interval'):
            # Extract only the required attributes
            interval_id = interval_element.attrib.get('id')
            interval_end = interval_element.attrib.get('end')
            interval_nVehContrib = interval_element.attrib.get('nVehContrib')

            # Append to data_list only if the required attributes exist
            data_list.append({
                'id': str(interval_id),  # Ensure id is a string
                'end': float(interval_end) / 60,  # Convert to float and in minutes
                'nVehContrib': float(interval_nVehContrib)  # Convert to float
            })

        # Create a Pandas DataFrame from the list of dictionaries
        xml_df = pd.DataFrame(data_list)

        # Read in csv as DF
        csv_df = pd.read_csv(csv_filepath, delimiter=';', usecols=['Detector', 'Time', 'qPKW'])
        csv_df['Time'] = csv_df['Time'].astype(float)
        csv_df['qPKW'] = csv_df['qPKW'].astype(int)

        # Merge the two dataframes on the relevant columns (e.g., detector ID and time)
        merged_df = pd.merge(csv_df, xml_df, left_on=['Detector', 'Time'], right_on=['id', 'end'], how='inner')
        # Drop id and end columns from the merged dataframe for readability
        merged_df = merged_df.drop(columns=['id', 'end'])

        # Sum the count values over all times at each detector (so we have one row per detector)
        merged_df = merged_df.groupby('Detector').sum().reset_index()
        # Drop the time column since we have aggregated the data
        merged_df = merged_df.drop(columns=['Time'])

        # Drop rows where both qPKW and nVehContrib are zero (GEH undefined when denom is 0)
        merged_df = merged_df[(merged_df['qPKW'] != 0) | (merged_df['nVehContrib'] != 0)]

        extrapolation_factor = 60 / sim_duration  # Assuming sim_duration is in minutes

        # Extrapolate count values out to an hourlong period
        merged_df['qPKW'] = merged_df['qPKW'] * extrapolation_factor  # Assuming the original data was in 10-minute intervals
        merged_df['nVehContrib'] = merged_df['nVehContrib'] * extrapolation_factor  # Assuming the original data was in 10-minute intervals

        # Apply the GEH calculation to each row
        merged_df['GEH'] = np.sqrt(2 * ((merged_df['nVehContrib'] - merged_df['qPKW'])**2) / (merged_df['nVehContrib'] + merged_df['qPKW']))

        # Average the GEH statistic across all detectors
        geh_stat = merged_df['GEH'].mean()

        return geh_stat

    def compute_speed_iou(self, xml_filepath, csv_filepath, threshold = 48):
        """
        Function to compute the Intersection over Union (IoU) for speed data from a SUMO XML output file and an INCEPTION-derived CSV file.
        Assessment is made at each detector-time pair, comparing the speed values from both sources against the specified threshold.
        Assigns a speed of 0 to detectors with no detections, as per the vPKW CSV format.
        
        Args:
            xml_filepath (str): Path to the SUMO XML output file.
            csv_filepath (str): Path to the INCEPTION-derived CSV file.
            threshold (float): Speed threshold in km/h for determining whether a speed is considered "over the threshold".

        Returns:
            float: The Intersection over Union (IoU) statistic computed from the merged data.
                Returns 1.0 if no vehicles were detected in either XML or CSV, otherwise returns 0.0.
                Returns NaN if no valid data points are available for comparison.
        """

        # Read in xml as DF
        tree = ET.parse(xml_filepath)
        root = tree.getroot()

        # List to hold dictionaries, where each dictionary represents a row
        data_list = []

        # Iterate over each 'interval' element in the XML
        for interval_element in root.findall('interval'):
            # Extract only the required attributes
            interval_id = interval_element.attrib.get('id')
            interval_end = interval_element.attrib.get('end')
            interval_nVehContrib = interval_element.attrib.get('nVehContrib')
            interval_speed = interval_element.attrib.get('speed')

            # Append to data_list only if the required attributes exist
            data_list.append({
                'id': str(interval_id),  # Ensure id is a string
                'end': float(interval_end) / 60,  # Convert to float and in minutes
                'nVehContrib': float(interval_nVehContrib),
                'speed': float(interval_speed) * 3.6  if float(interval_speed) >= 0 else 0.0  # Convert speed from m/s to km/h, handle -1.00 case
            })

        # Create a Pandas DataFrame from the list of dictionaries
        xml_df = pd.DataFrame(data_list)

        # Read in csv as DF
        csv_df = pd.read_csv(csv_filepath, delimiter=';', usecols=['Detector', 'Time', 'qPKW', 'vPKW'])
        csv_df['Time'] = csv_df['Time'].astype(float)
        csv_df['qPKW'] = csv_df['qPKW'].astype(int)
        csv_df['vPKW'] = csv_df['vPKW'].astype(float)

        # Merge the two dataframes on the relevant columns (e.g., detector ID and time)
        merged_df = pd.merge(csv_df, xml_df, left_on=['Detector', 'Time'], right_on=['id', 'end'], how='inner')
        # Drop id and end columns from the merged dataframe for readability
        merged_df = merged_df.drop(columns=['id', 'end'])

        # Evaluate whether the speed is over the threshold for both XML and CSV data
        merged_df['xml_speed_over_threshold'] = merged_df['speed'] > threshold
        merged_df['csv_speed_over_threshold'] = merged_df['vPKW'] > threshold

        # Convert to integers
        xml_binary = merged_df['xml_speed_over_threshold'].astype(int)
        csv_binary = merged_df['csv_speed_over_threshold'].astype(int)

        # Calculate TP, FP, FN
        TP = (xml_binary & csv_binary).sum() # Logical AND for intersection
        FP = (~xml_binary & csv_binary).sum() # Not xml_binary AND csv_binary
        FN = (xml_binary & ~csv_binary).sum() # xml_binary AND Not csv_binary

        # Calculate IoU
        denominator = TP + FP + FN
        iou = TP / denominator if denominator != 0 else (1.0 if TP == 0 else 0.0)

        return iou

    def compute_throughput_error(self, xml_filepath, csv_filepath, throughput_detector_groups=["553-westbound", "562-westbound", "565-westbound", "565-eastbound", "562-eastbound", "557-eastbound"]):
        """
        TODO: Will need to wholesale improve this later -- mainly need to just add more detectors at exit throughput points,
                making sure their addition doesn't mess up other calcs, then compare at each exit point. 
                (Currently, I'm doing a crude approximation of this by comparing at various 'chokepoints')
        TODO: Also need to figure out how to handle vehicles still on sim when it ends. 

        Function to compare throughput from a SUMO XML output file and an INCEPTION-derived CSV file.

        Args:
            xml_filepath (str): Path to the SUMO XML output file.
            csv_filepath (str): Path to the INCEPTION-derived CSV file.
            throughput_detector_groups (list): List of detector group IDs to consider for throughput comparison.

        Returns:
            float: The total absolute error in throughput counts across the specified detector groups.
        """
        

        # Read in xml as DF
        tree = ET.parse(xml_filepath)
        root = tree.getroot()

        # List to hold dictionaries, where each dictionary represents a row
        data_list = []

        # Iterate over each 'interval' element in the XML
        for interval_element in root.findall('interval'):
            # Extract only the required attributes
            interval_id = interval_element.attrib.get('id')
            interval_end = interval_element.attrib.get('end')
            interval_nVehContrib = interval_element.attrib.get('nVehContrib')

            # Append to data_list only if the required attributes exist
            data_list.append({
                'id': str(interval_id),  # Ensure id is a string
                'end': float(interval_end) / 60,  # Convert to float and in minutes
                'nVehContrib': float(interval_nVehContrib)  # Convert to float
            })

        # Create a Pandas DataFrame from the list of dictionaries
        xml_df = pd.DataFrame(data_list)

        # Read in csv as DF
        csv_df = pd.read_csv(csv_filepath, delimiter=';', usecols=['Detector', 'Time', 'qPKW'])
        csv_df['Time'] = csv_df['Time'].astype(float)
        csv_df['qPKW'] = csv_df['qPKW'].astype(int)

        # Merge the two dataframes on the relevant columns (e.g., detector ID and time)
        merged_df = pd.merge(csv_df, xml_df, left_on=['Detector', 'Time'], right_on=['id', 'end'], how='inner')

        # Drop id and end columns from the merged dataframe for readability
        merged_df = merged_df.drop(columns=['id', 'end'])

        # Sum the count values over all times at each detector (so we have one row per detector)
        merged_df = merged_df.groupby('Detector').sum().reset_index()
        # Drop the time column since we have aggregated the data
        merged_df = merged_df.drop(columns=['Time'])

        # Extract edge ID from detector ID
        merged_df['detector_group_id'] = merged_df['Detector'].apply(lambda x: x.split('_')[0])  
        throughput_df = merged_df.drop(columns=['Detector'])
        # Group by edge ID and sum the counts
        throughput_df = throughput_df.groupby('detector_group_id').sum().reset_index()

        # Filter the throughput_df to only include the relevant detector groups
        throughput_df = throughput_df[throughput_df['detector_group_id'].isin(throughput_detector_groups)]

        # Compute absolute error
        throughput_df['abs_error'] = abs(throughput_df['nVehContrib'] - throughput_df['qPKW'])

        # Sum error
        total_abs_error = throughput_df['abs_error'].sum()

        return total_abs_error



if __name__ == "__main__":
    # EC = ErrorCalculator(
    #     cache_traj_start_time="2023-10-01 00:00:00",
    #     cache_traj_end_time="2023-10-01 23:59:59",
    # )
    # segment_dict = EC.gen_segments_by_length(
    #     58.8, 62.8, 1, directions=["eastbound", "westbound"]
    # )
    # print(segment_dict)

    # # Get one segment
    # segment = segment_dict["eastbound"][0]
    # print(f"Segment: {segment}")

    # # Get the feature distributions for the segment at a specific timestamp

    EC = ErrorCalculator(
    cache_traj_start_time="2022-11-30 07:40:00" ,
    cache_traj_end_time="2022-11-30 07:50:00",
    micro_obj_fn_timestamps=[
        "2022-11-30 07:43:00",
        "2022-11-30 07:46:00",
        # "2022-11-30 07:49:00",
    ],
    )
    xml_filepath = 'sim_files/e1_output.xml'
    csv_filepath = 'detector_measurements/1130/detections_0460-0470.csv'

    score = EC.objective_function(xml_filepath, csv_filepath, verbose=True)
    EC.eval_function(xml_filepath, csv_filepath, sim_duration=10, metrics=['GEH', 'throughput', 'speed_iou'])