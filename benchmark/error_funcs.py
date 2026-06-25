import os
import csv
import pickle
import sumolib
import logging
import numpy as np
import pandas as pd
from datetime import datetime 
import xml.etree.ElementTree as ET
from scipy.stats import wasserstein_distance
from typing import List, Union
import ijson
from matplotlib import pyplot as plt
from pyproj import CRS, Transformer
from scipy.interpolate import LinearNDInterpolator
import gc
from collections import defaultdict

from utils import get_array, cache_trajectories_for_timestamps, get_time_of_day_in_minutes, convert_to_cst_unix


class MacroscopicErrorCalculator:

    def __init__(self):
        pass

    @staticmethod
    def compute_speed_iou(xml_filepath, csv_filepath, threshold = 48):
        """
        Function to compute the Intersection over Union (IoU) for speed data from a SUMO XML output file and an INCEPTION-derived CSV file.
        Assessment is made at each detector-time pair, comparing the speed values from both sources against the specified threshold.
        Assigns a speed of 0 to detectors with no detections, as per the vPKW CSV format.
        
        Args:
            xml_filepath (str): Path to the SUMO detector XML output file.
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
                'end': round(float(interval_end) / 60 * 2) / 2,  # Convert to minutes and round to nearest 0.5 minute
                'nVehContrib': float(interval_nVehContrib),
                'speed': float(interval_speed) * 3.6  if float(interval_speed) >= 0 else 0.0  # Convert speed from m/s to km/h, handle -1.00 case
            })

        # Create a Pandas DataFrame from the list of dictionaries
        xml_df = pd.DataFrame(data_list)
        # Just in case, dedup on (id, end) pairs, keeping the first occurrence
        # Note: This shouldn't be necessary since detectors only (by default) output every 30 seconds, but safety first
        xml_df = xml_df.drop_duplicates(subset=['id', 'end'], keep='first')

        # Read in csv as DF
        csv_df = pd.read_csv(csv_filepath, delimiter=';', usecols=['Detector', 'Time', 'qPKW', 'vPKW'])
        csv_df['Time'] = csv_df['Time'].astype(float)
        csv_df['qPKW'] = csv_df['qPKW'].astype(int)
        csv_df['vPKW'] = csv_df['vPKW'].astype(float)

        # Merge the two dataframes on the relevant columns (e.g., detector ID and time)
        merged_df = pd.merge(csv_df, xml_df, left_on=['Detector', 'Time'], right_on=['id', 'end'], how='inner')
        # Drop id and end columns from the merged dataframe for readability
        merged_df = merged_df.drop(columns=['id', 'end'])

        if len(merged_df) == 0:
            logging.info("No matching detector-time pairs found between XML and CSV data.")
            return 0.0 # Assign low value to discourage this solution.

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
    
    @staticmethod
    def compute_tot_det_count_mae(xml_filepath, csv_filepath, detectors_to_omit=None, start_minute=None, verbose=False):
        """
        Context: The 30-sec counts at all detectors can be thought of as an (incomplete) proxy for the total number
        of vehicles in the system at a given time (a 'snapshot count'). This function assesses the mean error in these
        snapshot counts between the simulated and ground truth data. 

        Function to compute the mean absolute error (MAE) for total vehicle counts 
        from a SUMO XML output file and an INCEPTION-derived CSV file, where the absolute error
        is taken as the difference between the total counts across all detectors AT A GIVEN TIME in both 
        simulated and ground truth data. 
        The mean is thus computed across all time intervals.
        
        Args:
            xml_filepath (str): Path to the SUMO detector XML output file.
            csv_filepath (str): Path to the INCEPTION-derived CSV file.
            detectors_to_omit (list, optional): List of detector IDs (str) to exclude from the error calculation.
            start_minute (float, optional): If provided, adjusts the time in the XML data by adding this value (in minutes).
            verbose (bool): If True, prints the count differences for each time snapshot.

        Returns:
            float: The absolute error (AE) computed from the merged data.
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
                'end': round(float(interval_end) / 60 * 2) / 2,  # Convert to minutes and round to nearest 0.5 minute
                'nVehContrib': float(interval_nVehContrib),
                'speed': float(interval_speed) * 3.6  if float(interval_speed) >= 0 else 0.0  # Convert speed from m/s to km/h, handle -1.00 case
            })

        # Create a Pandas DataFrame from the list of dictionaries
        xml_df = pd.DataFrame(data_list)

        # Just in case, dedup on (id, end) pairs, keeping the first occurrence
        # Note: This shouldn't be necessary since detectors only (by default) output every 30 seconds, but safety first
        xml_df = xml_df.drop_duplicates(subset=['id', 'end'], keep='first')

        if start_minute is not None:
            # Adjust the 'end' column to account for the starting time
            xml_df['end'] = xml_df['end'] + start_minute

        # Read in csv as DF
        csv_df = pd.read_csv(csv_filepath, delimiter=';', usecols=['Detector', 'Time', 'qPKW', 'vPKW'])
        csv_df['Time'] = csv_df['Time'].astype(float)
        csv_df['qPKW'] = csv_df['qPKW'].astype(int)
        csv_df['vPKW'] = csv_df['vPKW'].astype(float)

        # Merge the two dataframes on the relevant columns (e.g., detector ID and time)
        merged_df = pd.merge(csv_df, xml_df, left_on=['Detector', 'Time'], right_on=['id', 'end'], how='inner')
        # Drop id and end columns from the merged dataframe for readability
        merged_df = merged_df.drop(columns=['id', 'end'])
        assert len(merged_df) > 0, "No valid data found in the merged DataFrame. Check XML and CSV files."

        # If specified, omit certain detectors from the error calculation (e.g., known faulty detectors)
        if detectors_to_omit:
            pattern = '|'.join(detectors_to_omit)
            merged_df = merged_df[~merged_df['Detector'].str.contains(pattern, regex=True)]

        # Group by time and sum the counts
        merged_df = merged_df[['Time', 'qPKW', 'nVehContrib']].groupby('Time').sum().reset_index()
        # Compute the error (AE) between the counts
        merged_df['count_diff'] = merged_df['qPKW'] - merged_df['nVehContrib']
        if verbose:
            print(f"Count difference snapshots: (time, diff): {dict(zip(merged_df['Time'], merged_df['count_diff']))}")
        # Calculate the absolute count difference
        merged_df['abs_count_diff'] = abs(merged_df['count_diff'])
        mae = merged_df['abs_count_diff'].mean()  # Mean of absolute differences

        return mae

    @staticmethod
    def compute_speed_count_mae_rmse(xml_filepath, csv_filepath, detectors_to_omit_from_counts=[], detectors_to_omit_from_speeds=[]):
        """
        Computes RMSE and Mean Absolute Error (MAE) for speed and vehicle counts
        by comparing simulated output (XML) with ground truth data (CSV) at EACH detector-time pair.

        Args:
            xml_filepath (str): Path to the SUMO detector XML output file.
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

        # Initialize lists to hold differences for speed and counts
        speed_differences = []
        count_differences = []

        # Parse XML data (simulated) and compare        
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

            # CH NOTE: Had to add 420 here for bilevel run since detector times are offset by 420 minutes (I believe)
            # TODO: Should revert and handle in a smarter way. 
            xml_end_time_minutes = round(float(xml_end_time_sec_str) / 60 * 2) / 2 + 420 # Convert to minutes and round to nearest 0.5 minute
            # XML speed can be -1.00 if no vehicles contributed, so we handle that case by setting to 0.0
            xml_speed_val = float(xml_speed_str) if float(xml_speed_str) >= 0 else 0.0
            xml_speed_val = xml_speed_val * 3.6  # Convert from m/s to km/h
            xml_counts_val = int(xml_counts_str)

            # Check if the XML interval matches any entry in the CSV data
            match_key = (xml_detector_id, xml_end_time_minutes)
            if match_key in csv_data:
                csv_entry = csv_data[match_key]
                csv_speed_val = csv_entry['speed']
                csv_counts_val = csv_entry['count']

                # First check if this detector should be omitted from count errors (e.g., known faulty detectors)
                if xml_detector_id.split('_')[0] not in detectors_to_omit_from_counts:
                    count_diff = int(xml_counts_val) - int(csv_counts_val)
                    count_differences.append(count_diff)

                # Now check if this detector should be omitted from speed errors (e.g., known faulty detectors)
                if xml_detector_id.split('_')[0] not in detectors_to_omit_from_speeds:
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
        if num_speed_points > 0:
            results['speed_mae'] = sum(abs(d) for d in speed_differences) / num_speed_points
            results['speed_rmse'] = np.sqrt(sum(d**2 for d in speed_differences) / num_speed_points)
        else:
            logging.info("Warning: No valid speed data points found for comparison.")
            results['speed_mae'] = 15.0 # Assign high error to discourage this solution.
            results['speed_rmse'] = 15.0 # Assign high error to discourage this solution.


        num_count_points = len(count_differences)
        import ipdb; ipdb.set_trace() # NOTE: See modification above made for bilevel
        if num_count_points > 0:
            results['count_mae'] = sum(abs(d) for d in count_differences) / num_count_points
            results['count_rmse'] = np.sqrt(sum(d**2 for d in count_differences) / num_count_points)
        else:
            logging.info("Warning: No valid count data points found for comparison.")
            results['count_mae'] = 15.0 # Assign high error to discourage this solution.
            results['count_rmse'] = 15.0 # Assign high error to discourage this solution.

        return results
    

class MicroscopicErrorCalculator:
    """
    Class to compute microscopic error metrics by comparing SUMO FCD data with INCEPTION trajectory data.
    Currently supports headway distribution comparisons using Wasserstein distance.
    """

    def __init__(self,
                 micro_obj_fn_timestamps,
                 fcd_xml_file_path,
                 inception_data_dir,
                 net_file_path,
                 mm_latlon_mapping_path,
                 cached_traj_dir,
                 segment_dict=None,
                 cache_traj_verbosity=False,
                 cache_traj_disable_tqdm=True
                 ):
        """
        Initializes the MicroscopicErrorCalculator with network and segment information.

        Args:
            micro_obj_fn_timestamps (list): List of timestamps (str) at which to evaluate the microscopic error.
                Ex: ["2022-11-21 07:30:00", "2022-11-21 07:45:00", ...]
            fcd_xml_file_path (str): Path to the SUMO FCD XML file.
            inception_data_dir (str): Directory containing INCEPTION trajectory CSV files.
            net_file_path (str): Path to the SUMO network file.
            mm_latlon_mapping_path (str): Path to the CSV file mapping mile markers to lat/lon coordinates.
            cached_traj_dir (str): Directory to store/load cached INCEPTION trajectories for faster processing.
            segment_dict (dict): Optional dictionary defining segments for analysis. If None, default segments are generated.
            cache_traj_verbosity (bool): If True, enables verbose output during trajectory caching.
            cache_traj_disable_tqdm (bool): If True, disables progress bars during trajectory caching.
        """

        # List of timestamps at which to evaluate the microscopic error
        self.micro_obj_fn_timestamps = micro_obj_fn_timestamps

        # Network info
        self.eb_route_start_edge_id = '108162303'
        self.eb_route_end_edge_id = '988591740#1'
        self.wb_route_start_edge_id = '27828382'
        self.wb_route_end_edge_id = '634155175'
        self.search_radius = 50 # in meters, used to find neighboring edges
        self.use_internal_edges = True  # Whether to use internal edges in the network
        self.net = sumolib.net.readNet(net_file_path, withInternal=self.use_internal_edges)

        # SUMO FCD processing params
        self.fcd_xml_file_path = fcd_xml_file_path  # Path to the FCD XML file
        self.fcd_numeric_cols = ['x', 'y', 'angle', 'speed', 'pos', 'slope', 'time', 'leaderGap']

        # INCEPTION data and processing params
        self.inception_data_dir = inception_data_dir
        self.cached_traj_dir = cached_traj_dir
        self.inception_features = [
            "timestamp",
            "x_position",
            "y_position",
        ]
        self.min_acceptable_traj_length = 10  # at current inception data freq of 25Hz, this is 0.4s

        # Params for caching trajectories
        self.cache_traj_verbosity = cache_traj_verbosity
        self.cache_traj_disable_tqdm = cache_traj_disable_tqdm

        # Headway feature computation params
        self.headway_nan_fill_val = -1.0  # -1.0 is what is used in SUMO's --fcd-output; currently, these are dropped when computing headway dist
        self.lane_width = 12  # in feet, used for computing headway
        self.max_leader_distance = 100 * 0.000621371  # (m * miles/m) -- in miles, used for computing headway
        
        # Wasserstein distance params
        self.weight_term = 1.0 # Using uniform 1.0 weights, but could implement alt schemes here

        # Segment params & info
        # Anchor points may or may not be on starting edges (e.g., if segment of interest is interior), so define separately
        self.eb_anchor_mm = 58.8
        self.wb_anchor_mm = 62.8
        self.segment_len = 1.0  # Length of each segment in miles
        self.mm_latlon_df = pd.read_csv(mm_latlon_mapping_path)
        
        # Visualization params
        self.dist_plot_dir = "./cal_methods/sumo_baseline/headway_dist_plots/" # TODO parameterize this
        self.num_bins = 30
        self.figsize = (10, 6)  # Set the figure size for INCEPTION/SUMO feature distribution histograms
        self.alpha = 0.6  # Set the transparency for INCEPTION/SUMO feature distribution histograms
        
        # Initialize MM segment dictionary
        if segment_dict is None:
            self.segment_dict = self.gen_segments_by_length(self.eb_anchor_mm, self.wb_anchor_mm, self.segment_len)
        else:
            self.segment_dict = segment_dict

        # Get the full extent of the eb and wb routes
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

        # Initialize SUMO segment dictionary
        # Convert segments to SUMO distances from anchor point
        self.segment_dict_sumo = {}
        for direction in self.segment_dict.keys():
            self.segment_dict_sumo[direction] = [None] * len(self.segment_dict[direction])
            for i, segment in enumerate(self.segment_dict[direction]):
                self.segment_dict_sumo[direction][i] = self.convert_segment_to_sumo_cutoffs(direction, segment)

    def fcd_xml_to_df(self, fcd_xml_file_path: str, target_times: Union[str, List[str]]):
        """
        Reads a SUMO FCD XML file and returns a DataFrame containing vehicle data at specified timestamps.
        Given the potential size of FCD files, this function uses iterparse for efficiency.

        Args:
            fcd_xml_file_path (str): The path to the FCD XML file.
            target_times (Union[str, List[str]]): The specific timestamps to extract data for.

        Returns:
            pandas.DataFrame: A DataFrame containing the vehicle data at the specified timestamps.
        """
        if isinstance(target_times, str):
            target_times = [target_times]

        # Convert target_times to seconds in the day (used by SUMO in FCD file)
        for i, timestamp in enumerate(target_times):
            timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            timestamp = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
            assert type(timestamp) == int, "Timestamp must be an integer representing seconds in the day."
            target_times[i] = f"{timestamp:.2f}"

        # Grab the latest target time to know when we can stop parsing
        last_target_time = sorted([float(t) for t in target_times])[-1]

        # Use iterparse with start event for efficiency
        context = ET.iterparse(fcd_xml_file_path, events=('start', 'end'))
        
        all_vehicle_data = []

        print(f"Extracting SUMO sim traj data from {fcd_xml_file_path} for target times: {target_times}")
        
        for event, elem in context:
            if elem.tag == 'timestep' and event == 'end':
                current_time = elem.get('time')

                # NOTE: Assumes step_length is <= 1 second (if it's larger, we may miss target times)
                if f"{round(float(current_time)):.2f}" in target_times:
                    # Found a target timestep, extract its vehicle data
                    for vehicle_elem in elem.findall('vehicle'):
                        data = {'time': current_time, 'rounded_time': round(float(current_time))}
                        data.update(vehicle_elem.attrib)
                        all_vehicle_data.append(data)
            
                # Given possible FCD file size, clear this for memory management
                elem.clear()  # Only clear after processing the children

        if not all_vehicle_data:
            raise ValueError(f"No data found for the specified timestamps: {target_times}")

        # This DF has all FCD data at or around the target timestamps
        df = pd.DataFrame(all_vehicle_data)

        ### Due to variable step_length and warmup_time parameters, this the timestamps in this DF will often NOT perfectly align with the inception timestamps.
        ### We can't simply round, as that could alias multiple timestamps to one (e.g., if step length is 0.1s), resulting in duplicate data that biases the error metric.
        ### So for each target_time we have to grab data from (i) ONE timestep that is (ii) closest to the target_time, (iii) tiebreaking where required
        # Convert time column to float to enable comparisons
        df['time'] = df['time'].astype(float)
        # Calculate the absolute difference between 'time' and 'rounded_time'
        df['diff'] = (df['time'] - df['rounded_time']).abs()
        # Find the indices of the rows to keep by grouping by 'rounded_time' and then finding the indices of the minimum 'diff' for each group.
        # `idxmin()` automatically handles tiebreaking [(iii) above]: it returns the index of the first occurrence of the min val
        idxs = df.groupby('rounded_time')['diff'].idxmin()
        # select the 'time' values we want to keep
        closest_times = df.loc[idxs][['rounded_time', 'time']]

        # Merge back in to filter for all rows that match the criteria.
        df = pd.merge(df, closest_times, on=['rounded_time', 'time'])

        # Convert columns to appropriate data types
        numeric_cols = self.fcd_numeric_cols
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def get_cached_inception_trajectories(self, timestamp_list):
        """
        Function to cache trajectories from INCEPTION that overlap with any of the specified timestamps in the timestamp_list.
        Saves time by storing trajectories for a given set of timestamps and saves to a pickle file for future use.

        Args:
            timestamp_list (list): A list of timestamps for which to cache trajectories.

        Returns:
            list: A list of cached trajectories for the specified timestamps. Also saves them to a pickle file for future use.
        """

        # Ensure all timestamps in the list are on the same day
        for timestamp in timestamp_list:
            assert timestamp.split(" ")[0] == timestamp_list[0].split(" ")[0], (
                "All timestamps must be on the same day."
            )


        # Get date in 'MM-DD' format
        date_str_mmdd = (
            timestamp_list[0].split(" ")[0].split("-")[1]
            + "-"
            + timestamp_list[0].split(" ")[0].split("-")[2]
        )

        # Create dir for this day's output files if it doesn't exist
        cached_traj_dir_name = os.path.join(self.cached_traj_dir, date_str_mmdd)
        os.makedirs(cached_traj_dir_name, exist_ok=True)

        # Create full path for the cached trajectories file by using all timestamps in the filename
        cached_traj_fname = "-".join([get_time_of_day_in_minutes(t) for t in timestamp_list]) + ".pkl"
        self.cached_traj_file_path = os.path.join(cached_traj_dir_name, cached_traj_fname)

        # If the trajectories have already been cached, load them from the file
        if os.path.exists(self.cached_traj_file_path):
            with open(self.cached_traj_file_path, 'rb') as file:
                cached_trajectories = pickle.load(file)
                print(f"Loaded cached INCEPTION trajectories from {self.cached_traj_file_path}")
        # If not, load the trajectories from INCEPTION and cache them
        else:
            # Create full path for the inception trajectory data file
            inception_traj_data_filename = os.path.join(
                self.inception_data_dir,
                f"{date_str_mmdd}.json",
            )
            # Get the trajectories from INCEPTION and cache them
            cached_trajectories = cache_trajectories_for_timestamps(
                inception_traj_data_filename,
                timestamp_list,
                verbose=self.cache_traj_verbosity,
                disable_tqdm=self.cache_traj_disable_tqdm,
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
            To address this, self.process_and_filter_inception_trajectories() method filters the trajectories further to ensure the point(s) of interest is within the time and space thresholds.

        Args:
            timestamp (str): The timestamp to filter the trajectories.
            direction_string (str): segment's direction ("westbound" or "eastbound")
            segment (tuple): The segment to filter the trajectories. (e.g., (59, 60) or (61.1, 61.6))

        Returns:
            list: A list of valid trajectories that meet the criteria.
        """

        # Convert the timestamp to Unix timestamp in CST timezone
        timestamp_unix = convert_to_cst_unix(timestamp)

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
    
    def compute_headway_wasserstein_metric(self, plot_distributions=False):
        """
        Computes the Wasserstein distances between headway distributions from INCEPTION and SUMO for each segment at each timestamp.
        Averages the distances (or weighted averages, if desired) to get a final metric.
        Optionally plots the distributions for visual comparison.

        Args:
            plot_distributions (bool): Whether to plot the headway distributions for visual comparison. Default is False.
        """
        
        # (1) Load INCEPTION trajectories for all timestamps of interest
        #     It draws these from a preloaded pkl file if it exists, otherwise it loads from the raw INCEPTION data and saves to a pkl file for future use
        self.cached_trajectories = self.get_cached_inception_trajectories(self.micro_obj_fn_timestamps)

        # (2) Load trajectory data from the SUMO sim output (FCD xml file)
        self.fcd_df = self.fcd_xml_to_df(self.fcd_xml_file_path, self.micro_obj_fn_timestamps.copy())

        # (3) For each segment at each timestamp, compute Wasserstein distance between 
        #     the two distributions and then compute the average (or weighted avg, if desired)
        tot_wass_dist = 0
        num_wass_dists = 0

        # For each timestamp and segment, build the SUMO headway distribution
        for timestamp in self.micro_obj_fn_timestamps:
            for direction in self.segment_dict.keys():
                for segment_idx, i24_segment in enumerate(self.segment_dict[direction]):
                    
                    ## Build the INCEPTION headway distribution for this timestamp and segment
                    # From the cached trajectories, get those that overlap with the timestamp and segment of interest
                    valid_inception_trajectories = self.get_valid_inception_trajs(
                        timestamp, direction, i24_segment
                    )
                    # Filter trajectories to drop shorties and ensure they are in the segment AT the timestamp of interest, then process into a DF
                    inception_traj_df, num_excluded_short_trajs = self.process_and_filter_inception_trajectories(
                        valid_inception_trajectories, timestamp, i24_segment
                    )
                    # Compute the headway feature for the filtered INCEPTION trajectories
                    inception_headway_dist = self.compute_inception_headway_feature(
                        inception_traj_df, direction
                    )

                    ## Build the SUMO headway distribution for this timestamp and segment
                    # Get the SUMO segment in SUMO distances from anchor points
                    sumo_segment = self.segment_dict_sumo[direction][segment_idx]
                    # Compute the SUMO feature distributions at the timestamp & segment of interest
                    sumo_headway_dist = self.build_sumo_headway_dist(timestamp, direction, sumo_segment)

                    # Drop -1 values
                    inception_headway_dist = inception_headway_dist[inception_headway_dist['headway'] != self.headway_nan_fill_val]
                    sumo_headway_dist = sumo_headway_dist[sumo_headway_dist['leaderGap'] != self.headway_nan_fill_val]
                    
                    # If either distribution is empty, impute dist to be [0].
                    if inception_headway_dist["headway"].empty:
                        inception_headway_dist.loc[0] = {'headway': 0}
                    if sumo_headway_dist["leaderGap"].empty:
                        sumo_headway_dist.loc[0] = {'leaderGap': 0}
                        
                    # Get 1-Wasserstein distance -- aka Earth Mover's Distance -- between the INCEPTION and SUMO feature distributions
                    wass_dist = wasserstein_distance(
                        inception_headway_dist["headway"],
                        sumo_headway_dist["leaderGap"],
                    )

                    # Update values appropriately
                    tot_wass_dist += self.weight_term * wass_dist
                    num_wass_dists += 1

                    # If desired, plot distributions
                    if plot_distributions:
                        # Plot histograms of headway feature to compare INCEPTION and SUMO
                        plt.figure(figsize=self.figsize)
                        plt.hist(
                            inception_headway_dist["headway"],
                            bins=self.num_bins, 
                            alpha=self.alpha,
                            label='INCEPTION',
                            color='blue',
                            density=False,
                        )
                        plt.hist(
                            sumo_headway_dist["leaderGap"],
                            bins=self.num_bins, 
                            alpha=self.alpha,
                            label='SUMO',
                            color='red',
                            density=False,
                        )
                        plt.xlabel("Headway")
                        plt.ylabel("Frequency") # Or "Probability Density" if density=True
                        plt.title(f"Comparison of headway distributions at {timestamp}, {direction}:\nINCEPTION {i24_segment} & SUMO ({sumo_segment[0]:.2f}, {sumo_segment[1]:.2f}) | Wasserstein Distance: {wass_dist:.2f}")
                        plt.legend() 

                        # Get (or make) date-based directory
                        date_str = timestamp.split()[0]
                        date_path = os.path.join(self.dist_plot_dir, date_str)
                        os.makedirs(date_path, exist_ok=True)
                        # Create filename
                        time_str = timestamp.split()[1].replace(":", "-")
                        direction_abbrv = "wb" if direction == "westbound" else "eb"
                        plot_path = os.path.join(
                            date_path,
                            f"{time_str}_{direction_abbrv}-{i24_segment[0]}-{i24_segment[1]}_headway-dist_both.pdf",
                        )
                        plt.tight_layout()
                        plt.savefig(plot_path)
                        plt.close()

        # Get average Wass Dist
        avg_wass_dist = tot_wass_dist / num_wass_dists

        return avg_wass_dist
    
    def process_and_filter_inception_trajectories(
        self, valid_trajectories, target_timestamp, segment
    ):
        """
        Function to filter INCEPTION trajectories for a given segment at a specific timestamp and process into a DataFrame
        with the snapshot locations of vehicles at the desired time on that segment.
        This applies 2 exclusionary filters:
            1. If the x-position at the timestamp is beyond the segment, exclude the trajectory (veh must be in segment AT timestamp).
            2. If the trajectory has fewer than self.min_acceptable_traj_length points, exclude it. The # of trajs excluded via this condition are tracked.

        Args:
            valid_trajectories (list): List of valid trajectories to compute features from.
            target_timestamp (str): The timestamp at which to compute the features.
            segment (tuple): The segment of interest. This function excludes trajectories as described in criteria (1) above

        Returns:
            pd.DataFrame: A DataFrame where each row corresponds to a vehicle's 'snapshot' at the target_timestamp (timestamp, x_position, y_position).
                            The number of rows corresponds to the number of vehicles in the segment at that time.
            num_excluded_short_trajs (int): The number of trajectories excluded for being too short
        """

        # Convert the timestamp to Unix timestamp in CST timezone
        target_timestamp_unix = convert_to_cst_unix(target_timestamp)

        segment_min_mm = min(segment[0], segment[1])
        segment_max_mm = max(segment[0], segment[1])

        # Create a new dataframe to store outputs
        column_dtypes = {col: "float64" for col in self.inception_features}
        output_df = pd.DataFrame(columns=self.inception_features).astype(column_dtypes)

        num_excluded_short_trajs = 0
        for trajectory in valid_trajectories:

            timestamps = get_array(
                trajectory.get("timestamp", None)
            )  # in seconds (unix time)
            x_positions = (
                get_array(trajectory.get("x_position", None)) / 5280
            )  # Converting feet to miles
            y_positions = get_array(trajectory.get("y_position", None))  # in feet

            # Build dataframe for the current trajectory
            traj_df = pd.DataFrame(
                np.column_stack(
                    (timestamps, x_positions, y_positions,) 
                ),
            )
            traj_df.columns = self.inception_features

            # Get the row where 'timestamp' entry is closest to the target timestamp
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

            # Append the single row of the timestamp_index to the output dataframe
            output_df = pd.concat(
                [output_df, traj_df.iloc[[timestamp_index]]], axis=0, ignore_index=True,
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
        df["headway"] = self.headway_nan_fill_val

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

                # Condition 1: Leader's y_position is within ego vehicle's lane
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
                # Convert headway from miles to meters
                df.loc[ego_idx, "headway"] = min_found_headway * 1609.34

        return df

    def build_sumo_headway_dist(self, timestamp, direction_string, segment):
        """
        # TODO: Note that we throw out vehicles on internal edges since we can't get their positions from SUMO (shouldn't be that many)
        Function to get the SUMO headway distribution ('leaderGap' feature, in SUMO parlance) for a specific segment at a given timestamp.

        Args:
            timestamp (str): The timestamp 'snapshot' for which to get the SUMO data. (e.g., "2022-11-23 06:23:00")
            direction_string (str): The direction of the segment ("westbound" or "eastbound")
            segment (tuple): The MM segment of interest (e.g., (59, 60) or (61.1, 61.6)).

        Returns:
            pd.DataFrame: A DataFrame with the computed headway feature for the specified segment and timestamp.
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
        # Note: We exclude vehicles on internal edges since sumolib cannot retrieve their positions and thus we can't assign to segments
        valid_edge_ids = self.eb_route_edge_ids if direction_string == "eastbound" else self.wb_route_edge_ids
        snapshot_df = snapshot_df[snapshot_df['edge_id'].isin(valid_edge_ids)]

        # Get the anchor edge and position based on the direction string
        anchor_edge = self.eb_anchor_edge if direction_string == "eastbound" else self.wb_anchor_edge
        anchor_pos_on_edge = self.eb_anchor_pos_on_edge if direction_string == "eastbound" else self.wb_anchor_pos_on_edge

        # For each remaining detection, get the driving distance from the anchor point
        if not snapshot_df.empty:
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
        else: # Handle empty dataframe case (occurs when no vehicles on valid segement edges at that time)
            snapshot_df['dist_from_anchor'] = []

        # Get the distances for the segment
        segment_start_dist_from_anchor, segment_end_dist_from_anchor = segment
        # Filter the snapshot_df to only include vehicles within the segment distances
        snapshot_df = snapshot_df[snapshot_df['dist_from_anchor'].between(segment_start_dist_from_anchor, segment_end_dist_from_anchor)]

        # Keep only the columns of interest
        snapshot_df = snapshot_df[['id', 'time', 'x', 'y', 'dist_from_anchor', 'type', 'speed', 'leaderGap']]

        return snapshot_df
    
class VelocityGridErrorCalculator:
    """
    Class to aggregate SUMO FCD and INCEPTION trajectory data into velocity grids for comparison.
    """

    def __init__(self,
                 micro_obj_fn_timestamps,
                 fcd_xml_file_path,
                 net_file_path,
                 mm_latlon_mapping_path,
                 segment_dict=None,
                 cache_traj_verbosity=False,
                 cache_traj_disable_tqdm=True
                 ):
        """
        Initializes the VelocityGridErrorCalculator with network and segment information.

        Args:
            micro_obj_fn_timestamps (list): List of timestamps (str) at which to evaluate the microscopic error.
                Ex: ["2022-11-21 07:30:00", "2022-11-21 07:45:00", ...]
            fcd_xml_file_path (str): Path to the SUMO FCD XML file.
            inception_data_dir (str): Directory containing INCEPTION trajectory CSV files.
            net_file_path (str): Path to the SUMO network file.
            mm_latlon_mapping_path (str): Path to the CSV file mapping mile markers to lat/lon coordinates.
            cached_traj_dir (str): Directory to store/load cached INCEPTION trajectories for faster processing.
            segment_dict (dict): Optional dictionary defining segments for analysis. If None, default segments are generated.
            cache_traj_verbosity (bool): If True, enables verbose output during trajectory caching.
            cache_traj_disable_tqdm (bool): If True, disables progress bars during trajectory caching.
        """

        # List of timestamps at which to evaluate the microscopic error
        self.micro_obj_fn_timestamps = micro_obj_fn_timestamps[:-1]

        # Network info
        self.eb_route_start_edge_id = '108162303'
        self.eb_route_end_edge_id = '988591740#1'
        self.wb_route_start_edge_id = '27828382'
        self.wb_route_end_edge_id = '634155175'
        # # Vals for Maryam network
        # self.eb_route_start_edge_id = '108162303'
        # self.eb_route_end_edge_id = '988591740#0'
        # self.wb_route_start_edge_id = '974949114'
        # self.wb_route_end_edge_id = '634155175'
        self.search_radius = 50 # in meters, used to find neighboring edges
        self.use_internal_edges = True  # Whether to use internal edges in the network
        self.net = sumolib.net.readNet(net_file_path, withInternal=self.use_internal_edges)

        # SUMO FCD processing params
        self.fcd_xml_file_path = fcd_xml_file_path  # Path to the FCD XML file
        self.fcd_numeric_cols = ['x', 'y', 'angle', 'speed', 'pos', 'slope', 'time', 'leaderGap']

        # Segment params & info
        # Anchor points may or may not be on starting edges (e.g., if segment of interest is interior), so define separately
        self.eb_anchor_mm = 58.8
        self.wb_anchor_mm = 62.8
        self.segment_len = 1.0  # Length of each segment in miles
        self.mm_latlon_df = pd.read_csv(mm_latlon_mapping_path)
        
        # Visualization params
        self.dist_plot_dir = "./cal_methods/sumo_baseline/headway_dist_plots/" # TODO parameterize this
        self.num_bins = 30
        self.figsize = (10, 6)  # Set the figure size for INCEPTION/SUMO feature distribution histograms
        self.alpha = 0.6  # Set the transparency for INCEPTION/SUMO feature distribution histograms
        
        # Initialize MM segment dictionary
        if segment_dict is None:
            self.segment_dict = self.gen_segments_by_length(self.eb_anchor_mm, self.wb_anchor_mm, self.segment_len)
        else:
            self.segment_dict = segment_dict

        # Get the full extent of the eb and wb routes
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

        # Initialize SUMO segment dictionary
        # Convert segments to SUMO distances from anchor point
        self.segment_dict_sumo = {}
        for direction in self.segment_dict.keys():
            self.segment_dict_sumo[direction] = [None] * len(self.segment_dict[direction])
            for i, segment in enumerate(self.segment_dict[direction]):
                self.segment_dict_sumo[direction][i] = self.convert_segment_to_sumo_cutoffs(direction, segment)
    
    @staticmethod
    def to_seconds_since_midnight(ts):
        """Convert datetime string or object to seconds after midnight."""
        if isinstance(ts, str):
            ts = pd.to_datetime(ts)
        elif isinstance(ts, (pd.Timestamp,)):
            ts = ts.to_pydatetime()
        return ts.hour * 3600 + ts.minute * 60 + ts.second
    
    def fcd_xml_to_df(self, fcd_xml_file_path: str):
        """
        Reads a SUMO FCD XML file and returns a DataFrame containing vehicle data.
        Uses iterparse for efficiency on large files.

        Args:
            fcd_xml_file_path (str): Path to the FCD XML file.

        Returns:
            pandas.DataFrame: A DataFrame containing all vehicle data.
        """        

        all_vehicle_data = []

        print(f"Extracting SUMO sim traj data from {fcd_xml_file_path}")

        # Efficient streaming parse
        for event, elem in ET.iterparse(fcd_xml_file_path, events=('start', 'end')):
            if event == 'start' and elem.tag == 'timestep':
                current_time = float(elem.attrib['time'])  # store timestep time
            elif event == 'end' and elem.tag == 'vehicle':
                vehicle_data = elem.attrib.copy()  # copy vehicle attributes
                vehicle_data['time'] = current_time  # add timestep time
                all_vehicle_data.append(vehicle_data)
                elem.clear()  # free memory

        if not all_vehicle_data:
            raise ValueError(f"No data found in FCD XML file: {fcd_xml_file_path}")

        # Convert to DataFrame
        df = pd.DataFrame(all_vehicle_data)

        # Convert numeric columns (using your defined list of numeric fields)
        numeric_cols = self.fcd_numeric_cols
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Define start and end time from self.micro_obj_function_timestamps
        start_time = VelocityGridErrorCalculator.to_seconds_since_midnight(self.micro_obj_fn_timestamps[0])
        end_time = VelocityGridErrorCalculator.to_seconds_since_midnight(self.micro_obj_fn_timestamps[-1])
        # start_time = datetime.strptime(self.micro_obj_fn_timestamps[0], "%Y-%m-%d %H:%M:%S")
        # start_time = start_time.hour * 3600 + start_time.minute * 60 + start_time.second
        # assert type(start_time) == int, "Start time must be an integer representing seconds in the day."
        # end_time = datetime.strptime(self.micro_obj_fn_timestamps[-1], "%Y-%m-%d %H:%M:%S")
        # end_time = end_time.hour * 3600 + end_time.minute * 60 + end_time.second
        # assert type(end_time) == int, "End time must be an integer representing seconds in the day."

        # Filter rows within time window
        df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]

        return df

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
    
    def build_sumo_xy_to_mm_mapper(self, widen_half_dist_meters=40):
        """
        Builds a mapping function from SUMO (x, y) coordinates on the roadway to mile markers.
        Uses a more robust interpolation method by generating additional points offset (otherwise
        the standard linear interpolator can produce NaNs for points slightly off the road centerline).

        Parameters:
        widen_half_dist_meters: The distance in meters to offset points on either side of the interpolation line.
                                Further out allows vehicles further from the centerline to be mapped, but too far may introduce inaccuracies.

        Returns:
            interp_func_robust: A function that takes (lon, lat) and returns the corresponding mile marker.
        """

        # --- 1. Setup Coordinate Transformation ---
        crs_wgs84 = CRS("EPSG:4326")
        crs_projected = CRS("EPSG:32616") # Nashville UTM Zone
        to_projected = Transformer.from_crs(crs_wgs84, crs_projected, always_xy=True)
        to_wgs84 = Transformer.from_crs(crs_projected, crs_wgs84, always_xy=True)

        # --- 2. Convert original points to meters ---
        points_df = self.mm_latlon_df # Just create an alias for easier reference
        lons = points_df['X_WGS84'].values
        lats = points_df['Y_WGS84'].values
        x_meter, y_meter = to_projected.transform(lons, lats)

        # --- 3. Generate Widened Points and Store Intermediate Values ---

        # These lists are for building the interpolator
        wide_points_meter = []
        all_mm_values = []

        for i in range(len(x_meter)):
            # Add original point for interpolator
            wide_points_meter.append((x_meter[i], y_meter[i]))
            all_mm_values.append(points_df['MM'].iloc[i])

            # Calculate vector for perpendicular offset 
            if i < len(x_meter) - 1:
                vec_x, vec_y = x_meter[i+1] - x_meter[i], y_meter[i+1] - y_meter[i]
            else:
                vec_x, vec_y = x_meter[i] - x_meter[i-1], y_meter[i] - y_meter[i-1]
            norm = np.sqrt(vec_x**2 + vec_y**2)
            perp_x, perp_y = -vec_y / norm, vec_x / norm

            # Calculate meter offsets and the new points
            dx1, dy1 = perp_x * widen_half_dist_meters, perp_y * widen_half_dist_meters
            dx2, dy2 = -perp_x * widen_half_dist_meters, -perp_y * widen_half_dist_meters
            new_x1, new_y1 = x_meter[i] + dx1, y_meter[i] + dy1
            new_x2, new_y2 = x_meter[i] + dx2, y_meter[i] + dy2

            # Add offset points for the interpolator
            wide_points_meter.append((new_x1, new_y1))
            wide_points_meter.append((new_x2, new_y2))
            all_mm_values.append(points_df['MM'].iloc[i])
            all_mm_values.append(points_df['MM'].iloc[i])

        # --- 5. Build and Return the Robust Interpolator  ---
        wide_points_meter_arr = np.array(wide_points_meter)
        wide_lons, wide_lats = to_wgs84.transform(wide_points_meter_arr[:, 0], wide_points_meter_arr[:, 1])
        wide_lonlat_points = np.vstack((wide_lons, wide_lats)).T
        interp_func_robust = LinearNDInterpolator(wide_lonlat_points, all_mm_values)

        return interp_func_robust
    
    def compute_velocity_grid_from_fcd(
        self,
        time_interval=pd.Timedelta(seconds=10),
        space_interval=400,     # in meters
        x_min_miles=58.8,
        x_max_miles=62.8,
        expected_duration=3600,  # <- NEW: in seconds (optional)
        return_df=False
    ):
        """
        Build a 2D grid of average velocities from SUMO FCD data.
        Rows = time bins, Cols = space bins.

        Args:
            time_interval (pd.Timedelta): Width of each time bin.
            space_interval (float): Width of each space bin (meters).
            x_min_miles (float): Lower bound of road segment in miles.
            x_max_miles (float): Upper bound of road segment in miles.
            expected_duration (float, optional): If set, ensures time bins span exactly this duration
                                                (in seconds) starting from t_min.
            return_df (bool): If True, also return the pivoted DataFrame for inspection.
        """

        # --- Load FCD file ---
        df = self.fcd_xml_to_df(self.fcd_xml_file_path)
        print(f"FCD DataFrame shape (before filtering): {df.shape}")

        # Extract edge_id (direction indicator)
        df['edge_id'] = df['lane'].apply(lambda x: x.split('_')[0])

        # Filter to only WB data
        # df['edge_id'] = df['lane'].apply(lambda x: x.split('_')[0])
        # valid_edge_ids = self.wb_route_edge_ids
        # df = df[df['edge_id'].isin(valid_edge_ids)]
        # print(f"FCD DataFrame shape (after WB filter): {df.shape}")

        # --- Build interpolator for mile marker mapping ---
        interp_func_robust = self.build_sumo_xy_to_mm_mapper()

        # --- Map SUMO (x,y) → lon/lat → mile marker ---
        lonlat = np.array([self.net.convertXY2LonLat(x, y) for x, y in zip(df['x'], df['y'])])
        df['mm'] = interp_func_robust(lonlat)
        df = df.dropna(subset=['mm']) # Drop rows where mm is NaN; these points are beyond the convex hull of the mile marker data (likely too far up/down road or off to the side)
        print(f"FCD DataFrame shape (after mm mapping & filtering): {df.shape}")

        # --- Filter to road segment ---
        df = df[(df['mm'] >= x_min_miles) & (df['mm'] <= x_max_miles)]
        print(f"FCD DataFrame shape (after segment filter): {df.shape}")

        # # # Filter to only one direction of data
        # df['edge_id'] = df['lane'].apply(lambda x: x.split('_')[0])
        # valid_edge_ids = self.eb_route_edge_ids
        # df = df[df['edge_id'].isin(valid_edge_ids)] 
        # print(f"FCD DataFrame shape (after WB filter): {df.shape}")

        # # --- Build interpolator for mile marker mapping ---
        # lonlat_points = self.mm_latlon_df[['X_WGS84', 'Y_WGS84']].values
        # mm_values = self.mm_latlon_df['MM'].values
        # interp_func = LinearNDInterpolator(lonlat_points, mm_values)

        # # --- Map SUMO (x,y) → lon/lat → mile marker ---
        # lonlat = np.array([self.net.convertXY2LonLat(x, y) for x, y in zip(df['x'], df['y'])])
        # df['mm'] = interp_func(lonlat)
        # df = df.dropna(subset=['mm'])
        # print(f"FCD DataFrame shape (after mm mapping): {df.shape}")

        # # --- Filter to road segment ---
        # df = df[(df['mm'] >= x_min_miles) & (df['mm'] <= x_max_miles)]
        # print(f"FCD DataFrame shape (after segment filter): {df.shape}")

        if df.empty:
            raise ValueError("No FCD data in the specified segment and time range.")

        # --- Setup time bins based on micro_obj_function_timestamps ---
        time_bins = np.array([VelocityGridErrorCalculator.to_seconds_since_midnight(ts) for ts in self.micro_obj_fn_timestamps],dtype=float)

        n_expected_bins = len(time_bins)
        print(f"Using {n_expected_bins} time bins from micro_obj_function_timestamps")

        # t_min = df["time"].min()
        # time_interval_sec = time_interval.total_seconds()

        # if expected_duration is not None:
        #     t_max = t_min + expected_duration
        # else:
        #     t_max = df["time"].max()

        # n_expected_bins = int(np.ceil((t_max - t_min) / time_interval_sec))
        # print(f"t_min = {t_min}, t_max = {t_max}, expected #bins = {n_expected_bins}")

        # time_bins = np.arange(t_min, t_min + n_expected_bins * time_interval_sec + 1, time_interval_sec)

        # --- Setup space bins ---
        x_min = x_min_miles * 1609.34
        x_max = x_max_miles * 1609.34
        space_bins = np.arange(x_min, x_max, space_interval)

            # --- Helper function to compute velocity grid for one direction ---
        def _build_velocity_grid(direction_ids, direction_label):
            sub_df = df[df['edge_id'].isin(direction_ids)]
            print(f"{direction_label} DataFrame shape: {sub_df.shape}")

            if sub_df.empty:
                print(f"Warning: no {direction_label} data found within this segment.")
                pivot_table = pd.DataFrame(np.nan, index=range(n_expected_bins), columns=range(len(space_bins)-1))
            else:
                # Assign bins
                sub_df["time_bin"] = pd.cut(sub_df["time"], bins=time_bins, labels=False, include_lowest=True)
                sub_df["space_bin"] = pd.cut(sub_df["mm"] * 1609.34, bins=space_bins, labels=False, include_lowest=True)
                sub_df = sub_df.dropna(subset=["time_bin", "space_bin"]).astype({"time_bin": int, "space_bin": int})

                # Pivot to grid
                pivot_table = sub_df.pivot_table(
                    index="time_bin",
                    columns="space_bin",
                    values="speed",
                    aggfunc="mean"
                ).reindex(
                    index=range(n_expected_bins),
                    columns=range(len(space_bins) - 1)
                )

                # Convert m/s → km/h
                pivot_table = pivot_table * 3.6

            if direction_label == "WB":
                # Flip horizontally
                velocity_grid = np.flip(pivot_table.to_numpy(), axis=1)

            else:
                velocity_grid = pivot_table.to_numpy()

            return velocity_grid, pivot_table

        # --- Compute both directions ---
        wb_velocity_grid, wb_pivot_table = _build_velocity_grid(self.wb_route_edge_ids, "WB")
        eb_velocity_grid, eb_pivot_table = _build_velocity_grid(self.eb_route_edge_ids, "EB")

        # --- Return according to flag ---
        if return_df:
            return eb_velocity_grid, eb_pivot_table, wb_velocity_grid, wb_pivot_table
        else:
            return eb_velocity_grid, wb_velocity_grid

    def _process_chunk(self, rows, interp_func, accum, time_bins, space_bins, x_min_miles, x_max_miles):
        """
        Process one chunk of FCD XML data:
        - convert to DataFrame
        - compute mile markers
        - bin by time and space
        - accumulate sum and count by direction
        """
        df = pd.DataFrame(rows)
        numeric_cols = self.fcd_numeric_cols
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Compute mile markers
        lonlat = np.array([self.net.convertXY2LonLat(x, y) for x, y in zip(df["x"], df["y"])])
        df["mm"] = interp_func(lonlat)
        df = df.dropna(subset=["mm"])

        # Filter to road segment
        df = df[(df["mm"] >= x_min_miles) & (df["mm"] <= x_max_miles)]
        if df.empty:
            return

        # Direction identification
        df["edge_id"] = df["lane"].apply(lambda x: x.split("_")[0])
        df["time_bin"] = pd.cut(df["time"], bins=time_bins, labels=False, include_lowest=True)
        df["space_bin"] = pd.cut(df["mm"] * 1609.34, bins=space_bins, labels=False, include_lowest=True)
        df = df.dropna(subset=["time_bin", "space_bin"]).astype({"time_bin": int, "space_bin": int})

        for direction, ids in [("EB", self.eb_route_edge_ids), ("WB", self.wb_route_edge_ids)]:
            sub_df = df[df["edge_id"].isin(ids)]
            for _, row in sub_df.iterrows():
                key = (row["time_bin"], row["space_bin"])
                accum[direction][key]["sum"] += row["speed"]
                accum[direction][key]["count"] += 1

        del df
        gc.collect()
      
    def compute_velocity_grid_from_fcd_streaming(
        self,
        time_interval=pd.Timedelta(seconds=10),
        space_interval=400,     # meters
        x_min_miles=58.8,
        x_max_miles=62.8,
        expected_duration=3600,
        return_df=False
    ):
        """
        Memory-efficient version of compute_velocity_grid_from_fcd().
        Streams the FCD XML file in chunks, filters by time window,
        bins data on the fly, and aggregates average velocities.

        Returns EB and WB velocity grids (km/h).
        """

        # === Setup constants ===
        print(f"Streaming FCD XML: {self.fcd_xml_file_path}")
        start_time = VelocityGridErrorCalculator.to_seconds_since_midnight(self.micro_obj_fn_timestamps[0])
        end_time   = VelocityGridErrorCalculator.to_seconds_since_midnight(self.micro_obj_fn_timestamps[-1])

        # Convert mile range to meters
        x_min = x_min_miles * 1609.34
        x_max = x_max_miles * 1609.34

        # Precompute bins
        time_bins = np.array(
            [VelocityGridErrorCalculator.to_seconds_since_midnight(ts) for ts in self.micro_obj_fn_timestamps],
            dtype=float
        )
        n_expected_bins = len(time_bins)
        space_bins = np.arange(x_min, x_max, space_interval)

        print(f"Expected bins: time={n_expected_bins}, space={len(space_bins)-1}")

        # Build interpolator for mm mapping
        interp_func_robust = self.build_sumo_xy_to_mm_mapper()

        # Prepare accumulators for each direction
        accum = {
            "EB": defaultdict(lambda: {"sum": 0.0, "count": 0}),
            "WB": defaultdict(lambda: {"sum": 0.0, "count": 0})
        }

        # === Stream parse XML ===
        chunk_rows = []
        chunk_size = 20000  # adjust as needed

        for event, elem in ET.iterparse(self.fcd_xml_file_path, events=("start", "end")):
            if event == "start" and elem.tag == "timestep":
                current_time = float(elem.attrib["time"])
                skip_time = not (start_time <= current_time <= end_time)

            elif event == "end" and elem.tag == "vehicle" and not skip_time:
                vehicle_data = elem.attrib
                vehicle_data["time"] = current_time
                chunk_rows.append(vehicle_data)

                if len(chunk_rows) >= chunk_size:
                    self._process_chunk(chunk_rows, interp_func_robust, accum, time_bins, space_bins, x_min_miles, x_max_miles)
                    chunk_rows.clear()
                    gc.collect()

                elem.clear()

        # Final leftover chunk
        if chunk_rows:
            self._process_chunk(chunk_rows, interp_func_robust, accum, time_bins, space_bins, x_min_miles, x_max_miles)
            chunk_rows.clear()
            gc.collect()

        print("Finished streaming parse. Building final grids...")

        # === Aggregate results ===
        def build_grid(direction):
            mat_sum = np.zeros((n_expected_bins, len(space_bins)-1))
            mat_count = np.zeros_like(mat_sum)
            for (tb, sb), v in accum[direction].items():
                if tb < n_expected_bins and sb < len(space_bins)-1:
                    mat_sum[tb, sb] += v["sum"]
                    mat_count[tb, sb] += v["count"]
            with np.errstate(invalid="ignore", divide="ignore"):
                grid = (mat_sum / mat_count) * 3.6  # m/s → km/h
            if direction == "WB":
                grid = np.flip(grid, axis=1)
            return grid

        eb_grid = build_grid("EB")
        wb_grid = build_grid("WB")

        print("Velocity grids ready.")

        if return_df:
            # convert to DataFrames for inspection
            eb_df = pd.DataFrame(eb_grid)
            wb_df = pd.DataFrame(wb_grid)
            return eb_grid, eb_df, wb_grid, wb_df
        else:
            return eb_grid, wb_grid
        
    @staticmethod
    def load_trajectories(inception_file_path, trajectory_timeframe=pd.Timedelta(minutes=60), min_time=None):
        # eastbound_trajectories = []
        westbound_trajectories = []
        t_min = None
        t_max = None
        # MILE_MARKER_61 = 98170  # meters
        MILE_MARKER_61 = 58.8 * 5280 * 0.3048 #meters
        # MILE_MARKER_62 = 99770  # meters
        #99769.99
        MILE_MARKER_62 = 62.8 * 5280 * 0.3048 # 2800 meters
        # Open file and stream data
        with open(inception_file_path, "r") as f:
            trajectory_iterator = ijson.items(f, "item")
            
            tracker = 0
            for traj in trajectory_iterator:
                # Mile marker 61 is 322080 feet or 98170 m
                # Mile marker 62 is 327360 feet or 99779.3 m
                x_positions = np.array(traj.get("x_position", []), dtype=np.float32) * 0.3048  # Convert feet to meters
                y_positions = np.array(traj.get("y_position", []), dtype=np.float32) * 0.3048  # Convert feet to meters
                direction = traj.get("direction")

                if len(x_positions) > 1 and direction == -1:
                    timestamps = np.array(traj.get("timestamp", []), dtype=np.float64)
                    timestamps = pd.to_datetime(timestamps, unit="s").astype(np.int64) / 1e9  # Convert to seconds
                    
                    if min_time and (timestamps[0] < min_time.timestamp()):
                        continue
                    
                    # eastbound_trajectories.append({
                    #     "trajectory": traj, 
                    #     "timestamps": timestamps,
                    #     "x_positions": x_positions,
                    #     "y_positions": y_positions
                    # })
                    westbound_trajectories.append({
                        "trajectory": traj, 
                        "timestamps": timestamps,
                        "x_positions": x_positions,
                        "y_positions": y_positions
                    })
                    
                    # Efficient min/max tracking
                    t_min = timestamps[0] if t_min is None else min(t_min, timestamps[0])
                    t_max = timestamps[0] if t_max is None else max(t_max, timestamps[0])

                    if tracker % 10_000 == 0:
                        print(f"Processed {tracker} trajectories. Current t_min: {pd.to_datetime(t_min, unit='s')}, t_max: {pd.to_datetime(t_max, unit='s')}")
                    tracker += 1

                    if t_max is not None and t_min is not None and (t_max - t_min) > trajectory_timeframe.total_seconds():
                        break # check later

        # print(f"Loaded {len(eastbound_trajectories)} eastbound trajectories.")
        print(f"Loaded {len(westbound_trajectories)} westbound trajectories.")

        # if not eastbound_trajectories:
        if not westbound_trajectories:
            
            return pd.DataFrame(columns=["trajectory_id", "timestamp", "x_position", "speed"])

        # Vectorized DataFrame creation
        all_trajectory_ids = []
        all_timestamps = []
        all_x_positions = []
        all_y_positions = []
        all_speeds = []

        # for idx, traj in enumerate(eastbound_trajectories):
        for idx, traj in enumerate(westbound_trajectories):
            mask = (traj["x_positions"] >= MILE_MARKER_61) & (traj["x_positions"] <= MILE_MARKER_62)
            filtered_timestamps = traj["timestamps"][mask]
            filtered_x_positions = traj["x_positions"][mask]
            filtered_y_positions = traj["y_positions"][mask]

            num_points = len(filtered_timestamps)
            all_trajectory_ids.extend([idx] * num_points)
            all_timestamps.extend(filtered_timestamps)
            all_x_positions.extend(filtered_x_positions)
            all_y_positions.extend(filtered_y_positions)
        df = pd.DataFrame({
            "trajectory_id": np.array(all_trajectory_ids, dtype=np.int32),
            "timestamp": pd.to_datetime(all_timestamps, unit="s"),
            "x_position": np.array(all_x_positions, dtype=np.float32),
            "y_position": np.array(all_y_positions, dtype=np.float32)
            # "speed": np.array(all_speeds, dtype=np.float32)
        })

        # print(df.columns.tolist())  # Should include 'trajectory_id'
        # print(df)
        
        return df
    
    def westbound_lane_func(self):
        return 4
    
    def get_ground_truth_npy(self, df, lane_func=westbound_lane_func, time_interval=pd.Timedelta(seconds=10), space_interval=400):
        # Compute min/max for time and space
        
        t_min, t_max = df["timestamp"].min(), df["timestamp"].max()
        x_min, x_max = df["x_position"].min(), df["x_position"].max()
        # print("xmax", x_max)
        # print("x_min", x_min)
        # Ensure valid ranges
        if x_min == x_max:
            raise ValueError("x_min and x_max are identical, meaning no variation in x_position.")
        
        # Create time and space bins
        time_bins = pd.date_range(start=t_min, end=t_max, freq=time_interval)
        # print(time_bins)
        space_bins = np.arange(x_min, x_max, space_interval)
        # print(space_bins)
        
        if len(space_bins) < 2:
            raise ValueError("space_bins array is empty or too small, adjust space_interval.")
        
        # labels = [58.8, 59.0666666667, 59.3333333333, 59.6, 
        # 59.8666666667, 60.1333333333, 60.4, 60.6666666667, 
        # 60.9333333333, 61.2, 61.4666666667, 61.7333333333, 
        # 62.0, 62.2666666667, 62.5333333333, 62.8]
        
        # Assign bin indices using `pd.cut()`
        df["time_bin"] = pd.cut(df["timestamp"], bins=time_bins, labels=False, include_lowest=True)
        df["space_bin"] = pd.cut(df["x_position"], bins=space_bins, labels=False, include_lowest=True)

        # Remove NaNs (out-of-range values)
        df = df.dropna(subset=["time_bin", "space_bin"]).astype({"time_bin": int, "space_bin": int})

        # Compute flow and density using `groupby()`
        flow_matrix = np.zeros((len(time_bins) - 1, len(space_bins) - 1))
        density_matrix = np.zeros_like(flow_matrix)
        traj_count_matrix = np.zeros_like(flow_matrix)
        # lane_matrix = np.zeros_like(flow_matrix)

        grouped = df.groupby(["time_bin", "space_bin"])
        area_bin = (space_interval/1000.) * time_interval.total_seconds() / 3600.0 #convert space interval to kilometers, time_interval to hours
        for (time_bin, space_bin), group in grouped:
            # print(time_bin, space_bin)
            traj_group = group.groupby("trajectory_id")
            # traj_dict = {traj_id: traj_data for traj_id, traj_data in traj_group}
        
            total_distance = sum(traj_group["x_position"].apply(lambda x: x.max()-x.min()))
            total_time = sum(traj_group["timestamp"].apply(lambda x: (x.max() - x.min()).total_seconds()))

            flow_matrix[time_bin, space_bin] = (total_distance / (1000.*lane_func(space_bin))) / area_bin
            density_matrix[time_bin, space_bin] = (total_time / (3600.0 * lane_func(space_bin))) / area_bin
            velocity_matrix = flow_matrix/density_matrix

            # --- Count unique trajectories in this bucket ---
            traj_count = traj_group.ngroups   # number of unique trajectory_ids
            traj_count_matrix[time_bin, space_bin] = traj_count

        # Plot histogram of y_position for each space_bin
        # space_grouped = df.groupby("space_bin")
        # for space_bin, group in space_grouped:
        #     plt.figure(figsize=(8, 5))
        #     plt.hist(group["y_position"], bins=30, alpha=0.7)
        #     plt.title(f"Histogram of y_position for space_bin {space_bin}")
        #     plt.xlabel("y_position")
        #     plt.ylabel("Count")
        #     plt.grid(True)
        #     plt.tight_layout()
        #     plt.savefig(f"data/histogram_ypos_spacebin_{space_bin}.png")  # Save to file
        #     plt.close()
        # print(grouped)
        
        return flow_matrix, density_matrix, velocity_matrix, traj_count_matrix
        
    @staticmethod    
    def mape(ground_truth, prediction):
        """
        Compute the Mean Absolute Percentage Error (MAPE) between ground truth and prediction.
        Safely handles NaNs and division by zero.

        Parameters:
            ground_truth (np.ndarray): Ground truth array of shape [t, i]
            prediction (np.ndarray): Predicted array of shape [t, i]

        Returns:
            float: The mean absolute percentage error (in percent)
        """
        # Ensure both are numpy arrays
        ground_truth = np.asarray(ground_truth, dtype=float)
        prediction = np.asarray(prediction, dtype=float)

        # Mask out invalid entries: NaNs or zeros in ground truth
        valid_mask = (~np.isnan(ground_truth)) & (~np.isnan(prediction)) & (ground_truth != 0)

        if not np.any(valid_mask):
            # No valid data points
            return np.nan

        error = np.abs((prediction[valid_mask] - ground_truth[valid_mask]) / ground_truth[valid_mask])

        return np.nanmean(error) * 100
