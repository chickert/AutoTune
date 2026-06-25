"""
File with utility functions for the benchmark.

Includes:
- key_func: Key function to aid in sorting the list of detector IDs, such that eastbound/westbound detectors are grouped together
- get_time_of_day_in_minutes: Converts a wall clock time to the number of minutes elapsed since the day began
- convert_to_cst_unix: Converts a date string in the format 'YYYY-MM-DD HH:MM' to a Unix timestamp in CST timezone (Nashville local time)
- get_array: Converts a list of decimal numbers into a NumPy array of floats
- get_valid_trajectories: Gets all nonzero trajectories that are within the time range, pass by the detector location, and are in the correct direction
- cache_trajectories: Gets all trajectories (of length > 1) that have at least one point within the time range 
    (May help speed up subsequent calls to get_valid_trajectories() by caching the trajectories that are within the time range)
- cache_trajectories_for_timestamps: Gets trajectories that overlap with any of the specified timestamps in the timestamp_list
- convert_sumo_xy_to_mm: Converts SUMO XY coordinates from FCD .xml file to a mile marker (MM) using 2D interpolation
- generate_time_intervals: Generates a list of time strings at equal intervals between a begin and end time on a given date

"""

import numpy as np
import datetime
import pytz
import ijson
import sumolib
import os
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
from scipy.interpolate import LinearNDInterpolator


def key_func(item):
    """
    Key function to aid in sorting the list of detector IDs, such that eastbound/westbound detectors are grouped together.
    """
    number, direction = item.split("-")
    return direction, int(number)


def get_time_of_day_in_minutes(time_str, return_float=False):
    """
    Converts a wall clock time to the number of minutes elapsed since the day began.
    If return_float is False, returns a string rounded down to the nearest integer and appends 0 to front of 3-digit numbers to assist with sorting.
    If return_float is True, returns a float instead of a string.

    Args:
        time_str: The timestamp in the format 'YYYY-MM-DD HH:MM:SS'.
        return_float: If True, returns the number of minutes as a float. If False, returns it as a string.

    Returns:
        str or float: The number of minutes elapsed since the day began, formatted as a string or float.
    """

    day_start_time_str = time_str.split(" ")[0] + " 00:00:00"
    day_start_time = datetime.datetime.strptime(day_start_time_str, "%Y-%m-%d %H:%M:%S")
    time = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")

    time_difference = time - day_start_time
    minutes_elapsed = time_difference.total_seconds() / 60

    assert minutes_elapsed >= 0, f"Minutes elapsed is negative: {minutes_elapsed}"

    if return_float:
        return minutes_elapsed

    else:
        # Append 0 to front of 3-digit numbers to assist with sorting
        if len(str(int(minutes_elapsed))) == 3:
            minutes_elapsed = "0" + str(int(minutes_elapsed))
        else:
            minutes_elapsed = str(int(minutes_elapsed))

        return minutes_elapsed


def convert_to_cst_unix(date_string):
    """
    Convert a date string in the format 'YYYY-MM-DD HH:MM' to a Unix timestamp in CST timezone (Nashville local time).
    Sourced from https://github.com/I24-MOTION w/ slight modifications.

    Args:
        date_string (str): The date string in the format 'YYYY-MM-DD HH:MM:SS'.

    Returns:
        float: The Unix timestamp in CST timezone.


    """
    # Parse the date string to a datetime object
    date_object = datetime.datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")

    # Convert to CST timezone
    cst_timezone = pytz.timezone("America/Chicago")
    date_object_cst = cst_timezone.localize(date_object)

    # Convert to Unix timestamp
    unix_timestamp = int(date_object_cst.timestamp())
    return unix_timestamp


def get_array(decimal_list):
    """
    Converts a list of decimal numbers into a NumPy array of floats.
    Sourced from https://github.com/I24-MOTION w/ slight modifications.

    Args:
        decimal_list (list): A list of numbers (integers or floats) to be converted.
    Returns:
        numpy.ndarray: A NumPy array containing the elements of decimal_list as floats.
    """
    output_list = [float(decimal) for decimal in decimal_list]
    return np.array(output_list)


def get_valid_trajectories(input_filename, start_time, end_time, direction_string, detector_loc_milemarker, verbose, cached_trajectories=None, disable_tqdm=False):
    """
    Gets all trajectories (of length > 1) that are within the time range, pass by the detector location, and are in the correct direction.
    Uses cached_trajectories if provided, otherwise reads from the file at input_filename.
    NOTE: This function only checks trajectory start- and endpoints, so it may include trajectories that pass by the detector and occur at the correct time, 
            BUT do not actually drive BY the detector AT the correct time (that is, they meet the time and space thresholds independently).
            To address this, the get_lane_counts_and_speeds() method in the DetectionBuilder class in get_detections.py checks when each trajectory crosses the detector and only includes those that do so within the specified time range.
    Modified based on code in data tutorial at https://github.com/I24-MOTION 
            
    Args:
        input_filename (str): Path to the file containing the I-24 trajectory data.
        start_time (str): Start time of the time range. (YYYY-MM-DD HH:MM:SS)
        end_time (str): End time of the time range. (YYYY-MM-DD HH:MM:SS)
        direction_string (str): "westbound" or "eastbound"
        detector_loc_milemarker (float): Milemarker of the 'detector' location for which we want the data.
        verbose (bool): Whether to print additional information.
        cached_trajectories (list, optional): List of cached trajectories. If provided, this function will use the cached trajectories instead of reading from the input file.
        disable_tqdm (bool): Whether to disable the progress bar. Defaults to False.

    Returns:
        list: List of valid trajectories.
    """

    start_time_unix = convert_to_cst_unix(start_time)
    end_time_unix = convert_to_cst_unix(end_time)
    
    # Check direction
    assert direction_string in ['westbound', 'eastbound'], f"Invalid direction_string: {direction_string}"

    # -1 for westbound, 1 for eastbound
    direction = -1 if direction_string == 'westbound' else 1

    total_trajectories = 0

    if cached_trajectories is None:
        with open(input_filename, 'r') as input_file:
            parser = ijson.items(input_file, 'item')
            valid_trajectories = []
            for trajectory in tqdm(parser, desc="Reading trajectories", disable=disable_tqdm):
                # Need to get max and min milemarkers as follows, since they are inverted for eastbound vs westbound
                max_milemarker = max(float(trajectory['starting_x']) / 5280, float(trajectory['ending_x']) / 5280)
                min_milemarker = min(float(trajectory['starting_x']) / 5280, float(trajectory['ending_x']) / 5280)
                if ((float(trajectory['last_timestamp']) >= start_time_unix)
                        & (float(trajectory['first_timestamp']) <= end_time_unix)
                        & (int(trajectory['direction']) == direction)
                        & (max_milemarker >= detector_loc_milemarker)
                        & (min_milemarker <= detector_loc_milemarker)
                        & (int(trajectory['length']) > 1)):             # Trajectory needs at least 2 points to cross the detector (and for speed calc)
                    total_trajectories += 1
                    valid_trajectories.append(trajectory)
    else:
        valid_trajectories = []
        for trajectory in tqdm(cached_trajectories, desc="Reading trajectories from cache", disable=disable_tqdm):
            # Need to get max and min milemarkers as follows, since they are inverted for eastbound vs westbound
            max_milemarker = max(float(trajectory['starting_x']) / 5280, float(trajectory['ending_x']) / 5280)
            min_milemarker = min(float(trajectory['starting_x']) / 5280, float(trajectory['ending_x']) / 5280)
            if ((float(trajectory['last_timestamp']) >= start_time_unix)
                    & (float(trajectory['first_timestamp']) <= end_time_unix)
                    & (int(trajectory['direction']) == direction)
                    & (max_milemarker >= detector_loc_milemarker)
                    & (min_milemarker <= detector_loc_milemarker)
                    & (int(trajectory['length']) > 1)):             # Trajectory needs at least 2 points to cross the detector (and for speed calc)
                total_trajectories += 1
                valid_trajectories.append(trajectory)

    if verbose:
        print(f"\tTotal # of {direction_string} trajectories crossing mile marker {detector_loc_milemarker} & occur during {start_time} to {end_time}: {total_trajectories}")
    
    return valid_trajectories


def cache_trajectories(input_filename, start_time, end_time, verbose, disable_tqdm=False):
    """
    Gets all trajectories (of length > 1) that have at least one point within the time range.
    May help speed up subsequent calls to get_valid_trajectories() by caching the trajectories that are within the time range.
    Modified based on code in data tutorial at https://github.com/I24-MOTION 
            
    Args:
        input_filename (str): Path to the file containing the I-24 trajectory data.
        start_time (str): Start time of the time range. (YYYY-MM-DD HH:MM:SS)
        end_time (str): End time of the time range. (YYYY-MM-DD HH:MM:SS)
        verbose (bool): Whether to print additional information.
        disable_tqdm (bool): Whether to disable the progress bar. Defaults to False.

    Returns:
        list: List of cached trajectories based on the specified time range.
    """

    start_time_unix = convert_to_cst_unix(start_time)
    end_time_unix = convert_to_cst_unix(end_time)

    total_trajectories = 0

    with open(input_filename, 'r') as input_file:
        parser = ijson.items(input_file, 'item')
        cached_trajectories = []
        for trajectory in tqdm(parser, desc="Reading trajectories", disable=disable_tqdm):
            if ((float(trajectory['last_timestamp']) >= start_time_unix)
                    & (float(trajectory['first_timestamp']) <= end_time_unix)
                    & (int(trajectory['length']) > 1)):             # Trajectory needs at least 2 points to cross the detector (and for speed calc)
                total_trajectories += 1
                cached_trajectories.append(trajectory)

    if verbose:
        print(f"\tTotal # of trajectories during {start_time} to {end_time}: {total_trajectories}")
    
    return cached_trajectories


def cache_trajectories_for_timestamps(input_filename, timestamp_list, verbose, disable_tqdm=False):
    """
    Gets trajectories that overlap with any of the specified timestamps in the timestamp_list.
    
    Args:
        input_filename (str): Path to the file containing the I-24 trajectory data.
        timestamp_list (list): A list of timestamp strings (YYYY-MM-DD HH:MM:SS).
        verbose (bool): Whether to print additional information.
        disable_tqdm (bool): Whether to disable the progress bar. Defaults to False.

    Returns:
        list: List of cached trajectories based on the specified timestamps.
    """
    # Convert the list of timestamp strings to a set of Unix timestamps for efficient lookup
    unix_timestamps = {convert_to_cst_unix(ts) for ts in timestamp_list}
    
    total_trajectories = 0
    cached_trajectories = []

    with open(input_filename, 'r') as input_file:
        parser = ijson.items(input_file, 'item')
        
        for trajectory in tqdm(parser, desc="Reading trajectories", disable=disable_tqdm):
            first_timestamp_unix = float(trajectory.get('first_timestamp'))
            last_timestamp_unix = float(trajectory.get('last_timestamp'))
            
            # Check if the trajectory's time range overlaps with any of the specific timestamps
            has_matching_point = any(
                (first_timestamp_unix <= ts <= last_timestamp_unix) for ts in unix_timestamps
            )
            
            # Trajectory needs at least 2 points for speed calculations
            if has_matching_point and (int(trajectory['length']) > 1):
                total_trajectories += 1
                cached_trajectories.append(trajectory)

    if verbose:
        print(f"\tTotal # of trajectories with points at specified timestamps: {total_trajectories}")
    
    return cached_trajectories


def convert_sumo_xy_to_mm(
            sumo_x,
            sumo_y,
            net_file_path,
            mm_latlon_mapping_path="build_data/mile_marker_layer.csv"
            ):
        """
        Converts SUMO XY coordinates from FCD .xml file to a mile marker (MM) using 2D interpolation.
        This method uses both longitude and latitude to interpolate the MM, since the
        road that is not aligned with a single axis.
        
        Args:
            sumo_x (float): The SUMO x coordinate (in meters).
            sumo_y (float): The SUMO y coordinate (in meters).
            net_file_path (str): Path to the SUMO network file.
            mm_latlon_mapping_path (str): Path to the CSV file containing the mapping between mile markers and lat/lon coordinates.
        
        Returns:
            float: The interpolated mile marker.
        """

        # Step 0: Load the SUMO network and the mile marker to lat/lon mapping
        net = sumolib.net.readNet(net_file_path, withInternal=True)
        mm_latlon_df = pd.read_csv(mm_latlon_mapping_path)

        # Step 1: Convert the input SUMO x,y to geographic coordinates (lon, lat)
        lon, lat = net.convertXY2LonLat(sumo_x, sumo_y)
        
        # Step 2: Prepare the data for 2D interpolation
        lonlat_points = mm_latlon_df[['X_WGS84', 'Y_WGS84']].values
        mm_values = mm_latlon_df['MM'].values
        
        # Step 3: Create the 2D linear interpolator
        # The interpolator takes the (lon, lat) points as input and maps them to the MM values.
        interp_func = LinearNDInterpolator(lonlat_points, mm_values)

        # Step 4: Use the interpolation function to find the new MM
        # The result of the interpolation is an array, so we extract the single value.
        interpolated_mm = interp_func([(lon, lat)])[0]
        
        # Handle the case where the interpolated value might be a NaN (if outside the convex hull of the points)
        if np.isnan(interpolated_mm):
            print("Warning: Interpolation failed, coordinate is outside the range of known points. Returning NaN.")
            return np.nan
        
        return float(interpolated_mm)


def generate_time_intervals(year_str, date_str, begin_time_in_min, end_time_in_min, interval_len_in_min):
        """
        Generates a list of time strings at equal intervals between a begin and end time on a given date.

        Args:
            year_str (str): The year string in 'YYYY' format.
            date_str (str): The date string in 'MMDD' format.
            begin_time_in_min (int): The start time in minutes from midnight (0-1439).
            end_time_in_min (int): The end time in minutes from midnight (0-1439).
            interval_len_in_min (int): The length of the interval in minutes.

        Returns:
            list: A list of time strings in 'YYYY-MM-DD HH:MM:SS' format.
        """
        time_list = []
        
        # Start with the beginning time in minutes
        current_time_in_min = begin_time_in_min
        
        # Check for invalid interval length to prevent infinite loop
        if interval_len_in_min <= 0:
            return []

        # Parse the MMDD date string and combine with the current year
        date_parts = datetime.datetime.strptime(date_str, "%m%d")
        base_date = datetime.datetime(int(year_str), date_parts.month, date_parts.day)

        # Loop from the beginning time to the end time, inclusive
        while current_time_in_min <= end_time_in_min:
            # Calculate hours and minutes from the total minutes
            hours = current_time_in_min // 60
            minutes = current_time_in_min % 60
            
            # Create a datetime object for the current time by adding to the base date
            current_time_obj = base_date + datetime.timedelta(hours=hours, minutes=minutes)
            
            # Format the datetime object as a string and add to the list
            time_list.append(current_time_obj.strftime("%Y-%m-%d %H:%M:%S"))
            
            # Increment the time by the interval
            current_time_in_min += interval_len_in_min
            
        return time_list

def generate_time_intervals_seconds(year_str, date_str, begin_time_in_sec, end_time_in_sec, interval_len_in_sec):
    """
    Generates a list of time strings at equal intervals between a begin and end time on a given date.
    Works at the second level.

    Args:
        year_str (str): Year string in 'YYYY' format.
        date_str (str): Date string in 'MMDD' format.
        begin_time_in_sec (int): Start time in seconds from midnight (0–86399).
        end_time_in_sec (int): End time in seconds from midnight (0–86399).
        interval_len_in_sec (int): Interval length in seconds.

    Returns:
        list: A list of time strings in 'YYYY-MM-DD HH:MM:SS' format.
    """
    if interval_len_in_sec <= 0:
        return []

    # Parse date string
    date_parts = datetime.datetime.strptime(date_str, "%m%d")
    base_date = datetime.datetime(int(year_str), date_parts.month, date_parts.day)

    time_list = []
    current_time_in_sec = begin_time_in_sec

    while current_time_in_sec <= end_time_in_sec:
        current_time_obj = base_date + datetime.timedelta(seconds=current_time_in_sec)
        time_list.append(current_time_obj.strftime("%Y-%m-%d %H:%M:%S"))
        current_time_in_sec += interval_len_in_sec

    return time_list


def extract_sim_meas(measurement_locations, file_dir = ""):
    """
    Extract simulated traffic measurements (Q, V, Occ) from SUMO detector output files (xxx.out.xml).
    Q/V/Occ: [N_dec x N_time]
    measurement_locations: a list of strings that map detector IDs
    """
    # Initialize an empty list to store the data for each detector
    detector_data = {"speed": [], "volume": [], "occupancy": []}

    for detector_id in measurement_locations:
        # Construct the filename for the detector's output XML file
        # print(f"reading {detector_id}...")
        filename = os.path.join(file_dir, f"det_{detector_id}.out.xml")
        
        # Check if the file exists
        if not os.path.isfile(filename):
            print(f"File {filename} does not exist. Skipping this detector.")
            continue
        
        # Parse the XML file
        tree = ET.parse(filename)
        root = tree.getroot()

        # Initialize a list to store the measurements for this detector
        speed = []
        volume = []
        occupancy = []

        # Iterate over each interval element in the XML
        for interval in root.findall('interval'):
            # Extract the attributes 
            raw_speed_ms = float(interval.get('speed'))
            # SUMO detectors set speed to -1.00 if flow is 0. We impute this to 0.0 to align w/ input data.
            if raw_speed_ms < 0:
                speed.append(0.0)
            else:
                speed.append(raw_speed_ms * 2.237) # convert valid speeds to mph
            volume.append(float(interval.get('flow')))
            occupancy.append(float(interval.get('occupancy')))
        
        # Append the measurements for this detector to the detector_data list
        detector_data["speed"].append(speed) # in mph
        detector_data["volume"].append(volume) # in veh/hr
        detector_data["occupancy"].append(occupancy) # in %
    
    for key, val in detector_data.items():
        detector_data[key] = np.array(val)
        # print(val.shape)
    
    detector_data["flow"]=detector_data["volume"]
    # Note: speed can be 0, leading to Inf density. 
    # Suppress divide by zero warning to keep output clean, but allow Infs as per reference logic
    with np.errstate(divide='ignore', invalid='ignore'):
        detector_data["density"] = detector_data["flow"] / detector_data["speed"]
    return detector_data


def flowrouter_rds_to_matrix(data_file, measurement_locations):
    """
    Extract simulated traffic measurements (Q, V) from a single SUMO Flowrouter CSV.
    Mirrors extract_sim_meas logic.
    
    Parameters:
    - data_file: Path to the CSV file ingested by flowrouter/dfrouter (semicolon separated).
    - measurement_locations: List of detector ID strings (e.g. "56_7_3").
    
    Returns:
    - detector_data: Dictionary containing numpy arrays [N_dec x N_time]
    """
    # Read the CSV file
    df = pd.read_csv(data_file, sep=';')
    
    # Initialize container
    detector_data = {"speed": [], "volume": [], "occupancy": []}

    for detector_id in measurement_locations:
        # Convert input "56_7_3" -> CSV target "56.7_3"
        parts = detector_id.split('_')
        target_id = f"{parts[0]}.{parts[1]}_{parts[2]}"
        
        # Filter strictly for this detector
        filtered_df = df[df['Detector'] == target_id]
        
        if filtered_df.empty:
            print(f"No data for detector {target_id}. Skipping this detector.")
            continue
            
        # Sort by time to ensure alignment
        filtered_df = filtered_df.sort_values('Time')
        
        # Extract columns
        # 1. Speed: Convert flowrouter csv's km/h -> mph. (1 km/h = 0.621371 mph).
        speed_vals = filtered_df['vPKW'].values * 0.621371
        
        # 2. Volume: SUMO CSV 'qPKW' is count per interval (30s).
        # To match the output (veh/hr), we multiply count by 120 (since interval is 30s).
        # (3600 seconds / 30 seconds = 120 intervals per hour)
        volume_vals = filtered_df['qPKW'].values * 120
        
        detector_data["speed"].append(speed_vals)
        detector_data["volume"].append(volume_vals)

    # Convert lists to numpy arrays [N_dec, N_time]
    for key, val in detector_data.items():
        # Stack vertically
        detector_data[key] = np.array(val)

    # Compute Derived Physics
    detector_data["flow"] = detector_data["volume"]
    # Note: speed can be 0 (when no vehicles are present), leading to Inf density. 
    # Suppress divide by zero warning to keep output clean, but allow Infs as per reference logic
    with np.errstate(divide='ignore', invalid='ignore'):
        detector_data["density"] = detector_data["flow"] / detector_data["speed"]
        
    return detector_data