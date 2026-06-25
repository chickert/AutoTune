from csv import reader
import subprocess
import xml.etree.ElementTree as ET
import random
import time # For any explicit sleeps retained from the original.
import logging
import os
import re 
import sys
import numpy as np
from datetime import datetime

from param_utils import get_intermediate_file # For accessing current parameter values.

# Add the parent directory to the system path for error_funcs imports
current_script_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_script_dir, '../..'))
sys.path.insert(0, parent_dir)

from error_funcs import MacroscopicErrorCalculator, MicroscopicErrorCalculator, VelocityGridErrorCalculator # For computing error metric(s).

# Add the parent directory to the system path for utils imports
current_script_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_script_dir, '../..'))
sys.path.insert(0, parent_dir)
from utils import generate_time_intervals, generate_time_intervals_seconds, extract_sim_meas, flowrouter_rds_to_matrix


def _restore_vtypes_to_routes_file(routes_file_path, vtypes_to_restore):
    # Restores vType elements to a given routes file, typically after duarouter overwrites it.
    if not vtypes_to_restore:
        logging.debug(f"No vTypes provided to restore in {routes_file_path}.")
        return
    try:
        tree = ET.parse(routes_file_path)
        root = tree.getroot()
        for vtype_str in vtypes_to_restore: # Insert preserved vTypes at the beginning.
            vtype_elem = ET.fromstring(vtype_str)
            root.insert(0, vtype_elem)
        tree.write(routes_file_path, encoding="utf-8", xml_declaration=True)
        # logging.info(f"vType elements successfully restored to {routes_file_path}.")
    except FileNotFoundError:
        logging.warning(f"New {routes_file_path} not found. Could not restore vType elements.")
    except ET.ParseError:
        logging.warning(f"Error parsing new {routes_file_path}. Could not restore vType elements.")

def write_od_flows_and_routes(data_config, od_parameter_structure, solution_idx, intermediate_filename):
    """
    Writes the OD matrix file, then generates trips and routes using od2trips and duarouter.
    """
    param_values_from_yaml = get_intermediate_file(intermediate_filename)
    num_chunks = float(param_values_from_yaml['num_chunks']) # TODO: Double-check chunking scheme.
    # time.sleep(1) # Original sleep, consider if necessary.

    od_matrix_path = data_config["sim_dir_name"] + data_config["calibration_run"] + "/" + str(solution_idx) + "/od_matrix.taz.xml"

    # Construct the directory path from the file path
    dir_path = os.path.dirname(od_matrix_path)

    # Check if the directory exists, if not, create it
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # Get simulation start time and adjust for warmup for the OD matrix.
    sim_start_float = float(data_config['simulation_start'].replace(':', '.'))
    warmup = float(param_values_from_yaml["warmup_time"])
    # Ensure warmup is a multiple of 30 seconds, then convert to minutes and div by 100 for SUMO format.
    effective_warmup = (warmup // 30) * 30 / 60
    # Adjust start time by warmup period -- e.g., 7 o'clock start minus 5-min warmup should be 7.55 in SUMO format, not 7.95
    effective_sim_start = sim_start_float - (effective_warmup / 100) - 0.4 if effective_warmup > 0 else sim_start_float 
    effective_sim_start = max(0.0, effective_sim_start) # start time cannot be negative
    assert effective_warmup < 60, "Should be less than 60 to avoid subtraction error in SUMO format in od_matrix_path file."  #  7 o'clock minus 65-min would be wrong w/ current code
    sim_end = float(data_config['simulation_end'].replace(':', '.'))
    # TODO: Handle this more elegantly
    if 'smallnet' in data_config['network_file']:
        sim_end -= 0.03 # Adjust smallnet inflows to be more consistent with original CorridorCalibration sim, where same is done (inflows end 3min before sim end).  

    # print(od_matrix_path)
    with open(od_matrix_path, 'w') as file: # Write the OD matrix content.
        file.write('$OR;D2\n')
        file.write('* From-Time  To-Time\n')
        file.write(f"{effective_sim_start:.2f} {sim_end:.2f}\n") # SUMO format requires 2 decimal places for time. (Otherwise 7.2 maps to 7:02 instead of 7:20)
        file.write('* Factor\n')
        file.write('1.00\n')
        for key_tuple in od_parameter_structure.keys(): # Uses the structure of OD params for keys.
            file.write(f'{key_tuple[0]} {key_tuple[1]} {float(param_values_from_yaml[str(key_tuple)])/num_chunks}\n')
    # logging.info(f"OD matrix written to {od_matrix_path}")

    taz_zone_path = data_config["sim_dir_name"] + data_config["taz_file"]
    trips_file_path = data_config["sim_dir_name"] + data_config["calibration_run"] + "/" + str(solution_idx) +  data_config["trip_info_file_name"] # Output of od2trips, input for duarouter.
    routes_file_path = data_config["sim_dir_name"] + data_config["calibration_run"] + "/" + str(solution_idx) + data_config["output_routes"]

    cmd_od2trips = ["od2trips", "--net-file", taz_zone_path, "--od-matrix-files", od_matrix_path, "-o", trips_file_path]
    subprocess.run(cmd_od2trips, capture_output=True, text=True, check=True) 
    # logging.info(f"od2trips executed. Trips file: {trips_file_path}")

    existing_vtypes = [] # Preserve vTypes from existing routes file if it exists.
    try:
        tree = ET.parse(routes_file_path)
        root = tree.getroot()
        existing_vtypes = [ET.tostring(vtype, encoding='unicode') for vtype in root.findall("vType")]
    except (FileNotFoundError, ET.ParseError):
        logging.debug(f"{routes_file_path} not found or parse error before duarouter. No vTypes preserved.")

    cmd_duarouter = ["duarouter", "--net-file", data_config["sim_dir_name"] + data_config["network_file"], "--route-files", trips_file_path, "--output-file", routes_file_path]
    subprocess.run(cmd_duarouter, capture_output=True, text=True, check=True) 
    # logging.info(f"duarouter executed. Routes file: {routes_file_path}")

    if existing_vtypes: # Restore preserved vTypes if any.
        _restore_vtypes_to_routes_file(routes_file_path, existing_vtypes)

def configure_vehicle_types_and_routes(data_config, cfm_lc_parameter_structure, solution_idx, intermediate_filename):
    """
    Configures vehicle heterogeneity, creates vType distributions, and updates the main routes file w/ the new vTypes.
    """
    param_values_from_yaml = get_intermediate_file(intermediate_filename)
    variance = float(param_values_from_yaml['variance'])

    cfm_lc_dists_path = data_config["sim_dir_name"] + data_config["calibration_run"] + "/" + str(solution_idx) + "/cfm_lc_dists.txt"
    # Combine parameter definitions from 'cfm' and 'lc' blocks as in original logic.
    cfm_lc_definitions = data_config["block_definitions"]['cfm']['param_list'] + data_config["block_definitions"]['lc']['param_list']

    # Step 1: Compile the capped normal distributions for each parameter.
    with open(cfm_lc_dists_path, "w") as f: # Write vehicle model parameters to cfm_lc_dists.txt.
        for param_key in cfm_lc_parameter_structure.keys(): # Uses structure of cfm/lc params.
            if param_key == 'variance': continue # Variance is a multiplier, not a direct model param here.
            value = float(param_values_from_yaml[param_key])
            bounds_info = next((b_info[param_key] for b_info in cfm_lc_definitions if param_key in b_info), [None, None])
            low, high = bounds_info[0], bounds_info[1]
            f.write(f"{param_key}; normalCapped({value},{value*variance}, {low}, {high})\n")
        f.write("carFollowModel;IDM\n") # Car following model (e.g., IDM) can be configured.
    # logging.info(f"Vehicle configuration text file written to {cfm_lc_dists_path}")

    # Step 2: Build a vTypeDistribution w/ constituent vTypes sampled from the capped parameter distributions
    num_vtypes = data_config['num_vtypes']
    vtype_dist_add_xml_path = data_config["sim_dir_name"] + data_config["calibration_run"] + "/" + str(solution_idx) + "/vTypeDistributions.add.xml"
    # Some versions of createVehTypeDistribution.py struggle with existing files, so remove it if it exists.
    if os.path.exists(vtype_dist_add_xml_path):
        os.remove(vtype_dist_add_xml_path)
    cmd_create_vtype_dist = ["python3", f"{os.environ['SUMO_HOME']}/tools/createVehTypeDistribution.py", # Path to tool might need configuration.
                             cfm_lc_dists_path, "--output-file", vtype_dist_add_xml_path, "--size", str(num_vtypes)]
    # Append seed if it exists
    seed = data_config.get('block_definitions', {}).get('heterogeneity', {}).get('alg_params', {}).get('random_seed', None)
    if seed:
        cmd_create_vtype_dist += ["--seed", str(seed)]
        random.seed(seed)   # Also set seed for 'random' module used later in this func for vtype assignment
    subprocess.run(cmd_create_vtype_dist, capture_output=True, text=True, check=True) # Consider error handling.
    # logging.info(f"createVehTypeDistribution.py executed. Output: {vtype_dist_add_xml_path}")

    # Step 4: Modify the generated vTypes to include speed factor variance.
    tree_vtype_dist = ET.parse(vtype_dist_add_xml_path) # Modify generated vType distributions (e.g., speed factor).
    root_vtype_dist = tree_vtype_dist.getroot()
    for vtype_elem in root_vtype_dist.iter('vType'):
        speed_factor = vtype_elem.get('speedFactor')
        speed_factor = re.search(r'(\d+\.\d+)', speed_factor).group(1)
        if speed_factor is not None: # Apply speed factor variance.
            vtype_elem.set('speedFactor', f'norm({speed_factor},{param_values_from_yaml["speedFactorVariance"]})')
    tree_vtype_dist.write(vtype_dist_add_xml_path, encoding='utf-8', xml_declaration=True)
    # logging.info(f"Modified speedFactors in {vtype_dist_add_xml_path}")

    # Step 5: Update the main routes file with new vTypes and assign them to vehicles.
    routes_file_path = data_config["sim_dir_name"] + data_config["calibration_run"] + "/" + str(solution_idx) + data_config["output_routes"]
    tree_routes = ET.parse(routes_file_path) # Update the main routes file with these new vTypes.
    root_routes = tree_routes.getroot()

    for vtype_elem in root_routes.findall("vType"): root_routes.remove(vtype_elem) # Remove old vTypes.

    vtype_distributions_container = tree_vtype_dist.find("vTypeDistribution")
    num_new_vtypes = 0
    if vtype_distributions_container is not None: # Add new vTypes.
        new_vtypes_list = vtype_distributions_container.findall("vType")
        num_new_vtypes = len(new_vtypes_list)
        for vtype_elem in new_vtypes_list: root_routes.insert(0, vtype_elem)
    
    if num_new_vtypes > 0: # Assign new random vTypes to vehicles.
        for vehicle_elem in root_routes.iter("vehicle"):
            vehicle_elem.set("type", f"vehDist{random.randint(0, num_new_vtypes - 1)}")
    else:
        logging.warning("No new vTypes generated from vTypeDistributions.add.xml to assign to vehicles.")
    tree_routes.write(routes_file_path)
    # logging.info(f"Main routes file {routes_file_path} updated with new vTypes and vehicle assignments.")

def write_sumo_config_files(data_config, current_solution_structures, solution_idx, intermediate_filename):
    """
    Orchestrates the writing of all SUMO input files based on current parameters.
    Draws params FROM the intermediate_filename and writes resulting SUMO files TO the solution_idx directory.
    current_solution_structures provides the keys/structure for OD and CFM/LC params.
    """
    write_od_flows_and_routes(data_config, current_solution_structures.get("od", {}), solution_idx, intermediate_filename)
    print("finished writing od_flows\n")
    configure_vehicle_types_and_routes(data_config, current_solution_structures.get("cfm", {}), solution_idx, intermediate_filename) # cfm structure includes lc params.
    print("finished writing vehicle types and routes\n")
    # logging.info("All SUMO configuration files have been updated/written.")

def set_simulation_time_in_cfg(data_config, start_time_sim, end_time_sim, sumo_cfg_file_path, intermediate_filename, solution_idx):
    # Updates the begin, end, and step-length times in a SUMO configuration file.
    dir_path = data_config["sim_dir_name"] + data_config["calibration_run"] + "/" + str(solution_idx) + "/"

    # Check if the directory exists, if not, create it
    source_file = data_config["sim_dir_name"] + data_config["source_sumocfg_file"]

    # Copy contents from source to destination
    with open(source_file, "r") as src, open(sumo_cfg_file_path, "w") as dst:
        dst.write(src.read())

    params_from_yaml = get_intermediate_file(dir_path + intermediate_filename)
    warmup = float(params_from_yaml["warmup_time"])
    step_len = float(params_from_yaml["step_length"])
    effective_warmup = (warmup // 30) * 30 # Ensure warmup is a multiple of 30 seconds.

    # Start time cannot be negative after warmup adjustment.
    if start_time_sim - effective_warmup < 0:
        effective_warmup = start_time_sim

    try:
        tree = ET.parse(sumo_cfg_file_path)
        root = tree.getroot()
        time_section = root.find('time')
        input_section = root.find('input')
        if time_section is not None: # Set begin, end, and step-length values.
            begin_val_elem = time_section.find('begin'); begin_val_elem.set('value', str(start_time_sim - effective_warmup))
            end_val_elem = time_section.find('end'); end_val_elem.set('value', str(end_time_sim))
            step_len_elem = time_section.find('step-length'); step_len_elem.set('value', str(step_len))
            
            tree.write(sumo_cfg_file_path)
            # logging.info(f"Updated time values in SUMO config: {sumo_cfg_file_path}")
        else:
            logging.error(f"<time> section not found in {sumo_cfg_file_path}.")
    except FileNotFoundError:
        logging.error(f"SUMO config file {sumo_cfg_file_path} not found for time update.")
    except ET.ParseError:
        logging.error(f"Error parsing SUMO config file {sumo_cfg_file_path}.")

def build_sumo_command(data_config, solution_idx, gui_enabled=False, is_eval_run=False):
    # Constructs the command list needed to run a SUMO simulation.
    dir_path = data_config['sim_dir_name'] + data_config['calibration_run']
    worker_dir_path = dir_path + "/" + str(solution_idx)
    sumo_exe = 'sumo-gui' if gui_enabled else 'sumo' # Choose SUMO executable (GUI or command-line).
    
    cmd_list = [sumo_exe, '-c', worker_dir_path + data_config['sumocfg_file_name'],
                '--no-internal-links', '--no-warnings', '--no-step-log', # Common SUMO options.
                '--additional', worker_dir_path + data_config['detector_file'],
                ]
    # Need SUMO's FCD output data for the following error metrics or viz purposes
    if data_config["error_func_type"] == "micro" or data_config["error_func_type"] == "velocity_grid" or (is_eval_run & (data_config["error_func_type"] == "smallnet")) or (is_eval_run & (data_config["error_func_type"] == "mediumnet")): 
        cmd_list.append('--fcd-output')
        cmd_list.append(worker_dir_path + data_config['fcd_output_file'])
        cmd_list.append('--fcd-output.max-leader-distance')
        cmd_list.append('100')
    return cmd_list

def execute_sumo_simulation(sumo_command_list):
    # Executes a SUMO simulation using the provided command list.
    print(sumo_command_list)
    # logging.info(f"Executing SUMO command: {' '.join(sumo_command_list)}")
    result = subprocess.run(sumo_command_list, capture_output=True, text=True, check=True)
    if result.stdout: logging.debug(f"SUMO stdout:\n{result.stdout}")
    if result.stderr: logging.warning(f"SUMO stderr:\n{result.stderr}") # SUMO often uses stderr for info.
    if result.returncode != 0: logging.error(f"SUMO simulation exited with error code {result.returncode}.")
    return result

def run_sim_and_calculate_error_metric(overall_sim_start, overall_sim_end, data_config,
                                       solution_idx, intermediate_filename, is_eval_run=False):
    """
    Runs SUMO simulation in chunks (if specified) and calculates error metrics.
    """
    
    # Get the number of simulation chunks from the solution_idx's intermediate file.
    intermediate_file_path = data_config["sim_dir_name"] + data_config["calibration_run"] + "/" + str(solution_idx) + "/" + intermediate_filename
    params_from_yaml = get_intermediate_file(intermediate_file_path)
    num_sim_chunks = int(float(params_from_yaml.get("num_chunks", 1))) # Default to 1 chunk if not set.
    
    total_metric = 0.0  # Cumulative metric value across chunks, for later averaging.
    
    simulation_duration = overall_sim_end - overall_sim_start
    if simulation_duration <= 0: # Handle zero or negative duration.
        logging.warning("Simulation duration is zero or negative. Skipping simulation runs for error metrics.")
        return 0.0, 0.0
        
    actual_num_chunks = max(1, num_sim_chunks) # Ensure at least one chunk.
    chunk_interval = simulation_duration / actual_num_chunks
    # Ensure interval is a multiple of 30s if it's reasonably large, as in original.
    if chunk_interval >= 30: chunk_interval = (chunk_interval // 30) * 30
    if chunk_interval == 0: chunk_interval = simulation_duration / actual_num_chunks # Recalc if floored to zero.

    current_chunk_start_time = overall_sim_start
    # logging.info(f"Calculating error metrics over {actual_num_chunks} chunk(s), each ~{chunk_interval:.2f}s long.")

    for i in range(actual_num_chunks):
        chunk_sim_start_wout_warmup = current_chunk_start_time
        chunk_sim_end = current_chunk_start_time + chunk_interval
        # Ensure the last chunk ends precisely at overall_sim_end.
        if i == actual_num_chunks - 1 or chunk_sim_end > overall_sim_end:
            chunk_sim_end = overall_sim_end
        
        if chunk_sim_start_wout_warmup >= chunk_sim_end : # Avoid zero-duration chunks if logic leads to it.
            if chunk_sim_start_wout_warmup == overall_sim_end and i > 0 : break # Already processed up to end.
            chunk_sim_end = overall_sim_end # Ensure last bit is processed.
            if chunk_sim_start_wout_warmup >= chunk_sim_end : continue # Still bad, skip.


        sumo_cfg_path = data_config["sim_dir_name"] + data_config["calibration_run"] + "/" + str(solution_idx) + data_config["sumocfg_file_name"]
        # Set simulation time for the current chunk (accounts for warmup via value in intermediate_filename).
        set_simulation_time_in_cfg(data_config, chunk_sim_start_wout_warmup, chunk_sim_end, sumo_cfg_path, intermediate_filename, solution_idx)

        sumo_cmd = build_sumo_command(data_config, solution_idx, gui_enabled=False, is_eval_run=is_eval_run) # Build SUMO command
        execute_sumo_simulation(sumo_cmd) # Run the simulation for this chunk.

        ### Compute and store error metric for this chunk using the specified error calculation function.
        # Macroscopic error calc
        if data_config["error_func_type"] == "macro":
            # Instantiate the macroscopic error calculator
            MacroEC = MacroscopicErrorCalculator()
            # Get the simulation's E1 detector output file path
            sim_e1_xml_filepath = data_config["sim_dir_name"] + data_config["calibration_run"] + "/" + str(solution_idx) + "/out.xml"

            # Get the corresponding INCEPTION detector data. Allow noised or unnoised data for calibration, but always use unnoised for eval.
            if is_eval_run:
                inception_measurements_csv = data_config["data_dir_name"] + data_config["data_path_for_eval"]
            else:
                inception_measurements_csv = data_config["data_dir_name"] + data_config["data_path"]

            if data_config["fitness_criterion"] == "speed_iou":
                # Calculate speed IOU for this chunk
                speed_iou = MacroEC.compute_speed_iou(sim_e1_xml_filepath, inception_measurements_csv)
                total_metric += speed_iou

            elif data_config["fitness_criterion"] == "count_rmse":
                # Exclude count data from detector with known issues in INCEPTION data, but still use its speed data (since unaffected)
                detectors_to_omit_from_counts = data_config.get('detectors_to_omit_from_counts', [])
                speed_count_mae_rmse_dict = MacroEC.compute_speed_count_mae_rmse(sim_e1_xml_filepath, inception_measurements_csv, detectors_to_omit_from_counts=detectors_to_omit_from_counts, detectors_to_omit_from_speeds=[]) 
                count_rmse = speed_count_mae_rmse_dict['count_rmse']
                total_metric += count_rmse
            else:
                # tot_det_count_mae_score = MacroEC.compute_tot_det_count_mae(sim_e1_xml_filepath, inception_measurements_csv, detectors_to_omit=detectors_to_omit_from_counts, verbose=False)
                raise ValueError(f"Unknown {data_config['error_func_type']}-type fitness_criterion {data_config['fitness_criterion']} in config.")
        # Microscopic error calc
        elif data_config["error_func_type"] == "micro":
            ## Get the timestamps for the microscopic objective function
            date_str = data_config["data_path"].split('/')[1] if data_config["data_path"].split('/')[1] else data_config["data_path_for_eval"].split('/')[1]
            full_sim_micro_obj_fn_timestamps = generate_time_intervals(data_config["data_year_str"], date_str, int(overall_sim_start / 60), int(overall_sim_end / 60), data_config["interval_len_in_min"])

            # Filter out the timestamps outside the current chunk
            this_chunk_micro_obj_fn_timestamps = []
            for timestamp in full_sim_micro_obj_fn_timestamps:
                timestamp_dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                timestamp_in_sec = (timestamp_dt.hour * 3600 +
                                timestamp_dt.minute * 60 +
                                timestamp_dt.second)
                if chunk_sim_start_wout_warmup <= timestamp_in_sec < chunk_sim_end:
                    this_chunk_micro_obj_fn_timestamps.append(timestamp)
                elif timestamp_in_sec == chunk_sim_end and chunk_sim_end == overall_sim_end:
                        this_chunk_micro_obj_fn_timestamps.append(timestamp)
    
            if data_config["fitness_criterion"] == "headway_wass_dist":
                MicroEC = MicroscopicErrorCalculator(micro_obj_fn_timestamps=this_chunk_micro_obj_fn_timestamps,
                                fcd_xml_file_path=data_config["sim_dir_name"] + data_config["calibration_run"] + "/" + str(solution_idx) + data_config["fcd_output_file"],
                                inception_data_dir=data_config["inception_data_dir"],
                                net_file_path=data_config["sim_dir_name"] + data_config["network_file"],
                                mm_latlon_mapping_path=data_config["mm_latlon_mapping_path"],
                                cached_traj_dir=data_config["cached_traj_dir"],
                                )

                avg_headway_wass_dist = MicroEC.compute_headway_wasserstein_metric(plot_distributions=False)
                total_metric += avg_headway_wass_dist * len(this_chunk_micro_obj_fn_timestamps)  # Weighted by len of timestamp list (since chunk lengths may vary slightly due to rounding)
            
            else: 
                raise ValueError(f"Unknown {data_config['error_func_type']}-type fitness_criterion {data_config['fitness_criterion']} in config.")

        elif data_config["error_func_type"] == "velocity_grid":
            ## Get the timestamps for the microscopic objective function
            date_str = data_config["data_path"].split('/')[1] if data_config["data_path"].split('/')[1] else data_config["data_path_for_eval"].split('/')[1]
            full_sim_micro_obj_fn_timestamps = generate_time_intervals_seconds(data_config["data_year_str"], date_str, int(overall_sim_start), int(overall_sim_end), data_config["interval_len_in_sec"])

            # Filter out the timestamps outside the current chunk
            this_chunk_micro_obj_fn_timestamps = []
            for timestamp in full_sim_micro_obj_fn_timestamps:
                timestamp_dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                timestamp_in_sec = (timestamp_dt.hour * 3600 +
                                timestamp_dt.minute * 60 +
                                timestamp_dt.second)
                if chunk_sim_start_wout_warmup <= timestamp_in_sec < chunk_sim_end:
                    this_chunk_micro_obj_fn_timestamps.append(timestamp)
                elif timestamp_in_sec == chunk_sim_end and chunk_sim_end == overall_sim_end:
                        this_chunk_micro_obj_fn_timestamps.append(timestamp)

            if data_config["fitness_criterion"] == "mape":
                velocity_grid_error_calculator = VelocityGridErrorCalculator(micro_obj_fn_timestamps=this_chunk_micro_obj_fn_timestamps,
                                fcd_xml_file_path=data_config["sim_dir_name"] + data_config["calibration_run"] + "/" + str(solution_idx) + data_config["fcd_output_file"],
                                net_file_path=data_config["sim_dir_name"] + data_config["network_file"],
                                mm_latlon_mapping_path=data_config["mm_latlon_mapping_path"],
                                )
                
                wb_ground_truth_velocity = np.load("../../reference_data/wb_velocity_7_to_8_trimmed.npy")
                # eb_ground_truth_velocity = np.load("../../reference_data/eb_velocity__7_to_8_10s_400m_unflipped.npy")
                # eb_ground_truth_velocity = eb_ground_truth_velocity[:,3:]

                eb_predicted_vel_npy, wb_predicted_vel_npy = velocity_grid_error_calculator.compute_velocity_grid_from_fcd_streaming()
                wb_predicted_vel_npy = wb_predicted_vel_npy[:,:-10]
                # eb_predicted_vel_npy = eb_predicted_vel_npy[:,3:]

                mape_wb = velocity_grid_error_calculator.mape(wb_ground_truth_velocity, wb_predicted_vel_npy)
                # mape_eb = velocity_grid_error_calculator.mape(eb_ground_truth_velocity, eb_predicted_vel_npy)

                # print("EB mape:", mape_eb)
                print("WB mape:", mape_wb)

                # total_metric += (mape_wb + mape_eb) / 2 * len(this_chunk_micro_obj_fn_timestamps)  # Weighted by len of timestamp list (since chunk lengths may vary slightly due to rounding)
                total_metric += mape_wb * len(this_chunk_micro_obj_fn_timestamps)
                # print("total mape:", total_metric)

            else: 
                raise ValueError(f"Unknown {data_config['error_func_type']}-type fitness_criterion {data_config['fitness_criterion']} in config.")

        elif data_config["error_func_type"] == "smallnet":

            measure = data_config["fitness_criterion"].split('_')[0]  # "speed" or "volume" or "occupancy"

            measurement_locations = ['upstream_0', 'upstream_1', 
                            'merge_0', 'merge_1', 'merge_2', 
                            'downstream_0', 'downstream_1']

            # Get the ground truth data (here, synthetic measurements from the smallnet sim)
            synthetic_gt_dir = os.path.join(data_config["data_dir_name"], data_config["data_path"].split('/')[1])
            synthetic_gt_output = extract_sim_meas(measurement_locations, file_dir=synthetic_gt_dir)

            # Extract simulated traffic volumes
            solution_idx_dir = os.path.join(data_config["sim_dir_name"] + data_config["calibration_run"], str(solution_idx))
            simulated_output = extract_sim_meas(["trial_"+ location for location in measurement_locations], file_dir=solution_idx_dir)

            # Handle various actual_num_chunks possibilities
            assert actual_num_chunks in [1, 2, 4], f"Unexpected actual_num_chunks {actual_num_chunks} for smallnet config."
            target_len = synthetic_gt_output[measure].shape[1] // actual_num_chunks # Get number of time measurements to compare this chunk
            sim_vals = simulated_output[measure][:, -target_len:] # Exclude warm-up period from sim data, if present
            gt_start_idx = i * target_len
            gt_end_idx = gt_start_idx + target_len
            synthetic_gt_vals = synthetic_gt_output[measure][:, gt_start_idx:gt_end_idx] # Get corresponding segment of ground truth data

            # RMSE
            assert data_config["fitness_criterion"].split('_')[1] == "rmse"
            diff = sim_vals - synthetic_gt_vals # measured output may have nans
            rmse = np.sqrt(np.nanmean(diff.flatten()**2))
            total_metric += rmse

        elif data_config["error_func_type"] == "mediumnet":

            measure = data_config["fitness_criterion"].split('_')[0]  # "speed" or "volume" or "occupancy"

            measurement_locations = [
                         '56_7_0', '56_7_1', '56_7_2', '56_7_3', '56_7_4', 
                         '56_3_0', '56_3_1', '56_3_2', '56_3_3', '56_3_4',
                         '56_0_0', '56_0_1', '56_0_2', '56_0_3', '56_0_4',
                         '55_3_0', '55_3_1', '55_3_2', '55_3_3',
                         '54_6_0', '54_6_1', '54_6_2', '54_6_3',
                        ]
        
            # Get the ground truth data
            if is_eval_run:
                rds_data_path = data_config["data_dir_name"] + data_config["data_path_for_eval"]
            else:
                rds_data_path = data_config["data_dir_name"] + data_config["data_path"]
            gt_output = flowrouter_rds_to_matrix(rds_data_path, measurement_locations)

            # Extract simulated traffic volumes
            solution_idx_dir = os.path.join(data_config["sim_dir_name"] + data_config["calibration_run"], str(solution_idx))
            simulated_output = extract_sim_meas(measurement_locations, file_dir=solution_idx_dir)

            # Handle various actual_num_chunks possibilities
            assert actual_num_chunks in [1, 2, 4], f"Unexpected actual_num_chunks {actual_num_chunks} for mediumnet config."
            target_len = gt_output[measure].shape[1] // actual_num_chunks # Get number of time measurements to compare this chunk
            sim_vals = simulated_output[measure][:, -target_len:] # Exclude warm-up period from sim data, if present
            gt_start_idx = i * target_len
            gt_end_idx = gt_start_idx + target_len
            gt_vals = gt_output[measure][:, gt_start_idx:gt_end_idx] # Get corresponding segment of ground truth data

            # RMSE
            assert data_config["fitness_criterion"].split('_')[1] == "rmse"
            # Handle edge case when optimizing step_length param, where sim and gt shapes may slightly differ.
            if sim_vals.shape != gt_vals.shape:
                # This occurs because non-integer step lengths can lead to slight lag in sim detector readings (e.g., 30.1 sec instead of 30 sec intervals).
                # Although this can lead to very slight accumulating delay later in sim, it is negligible for most settings of interest. 
                num_time_diff = gt_vals.shape[1] - sim_vals.shape[1]
                gt_vals = gt_vals[:, :-num_time_diff] # Truncate gt_vals to match sim_vals length.
            diff = sim_vals - gt_vals # measured output may have nans
            rmse = np.sqrt(np.nanmean(diff.flatten()**2))
            total_metric += rmse

        else: 
            raise ValueError(f"Unknown error_func_type {data_config['error_func_type']} in config.")

        current_chunk_start_time += chunk_interval
        if current_chunk_start_time >= overall_sim_end: break # Stop if we've covered the whole period.

    if data_config["error_func_type"] == "macro" or data_config["error_func_type"] == "smallnet" or data_config["error_func_type"] == "mediumnet":
        avg_metric = total_metric / actual_num_chunks
    elif data_config["error_func_type"] == "micro" or data_config["error_func_type"] == "velocity_grid":
        avg_metric = total_metric / len(full_sim_micro_obj_fn_timestamps)
    else:
        raise ValueError(f"Unknown error_func_type {data_config['error_func_type']} in config.")

    logging.info(f"Average {data_config['fitness_criterion']} over {actual_num_chunks} chunks at solution_idx {solution_idx}: {avg_metric:.4f}")
    return avg_metric
