import os
import re
import csv
import sys
import datetime
import xml.etree.ElementTree as ET
from color_utils import color_vehs, color_flows
# Add the parent directory to the system path for importing the ErrorCalculator class
current_script_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_script_dir, '../..'))
sys.path.insert(0, parent_dir)
from utils import generate_time_intervals_seconds
from error_funcs import MacroscopicErrorCalculator, MicroscopicErrorCalculator

### Paths that generally won't change
shared_sim_dir = "sim_files"
baseline_support_files_dir_name = 'cal_methods/sumo_baseline/support_files'
network_file = 'sumo_test.net.xml'
network_file_path = os.path.join(shared_sim_dir, network_file)
detector_file_path = os.path.join(baseline_support_files_dir_name, 'sumo_test_detectors.xml')
# Output files for flowrouter
output_routes_path = os.path.join(baseline_support_files_dir_name, 'sumo_baseline.rou.xml')
output_flows_path = os.path.join(baseline_support_files_dir_name, 'sumo_baseline.flows.xml')
# Output files from simulation
outputs_dir = 'cal_methods/sumo_baseline/outputs'
fcd_output_file = 'fcd_output.xml'
fcd_output_file_path = os.path.join(outputs_dir, fcd_output_file)
sim_e1_xml_file = 'e1_output.xml' # Note that this is for reading; for writing, location is defined in sim's detectors.xml file
sim_e1_xml_filepath = os.path.join(outputs_dir, sim_e1_xml_file)
# Path to location where to save results CSV file
current_date = datetime.date.today().strftime("%Y-%m-%d")
save_file_name = f"sumo-baseline_full-results_generated-{current_date}.csv"
save_file_path = os.path.join(outputs_dir, save_file_name)
# Delete the results file if it exists
if os.path.exists(save_file_path):
    os.remove(save_file_path)
    print(f"Existing results file {save_file_path} deleted.")

# Grab all CSV files in the detector_measurements_scenarios directory and iterate over them to evaluate each
detector_measurements_dir = 'detector_measurements_scenarios'

if True:
# for f_name in sorted(os.listdir(detector_measurements_dir)):
    # Skip non-CSV files
    # if not f_name.endswith('.csv'):
    #     continue

    # Note: .sumocfg file will automatically set the sim's begin and end times based on the CSV file
    inception_measurements_csv = 'detector_measurements/1130/detections_0360-0600.csv' # Example path to data on which to evaluate
    # inception_measurements_csv = "detector_measurements/1130/scen3_unnoised_1130_0420-0480.csv"

    ### Modify if want to use an abridged simulation (e.g., for quicker testing)
    end_time_in_min = None 
    # end_time_in_min = 430
    # if end_time_in_min:
    #     print(f"\n\nNote: Using user-specified simulation end time of {end_time_in_min} minutes\n\n")

    # Set aggregation interval (in minutes) for the flowrouter method in SUMO 
    #(Also need to make explicit to handle sims that start at times other than 0:00)
    aggregation_interval = 60   # minutes 


    ### Generate flows using SUMO's flowrouter method (SUMO's method of reconstructing routes and traffic flows) 
    # From documentation: 'calculates a set of routes -o and traffic flows -e from given detectors -d and their measurements -f on a given network (option -n)'
    os.system(f"python3 {os.environ['SUMO_HOME']}/tools/detector/flowrouter.py -n {network_file_path} -d {detector_file_path} -f {inception_measurements_csv} -o {output_routes_path} -e {output_flows_path} -i {aggregation_interval} --respect-zero --revalidate-detectors --quiet")

    ### Prep simulation config file
    sumo_cmd = 'sumo'   # 'sumo' for regular running; 'sumo-gui' to see GUI 
    sumocfg_file = 'sumo_config.sumocfg'
    sumocfg_file_path = os.path.join(baseline_support_files_dir_name, sumocfg_file)
    if end_time_in_min is None:
        begin_time_in_min, end_time_in_min = map(int, re.search(r'(\d+)-(\d+)\.csv', inception_measurements_csv).groups())
    else:
        begin_time_in_min, _ = map(int, re.search(r'(\d+)-(\d+)\.csv', inception_measurements_csv).groups())
    begin_time_in_sec = begin_time_in_min * 60
    end_time_in_sec = end_time_in_min * 60

    # Update the .sumocfg file w/ the correct begin and end times
    tree = ET.parse(sumocfg_file_path)
    time_element = tree.getroot().find('time')
    if time_element:
        time_element.find('begin').set('value', str(begin_time_in_sec))
        time_element.find('end').set('value', str(end_time_in_sec))
        tree.write(sumocfg_file_path)

    ### Run simulation
    # Define the base command arguments as a list
    sumo_command_parts = [
        sumo_cmd,
        '-c', sumocfg_file_path,
        # '--fcd-output', fcd_output_file_path,
        # '--fcd-output.max-leader-distance', '100',
    ]

    # If using GUI, update visualization to improve realism
    if sumo_cmd == '':
        maptiles_dir = os.path.join(shared_sim_dir, 'maptiles')
        # Check if maptiles dir exists
        if not os.path.exists(maptiles_dir):
            print(f"Directory '{maptiles_dir}' does not exist. Creating it...")
            os.makedirs(maptiles_dir) # Create the directory if it doesn't exist
        # Now check if it's empty
        if not os.listdir(maptiles_dir):
            print(f"The directory '{maptiles_dir}' is empty. Calling the command to get map tiles...")
            os.system(f'python /opt/homebrew/opt/sumo/share/sumo/tools/tileGet.py -n sim_files/sumo_test.net.xml -d sim_files/maptiles/ --maptype satellite --tiles 2000 --min-file-size 2000')
            print(f"Map tiles downloaded to '{maptiles_dir}'.")
        
        # # If the command is 'sumo-gui', add the maptiles directory to the command
        sumo_command_parts.append('-g')
        sumo_command_parts.append(os.path.join(maptiles_dir, 'settings.xml'))

        # Update the vehicle colors to be more realistic
        vtype_dist_id = "realistic_colors_vtypes_dist"
        color_vehs(output_routes_path, vtype_dist_id)
        color_flows(output_flows_path, vtype_dist_id)

    # Join the list parts into a single string
    final_sumo_command = " ".join(sumo_command_parts)
    # Execute the command
    os.system(final_sumo_command)

    ### Evaluate simulation 
    MacroEC = MacroscopicErrorCalculator()

    # Exclude count data from detector with known issues in INCEPTION data, but still use its speed data (since unaffected)
    detectors_to_omit_from_counts = []

    # Store result name for reference in results CSV
    result_name = inception_measurements_csv.split('/')[1].split('.')[0]

    # Use the unnoised detector measurements for evaluation
    if 'wnoise' in inception_measurements_csv:
        inception_measurements_csv = inception_measurements_csv.replace('wnoise', 'unnoised')

    speed_iou_score = MacroEC.compute_speed_iou(sim_e1_xml_filepath, inception_measurements_csv)
    tot_det_count_mae_score = MacroEC.compute_tot_det_count_mae(sim_e1_xml_filepath, inception_measurements_csv, detectors_to_omit=detectors_to_omit_from_counts, verbose=False)
    speed_count_mae_rmse_score = MacroEC.compute_speed_count_mae_rmse(sim_e1_xml_filepath, inception_measurements_csv, detectors_to_omit_from_counts=detectors_to_omit_from_counts, detectors_to_omit_from_speeds=[])

    # Get the timestamps for the microscopic objective function
    year_str = '2022'
    interval_len_in_sec = 10 # Frequency of timestamps for microscopic objective function (inclusive)
    date_str = os.path.basename(inception_measurements_csv).split('_')[2]
    micro_obj_fn_timestamps = generate_time_intervals_seconds(year_str, date_str, begin_time_in_min, end_time_in_min, interval_len_in_sec)

    # Calculate microscopic error
    MicroEC = MicroscopicErrorCalculator(micro_obj_fn_timestamps=micro_obj_fn_timestamps,
                    fcd_xml_file_path=fcd_output_file_path,
                    inception_data_dir="./i24_data/",
                    net_file_path=network_file_path,
                    mm_latlon_mapping_path="build_data/mile_marker_layer.csv",
                    cached_traj_dir="./cal_methods/sumo_baseline/cached_trajectories/",
                    cache_traj_verbosity=True,
                    cache_traj_disable_tqdm=False
                    )

    avg_wass_dist = MicroEC.compute_headway_wasserstein_metric(plot_distributions=True)

    print(f"\nResults for measurements from '{result_name}':")
    print(f"\tSpeed IoU Score: {speed_iou_score}")
    print(f"\tTotal Detector Count MAE score: {tot_det_count_mae_score}")
    print(f"\tSpeed Count MAE/RMSE Score: {speed_count_mae_rmse_score}")
    print(f"\tMicro Error computed at times: {micro_obj_fn_timestamps}")
    print(f"\tAverage Per-Segment Wasserstein Distance for Headways: {avg_wass_dist}")

    ### Save results to CSV
    # Define the headers for the CSV file
    headers = [
        "Inception Measurements",
        "Speed IoU Score",
        "Total Detector Count MAE score",
        "Average Per-Segment Wasserstein Distance for Headways",
        "speed_mae",
        "speed_rmse",
        "count_mae",
        "count_rmse"
    ]

    # Prepare the data row from the provided variables
    data_row = [
        result_name, # Just the scenario details
        speed_iou_score,
        tot_det_count_mae_score,
        avg_wass_dist,
        speed_count_mae_rmse_score['speed_mae'],
        speed_count_mae_rmse_score['speed_rmse'],
        speed_count_mae_rmse_score['count_mae'],
        speed_count_mae_rmse_score['count_rmse']
    ]

    try:
        # Check if the file exists to decide whether to write headers
        file_exists = os.path.isfile(save_file_path)

        # Append data to the CSV file (append mode 'a')
        with open(save_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(headers) # Write headers if file doesn't exist
            writer.writerow(data_row) # Write the data row
        print(f"\tData successfully appended to {save_file_path}\n")

    except IOError as e:
        print(f"Error writing to file: {e}")
