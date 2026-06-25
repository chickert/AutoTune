import os
import sys
import traci
import xml.etree.ElementTree as ET

# Add the parent directory to the system path for importing the ErrorCalculator class
current_script_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_script_dir, '../..'))
sys.path.insert(0, parent_dir)
PROJECT_DIR = parent_dir

from error_funcs import MacroscopicErrorCalculator


def resulCorrection(Warmup_duration, result_sim_detectors,output_folder): #sample functio for correcting detector results

    warmup_time = 60 * Warmup_duration

    
    tree = ET.parse(result_sim_detectors)
    root = tree.getroot()


    # Iterate over the existing detector results
    for interval in list(root.findall("interval")):
        begin = float(interval.get("begin"))
        end = float(interval.get("end"))
        detector_id = interval.get("id")

        # Remove first 30 minutes for all detectors
        if begin < warmup_time:
            root.remove(interval)
            continue

        # Shift time back by warm-up duration
        new_begin = begin - warmup_time
        new_end = end - warmup_time

        interval.set("begin", str(new_begin))
        interval.set("end", str(new_end))
    

    
    os.makedirs(output_folder, exist_ok=True)
    modified_detector_results = os.path.join(output_folder, 'modified_detector_results.xml')


    tree.write(modified_detector_results,
               encoding="UTF-8", xml_declaration=True)
    
########################


def run_simulation(support_files_folder,config_file=None, cfg_name='opt.sumocfg.xml'):

    """end_time = config_file["end_time"]
    start_time = config_file["start_time"]
    sim_step_length = config_file["sim_step_length"]
    warmup_duration = config_file["warmup_duration"]
    real_data_step = config_file["real_data_step"]"""

    end_time = 540
    start_time = 420.5
    sim_step_length = 1
    warmup_duration = 20
    real_data_step = 0.5

   # print('running simulation')
    new_cfg_path = os.path.join(support_files_folder, cfg_name)
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # sumoCmd = ["sumo-gui", "-c", new_cfg_path, "--start", "--no-step-log", "--no-warnings","--seed", "42", '--fcd-output', './outputs/fcd_output.xml', '--fcd-output.max-leader-distance', '100']
    sumoCmd = ["sumo-gui", "-c", new_cfg_path, "--start", "--no-step-log", "--no-warnings","--seed", "42"]
    # traci.start(sumoCmd)
    final_sumo_command = " ".join(sumoCmd)
    os.system(final_sumo_command)
    # traci.gui.setSchema("View #0", "real world")

    # # hours, minutes = map(int, start_time.split(':'))
    # start_time_seconds = start_time * 60

    # # hours, minutes = map(int, end_time.split(':'))
    # end_time_seconds = end_time * 60

    # sim_end_time = end_time_seconds-start_time_seconds+ real_data_step*60+ warmup_duration*60

    # step = 0
    # while step <= sim_end_time*(1/sim_step_length):
    #     # time.sleep(0.5);
    #     traci.simulationStep()

    #     step = step+1

    # traci.close()



date_str = "1128"
option = 'opt2'

support_files_folder = os.path.join(PROJECT_DIR, "cal_methods", "bilevel", "support_files", date_str, option)
run_simulation(support_files_folder,config_file=None, cfg_name='opt.sumocfg.xml')

MacroEC = MacroscopicErrorCalculator()

sim_e1_xml_filepath = os.path.join(support_files_folder, "detector_results.xml")
inception_measurements_csv = f'detector_measurements/{date_str}/detections_0360-0600.csv'

detectors_to_omit_from_counts = ['563-westbound']

speed_count_mae_rmse_score = MacroEC.compute_speed_count_mae_rmse(sim_e1_xml_filepath, inception_measurements_csv, detectors_to_omit_from_counts=detectors_to_omit_from_counts, detectors_to_omit_from_speeds=[])


print(f"\tSpeed Count MAE/RMSE Score: {speed_count_mae_rmse_score}")



