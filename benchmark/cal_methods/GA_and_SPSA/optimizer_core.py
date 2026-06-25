import yaml
import logging
import threading # For creating a lock.
import pandas as pd
import numpy as np # For isnan checks.
import os
import shutil

# Custom module imports.
from param_utils import set_intermediate_file
from sumo_utils import write_sumo_config_files, run_sim_and_calculate_error_metric # For preparing SUMO for evaluations.
# For running evaluations and bound tests; also need the actual error function (e.g., get_counts_geh).
from blocks import Block # The Block class that instantiates and runs a specific model.
import xml.etree.ElementTree as ET

class Optimizer:
    # Main class to manage and orchestrate the multi-block optimization process.
    def __init__(self, main_config_yaml_path):
        self.yaml_config_path = main_config_yaml_path
        self.global_config_data = self._load_main_configuration()

        # Make a dir for this run, based on data path (if not already existing).
        self.global_config_data["calibration_run"] = '/' + self.global_config_data['data_path'].split('/')[2].replace('.csv', '')
        os.makedirs(self.global_config_data["sim_dir_name"] + self.global_config_data["calibration_run"], exist_ok=True)
        os.makedirs(self.global_config_data['checkpoints_dir'] + self.global_config_data['calibration_run'], exist_ok=True)

        self._parse_simulation_time_settings()
        self._initialize_logging_system()
        
        self.calibration_blocks = {}      # Stores initialized Block instances, keyed by block_type.
        # Stores the best solution values found for each parameter, structured for SUMO file writing.
        # e.g., {"od": {('taz1','taz2'): "val"}, "cfm": {"accel": "val"}, ...}
        self.current_best_solution_struct = {}
        # Stores metadata about parameters for each block: {block_type: {'names': [], 'config_for_model': [], 'count': X}}
        self.block_parameter_metadata = {}
        
        self._prepare_calibration_blocks_and_params() # Parses block definitions and param ranges, initializes parameters.
        # TODO: Update both these to use sol_per_pop, (not ga_workers), since # of directories corresponds to solutions per population.
        #   (In the meantime, simply setting sol_per_pop = ga_workers works.)
        self._write_initial_intermediate_params_file(   # Sets up intermediate_yaml_filename with values initialized above.
            self.global_config_data["ga_workers"], 
            self.global_config_data["intermediate_yaml_filename"]
            ) 
        self._write_initial_detector_files_to_workers(self.global_config_data, self.global_config_data["ga_workers"])

    def _load_main_configuration(self):
        # Loads the primary YAML configuration file for the optimizer.
        try:
            with open(self.yaml_config_path) as stream:
                cfg_data = yaml.safe_load(stream)
            return cfg_data
        except FileNotFoundError:
            logging.critical(f"FATAL: Configuration file {self.yaml_config_path} not found.")
            raise
        except yaml.YAMLError as exc:
            logging.critical(f"FATAL: Error parsing YAML file {self.yaml_config_path}: {exc}")
            raise

    def _parse_simulation_time_settings(self):
        # Converts simulation start/end times from "HH:MM" string format to seconds.
        start_h, start_m = map(int, self.global_config_data["simulation_start"].split(":"))
        self.simulation_start_seconds = start_h * 3600 + start_m * 60
        end_h, end_m = map(int, self.global_config_data["simulation_end"].split(":"))
        self.simulation_end_seconds = end_h * 3600 + end_m * 60

    def _initialize_logging_system(self):
        # Configures the logging system for the optimization process.
        log_output_file = self.global_config_data["optimizer_log_dir"] + '/' + self.global_config_data["data_path"].split('/')[2].replace('csv','log')
        # Remove any existing handlers to avoid duplicate logs if re-initialized.
        for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
        logging.basicConfig(filename=log_output_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
        logging.info("Optimizer logging system initialized.")
        logging.info(f"Using configuration file from: {self.yaml_config_path}")
        return

    def _prepare_calibration_blocks_and_params(self):
        # Initializes Block objects and extracts parameter information from the main config.
        defined_blocks_in_config = self.global_config_data["block_definitions"]
        # This dictionary will hold the initial values for all parameters for intermediate.yaml.
        self.all_initial_parameter_values = {}

        for block_name_key, block_yaml_config in defined_blocks_in_config.items():
            chosen_algorithm = block_yaml_config['alg'] # e.g., "genetic_alg", "spsa".
            # Algorithm-specific params (e.g., num_generations, SPSA factors).
            algorithm_config_params = block_yaml_config.get('alg_params', {})
            # Pass a reference to the error function (can be made configurable per block).
            # algorithm_config_params['error_function_to_use'] = get_counts_geh # Example
            
            current_block_param_names = []
            # This will be the 'gene_space' for GA or 'bounds' config for SPSA.
            current_block_param_config_for_model = [] 
            
            if block_name_key == "od": # Special handling for OD block parameters.
                flow_min_max = block_yaml_config['flow_ranges'] # [min_flow, max_flow].
                all_taz_pairs = [] # List of unique (source, sink) tuples.
                for taz_group_desc in block_yaml_config["taz_description"]:
                    src_tazs, sink_tazs, ordered_tazs = taz_group_desc["sources"], taz_group_desc["sinks"], taz_group_desc["order"]
                    # Iterate through ordered TAZs to create valid source-sink pairs.
                    for i_idx in range(len(ordered_tazs)):
                        if ordered_tazs[i_idx] not in src_tazs: continue # Must be a source, so we skip if it is not.
                        for j_idx in range(i_idx, len(ordered_tazs)): # Check all *following* TAZs as potential sinks.
                            # Ensure sink is different from source and is in the defined sink list for this group.
                            if ordered_tazs[j_idx] in sink_tazs and ordered_tazs[i_idx] != ordered_tazs[j_idx]:
                                all_taz_pairs.append((ordered_tazs[i_idx], ordered_tazs[j_idx]))
                
                unique_od_pairs = sorted(list(set(all_taz_pairs))) # Ensure unique and consistently ordered pairs.
                for od_pair_tuple in unique_od_pairs:
                    param_name_as_str = str(od_pair_tuple) # Store TAZ pair as a string key for YAML.
                    current_block_param_names.append(param_name_as_str)
                    # Format for GA gene_space or SPSA bounds.
                    current_block_param_config_for_model.append({"low": flow_min_max[0], "high": flow_min_max[1]})
                    # Initial value for this OD pair flow.
                    self.all_initial_parameter_values[param_name_as_str] = (flow_min_max[0] + flow_min_max[1]) / 2.0
            else: # Handling for other block types (cfm, lc, heterogeneity, etc.).
                params_list_from_yaml = block_yaml_config["param_list"] # List of dicts: [{'param_name': [val1, val2, ...]}, ...].
                for param_definition_entry in params_list_from_yaml:
                    p_yaml_name = list(param_definition_entry.keys())[0]
                    p_yaml_config = list(param_definition_entry.values())[0] # e.g., [low, high] or list of discrete values.
                    
                    current_block_param_names.append(p_yaml_name)

                    # For sim params, we provide the list itself as the config (since discrete) and select the first element in the list as the default
                    if block_name_key == "sim":
                        # Initialize num_chunks with a list since it's a discrete set, but use low/high dict for warmup_time and step_length
                        if p_yaml_name == "num_chunks":
                            current_block_param_config_for_model.append(p_yaml_config) # Store raw config.
                        else:
                            current_block_param_config_for_model.append({"low": p_yaml_config[0], "high": p_yaml_config[1]})
                        # Handle step_length to default to 1.0 if present in list, else first element.
                        if p_yaml_name == "step_length" and 1.0 in p_yaml_config:
                            self.all_initial_parameter_values[p_yaml_name] = 1.0
                        else:
                            # Set initial value from first element if list, or value itself.
                            self.all_initial_parameter_values[p_yaml_name] = p_yaml_config[0] if isinstance(p_yaml_config, list) else p_yaml_config
                    # For heterogeneity params, we specify a range as the config (since continuous) and select the first element in the list as the default
                    elif block_name_key == "heterogeneity":
                    # if p_yaml_name in ['step_length', 'num_chunks', 'warmup_time', 'variance', 'speedFactorVariance']:
                        current_block_param_config_for_model.append({"low": p_yaml_config[0], "high": p_yaml_config[1]})
                        # Set initial value from first element if list, or value itself.
                        self.all_initial_parameter_values[p_yaml_name] = p_yaml_config[0] if isinstance(p_yaml_config, list) else p_yaml_config
                    # For standard numeric parameter with low/high bounds for optimization (CFM & LC), we allow a range and take the midpoint (mean) as the default
                    elif isinstance(p_yaml_config, list) and len(p_yaml_config) == 2:
                        current_block_param_config_for_model.append({"low": p_yaml_config[0], "high": p_yaml_config[1]})
                        self.all_initial_parameter_values[p_yaml_name] = (p_yaml_config[0] + p_yaml_config[1]) / 2 # Default to midpoint
                    else: 
                        raise NotImplementedError(f"Unsupported parameter config for block '{block_name_key}': {p_yaml_config}. Expected range or discrete values.")

            # Store metadata for this block.
            self.block_parameter_metadata[block_name_key] = {
                'names': current_block_param_names, # List of parameter names this block optimizes.
                'config_for_model': current_block_param_config_for_model, # Structure for model's search space.
                'count': len(current_block_param_names)
            }
            
            # Initialize structure for `current_best_solution_struct` based on block type,
            # This mirrors the structure expected by `write_sumo_config_files` and original `write_od_file`.
            # NOTE: The values in current_best_solution_struct are NOT used directly in the optimization!
            if block_name_key == "od":
                self.current_best_solution_struct["od"] = {} # OD parameters are grouped under "od".
                for taz_pair_str_key in current_block_param_names:
                    try: # Convert string "(src,dst)" back to tuple (src,dst) for the key.
                        actual_tuple_key = eval(taz_pair_str_key)
                        self.current_best_solution_struct["od"][actual_tuple_key] = "0" # Placeholder value.
                    except SyntaxError: # Fallback if eval fails (e.g. malformed string).
                         logging.warning(f"Could not parse TAZ pair string '{taz_pair_str_key}' to tuple for OD solution structure. Using string key.")
                         self.current_best_solution_struct["od"][taz_pair_str_key] = "0"

            elif block_name_key == "cfm" or block_name_key == "lc": # CFM and LC parameters are grouped under "cfm".
                if "cfm" not in self.current_best_solution_struct: self.current_best_solution_struct["cfm"] = {}
                for p_name_item in current_block_param_names:
                    self.current_best_solution_struct["cfm"][p_name_item] = "0" # Placeholder.
            else: # For other blocks like 'heterogeneity'.
                self.current_best_solution_struct[block_name_key] = {}
                for p_name_item in current_block_param_names:
                    self.current_best_solution_struct[block_name_key][p_name_item] = "0"
                    
            # Instantiate the Block, which will in turn instantiate its specific optimization model.
            self.calibration_blocks[block_name_key] = Block(
                block_type_id=block_name_key,
                parameter_names_list=current_block_param_names,
                parameter_config_for_model=current_block_param_config_for_model,
                chosen_model_type_str=chosen_algorithm,
                sim_start_secs=self.simulation_start_seconds,
                sim_end_secs=self.simulation_end_seconds,
                global_data_config=self.global_config_data,
                all_current_block_solutions=self.current_best_solution_struct, # Pass full dict for context.
                block_specific_alg_params=algorithm_config_params
            )
            logging.info(f"Initialized calibration block: '{block_name_key}' with {len(current_block_param_names)} parameters, using '{chosen_algorithm}'.")
        
    def _write_initial_detector_files_to_workers(self, data_config, num_workers):
        global_path_to_detector = data_config["sim_dir_name"] + data_config["detector_file"]
        if num_workers == 0:
            target_file = data_config["sim_dir_name"] + data_config["calibration_run"] + "/" + str(0) + data_config["detector_file"]
            shutil.copy(global_path_to_detector, target_file)
        else:
            for i in range(num_workers):
                target_dir =  data_config["sim_dir_name"] + data_config["calibration_run"] + "/" + str(i)
                os.makedirs(target_dir, exist_ok=True)
                target_file = target_dir + data_config["detector_file"] 
                shutil.copy(global_path_to_detector, target_file)

    def _write_initial_intermediate_params_file(self, num_workers, intermediate_yaml_filename):
        # Writes all collected initial parameter values to the intermediate YAML file.
        intermediate_yaml_filepath = os.path.join(self.global_config_data["sim_dir_name"] + self.global_config_data["calibration_run"], intermediate_yaml_filename)
        with open(intermediate_yaml_filepath, 'w') as file:
            yaml.dump(self.all_initial_parameter_values, file, default_flow_style=False, sort_keys=False)

        # Copy the initial parameters to each worker's directory.
        if num_workers == 0:
            target_dir = self.global_config_data["sim_dir_name"] + self.global_config_data["calibration_run"] + "/" + str(0)
            os.makedirs(target_dir, exist_ok=True)
            target_file = os.path.join(target_dir, intermediate_yaml_filename)
            with open(target_file, 'w') as file:
                yaml.dump(self.all_initial_parameter_values, file, default_flow_style=False, sort_keys=False)
        else:
            for i in range(num_workers):
                target_dir = self.global_config_data["sim_dir_name"] + self.global_config_data["calibration_run"] + "/" + str(i)
                os.makedirs(target_dir, exist_ok=True)
                target_file = os.path.join(target_dir, intermediate_yaml_filename)
                with open(target_file, 'w') as file:
                    yaml.dump(self.all_initial_parameter_values, file, default_flow_style=False, sort_keys=False)

    def run_full_optimization_process(self, num_optimizer_iterations=2):
        # Executes the main optimization loop, iterating through all blocks multiple times.
        logging.info(f"Starting full optimization process for {num_optimizer_iterations} iteration(s).")
        # Stores detailed statistics from each block run.
        all_run_statistics_detailed = [] 

        # Initialize best fitness/cost based on the algorithm used. (NOTE: Assumes all blocks use the same algorithm type.)
        # This is used to determine when to update parameters for all blocks based on the performance of each block.
        if list(self.calibration_blocks.values())[0].model_type_name == 'genetic_alg':
            best_score_overall = float('-inf') # Higher is better (GA maximizes)
        elif list(self.calibration_blocks.values())[0].model_type_name == 'spsa':
            best_score_overall = float('inf') # Lower is better (SPSA minimizes)
        else:
            raise ValueError(f"Unsupported model type: {block_instance_obj.model_type_name}")

        for opt_iter_num in range(1, num_optimizer_iterations + 1):
            logging.info(f"--- Optimizer Master Iteration {opt_iter_num} ---")
            for block_id_str, block_instance_obj in self.calibration_blocks.items():

                # Skip blocks that have no parameters to optimize (e.g., heterogeneity if empty).
                if not self.block_parameter_metadata[block_id_str]['names']:
                     logging.info(f"Skipping block '{block_id_str}' in master iteration {opt_iter_num} as it has no parameters defined for optimization.")
                     continue

                logging.info(f"Running optimization for block: '{block_id_str}' (Master Iteration {opt_iter_num})")
                
                # param_block_all_score_hist is fitness or cost across all iterations of the block, not just this one.
                # this_iter_best_solution values is the best solution found in this iteration of the block.
                # this_iter_best_score is the fitness of that solution.
                # block_param_names is the parameter names for this block.
                # Pass best current overall fitness to the block's optimization to help system decide when to save parameters for new best solution.
                param_block_all_score_hist, this_iter_best_solution_values, this_iter_best_score, block_param_names = block_instance_obj.execute_block_optimization(best_score_overall)
                
                # TODO: Move this since all historical info is preserved so 
                # don't need to append each iteration. 
                all_run_statistics_detailed.append({ # Store detailed results.
                    "block_id": block_id_str,
                    "optimizer_iteration": opt_iter_num,
                    "block_run_fitness_cost_history": param_block_all_score_hist,
                    "block_best_solution_values": this_iter_best_solution_values,
                    "block_optimized_parameter_names": block_param_names
                })

                # Dictionary mapping criterion to a comparison function
                skip_dict = {
                    'genetic_alg': lambda x, y: x < y, # GA maximizes, so HIGHER is better; SKIP params update when this iter's best is LOWER than overall best
                    'spsa': lambda x, y: x > y, # SPSA minimizes, so LOWER is better, SKIP params update when this iter's best is HIGHER than overall best
                }
                
                # Only update the intermediate file(s) w/ new parameters for the next blocks if this block found a better solution.
                if skip_dict[block_instance_obj.model_type_name](this_iter_best_score, best_score_overall):
                    logging.info(f"Skipping default param update for block '{block_id_str}' as its best fitness {this_iter_best_score:.4f} is worse than the overall best {best_score_overall:.4f}.")
                else:
                    best_score_overall = this_iter_best_score

                    # Update `current_best_solution_struct` and intermediate_yaml_filename with the best solution from this block.
                    logging.info(f"Updating solutions and intermediate file for block '{block_id_str}' with: {this_iter_best_solution_values}")
                    if block_id_str == "od": # Update OD parameters.
                        for i_idx, param_name_str_key in enumerate(block_param_names): # param_name_str_key is like "('src','dst')"
                            try: # Convert string key back to tuple for `current_best_solution_struct`.
                                actual_tuple_key = eval(param_name_str_key)
                                self.current_best_solution_struct["od"][actual_tuple_key] = str(this_iter_best_solution_values[i_idx])
                            except SyntaxError: # Fallback if eval fails.
                                self.current_best_solution_struct["od"][param_name_str_key] = str(this_iter_best_solution_values[i_idx])
                            set_intermediate_file(
                                param_name_str_key,
                                str(this_iter_best_solution_values[i_idx]),
                                self.global_config_data["sim_dir_name"] + self.global_config_data["calibration_run"] + "/" + self.global_config_data["intermediate_yaml_filename"]
                            ) # Update intermediate_yaml_filename.

                    elif block_id_str == "cfm" or block_id_str == "lc": # Update CFM/LC parameters (grouped under "cfm").
                        for i_idx, param_name_item in enumerate(block_param_names):
                            self.current_best_solution_struct["cfm"][param_name_item] = str(this_iter_best_solution_values[i_idx])
                            set_intermediate_file(
                                param_name_item,
                                str(this_iter_best_solution_values[i_idx]),
                                self.global_config_data["sim_dir_name"] + self.global_config_data["calibration_run"] + "/" + self.global_config_data["intermediate_yaml_filename"]
                            )
                    else: # Update parameters for other block types.
                        if block_id_str not in self.current_best_solution_struct: self.current_best_solution_struct[block_id_str] = {}
                        for i_idx, param_name_item in enumerate(block_param_names):
                            self.current_best_solution_struct[block_id_str][param_name_item] = str(this_iter_best_solution_values[i_idx])
                            set_intermediate_file(
                                param_name_item,
                                str(this_iter_best_solution_values[i_idx]),
                                self.global_config_data["sim_dir_name"] + self.global_config_data["calibration_run"] + "/" + self.global_config_data["intermediate_yaml_filename"]
                            )

                    logging.debug(f"Current solution structure after block '{block_id_str}': {self.current_best_solution_struct}")
                    logging.info(f"Intermediate file updated with best solution from block '{block_id_str}'.")

                    # Copy the updated intermediate.yaml to each worker's directory to copy the best params from this block
                    # as defaults used in all the sims for the next block (so the next block can optimize other params conditional
                    # on the best params from this block).
                    # TODO: This will need to be updated w/ sol_per_pop as above, when that change is made
                    for worker_idx in range(self.global_config_data["ga_workers"]):
                        source_file = self.global_config_data["sim_dir_name"] + self.global_config_data["calibration_run"] + "/" + self.global_config_data["intermediate_yaml_filename"]
                        target_dir = self.global_config_data["sim_dir_name"] + self.global_config_data["calibration_run"] + "/" + str(worker_idx)
                        target_file = os.path.join(target_dir, self.global_config_data["intermediate_yaml_filename"])
                        shutil.copy(source_file, target_file)

        self._generate_and_save_optimization_summary_report(all_run_statistics_detailed)
        logging.info("Full optimization process completed.")

    def _generate_and_save_optimization_summary_report(self, detailed_statistics_list):
        # Saves a summary of the optimization statistics to a CSV file.
        # Transforms detailed stats into a simpler format if needed, or saves richer data.
        
        summary_output_records = [] # For a simplified CSV like the original.
        for stat_record in detailed_statistics_list:
            # `block_run_fitness_cost_history` is a list; take the last value as the final fitness/cost of that block's run.
            final_fitness_for_block_run = stat_record["block_run_fitness_cost_history"][-1] if stat_record["block_run_fitness_cost_history"] else None
            if final_fitness_for_block_run is not None:
                summary_output_records.append({
                    "block_id_label": stat_record["block_id"],
                    "optimizer_master_iteration": stat_record["optimizer_iteration"],
                    "final_fitness_cost_of_block_run": final_fitness_for_block_run
                })
        
        summary_df = pd.DataFrame(summary_output_records)
        output_report_csv_path = self.global_config_data["sim_dir_name"] + self.global_config_data["calibration_run"] + self.global_config_data["optimizer_summary_report_csv"]
        try:
            # Ensure the directory exists if it's nested (e.g., "data/").
            import os
            os.makedirs(os.path.dirname(output_report_csv_path), exist_ok=True)
            summary_df.to_csv(output_report_csv_path, index=False)
            logging.info(f"Optimization summary report successfully written to: {output_report_csv_path}")
        except IOError as e_io_report:
            logging.error(f"Error writing optimization summary CSV report to {output_report_csv_path}: {e_io_report}")

    def perform_single_evaluation_run(self, intermediate_yaml_for_eval, solution_idx=0):
        # Runs a single SUMO simulation using parameters from a specified YAML file and logs the error metrics.
        logging.info(f"Starting single evaluation run using parameters from: {intermediate_yaml_for_eval}")

        # Step 0: Preprocess filename if it's in the checkpoints dir
        if "checkpoints/" in intermediate_yaml_for_eval:
            intermediate_yaml_for_eval_fname = intermediate_yaml_for_eval.split("checkpoints/")[-1]

        # Step 1: Copy the specified intermediate YAML to a subdirectory for this solution index (from which sim will run)
        target_dir_path = self.global_config_data["sim_dir_name"] + self.global_config_data["calibration_run"] + "/" + str(solution_idx) + "/"
        intermediate_filename_for_run = target_dir_path + intermediate_yaml_for_eval_fname
        # Step 2: Create the directory if it doesn't exist
        os.makedirs(target_dir_path, exist_ok=True)
        # Step 3: Copy source file to target location
        shutil.copy(intermediate_yaml_for_eval, intermediate_filename_for_run)
        
        # Write SUMO config files based on the desired parameters in the specified solution_idx directory.
        write_sumo_config_files(self.global_config_data, self.current_best_solution_struct, solution_idx=solution_idx, intermediate_filename=intermediate_filename_for_run)
        # `run_sim_and_calculate_error_metric` runs the simulation. It uses `set_simulation_time_in_cfg`
        # which reads 'warmup_time', 'step_length' from the `intermediate_filename_for_run`.
        # It also reads 'num_chunks' from the same file.
        metric = run_sim_and_calculate_error_metric(
            self.simulation_start_seconds,
            self.simulation_end_seconds,
            self.global_config_data,
            solution_idx=solution_idx, # Pass the solution index (directory)
            intermediate_filename=intermediate_filename_for_run.split('/')[-1], # Pass the intermediate filename (without directory)
            is_eval_run=True # Indicate this is an evaluation run, so unnoised data will be used for assessment (whether it was used for calibration or not)
        )

        logging.info(f"Single evaluation run completed.")
        print(f"Single Evaluation Result (WITHOUT metric adjustment as may be done in calibration to suit optimization maximizing/minimizing) - {self.global_config_data['fitness_criterion']}={metric:.4f}")
        return metric

    def execute_parameter_bounds_test(self):
        # Tests the simulation with parameters set to their lower and upper bounds.
        logging.info("Starting parameter bounds testing procedure.")
        # Scenarios: one for all lower bounds, one for all upper bounds.
        bound_test_scenarios = {"lower_bounds_test": {}, "upper_bounds_test": {}}

        for block_id_key, block_meta_info in self.block_parameter_metadata.items():
            for i_idx, param_name_str in enumerate(block_meta_info['names']):
                param_model_config = block_meta_info['config_for_model'][i_idx] # e.g., {'low': X, 'high': Y} or list [v1, v2].
                
                # Determine low and high bound values for this parameter.
                low_bound_val, high_bound_val = None, None
                if isinstance(param_model_config, dict) and 'low' in param_model_config and 'high' in param_model_config:
                    low_bound_val, high_bound_val = param_model_config['low'], param_model_config['high']
                elif isinstance(param_model_config, list) and len(param_model_config) > 0:
                    # For lists (e.g., discrete choices or [min,max]), take first and last as bounds.
                    low_bound_val, high_bound_val = param_model_config[0], param_model_config[-1]
                
                if low_bound_val is not None:
                    bound_test_scenarios["lower_bounds_test"][str(param_name_str)] = low_bound_val
                if high_bound_val is not None:
                    bound_test_scenarios["upper_bounds_test"][str(param_name_str)] = high_bound_val
        
        # Parameters like 'variance', 'num_chunks', 'step_length' are usually fixed.
        # The `all_initial_parameter_values` dictionary (used as base for scenario YAMLs)
        # should already contain their configured values. This ensures they are present in the test YAMLs.

        for scenario_id_str, scenario_param_values_map in bound_test_scenarios.items():
            scenario_specific_yaml_file = f"{scenario_id_str}.yaml"

            scenario_path = f"0/{scenario_id_str}.yaml"
            dir_path = os.path.dirname(scenario_path)
            print(dir_path)

            # Check if the directory exists, if not, create it
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                
            try:
                # Create a full parameter set for this scenario, starting with all initial default values,
                # then updating with the specific bound values for this scenario.
                # This ensures non-optimized, fixed parameters are also included.
                full_scenario_params = self.all_initial_parameter_values.copy()
                full_scenario_params.update(scenario_param_values_map) # Override with bound values.
                
                with open(scenario_path, 'w') as file: # Write scenario YAML.
                    yaml.dump(full_scenario_params, file, default_flow_style=False, sort_keys=False)
                logging.info(f"Successfully wrote parameter bounds for '{scenario_id_str}' to {scenario_specific_yaml_file}")
                
                print(f"\n--- Running {scenario_id_str} ---")
                # Perform evaluation using this scenario's parameter file.
                # TODO: Change this; it is old. 
                geh_val, counts_val = self.perform_single_evaluation_run(intermediate_yaml_for_eval=scenario_specific_yaml_file)
                
                if np.isnan(geh_val) or np.isnan(counts_val): # Check for NaN results.
                    error_msg = f"FATAL: NaN detected in '{scenario_id_str}' run (GEH: {geh_val}, Counts: {counts_val}). Check parameter bounds definitions."
                    logging.error(error_msg)
                    raise ValueError(error_msg)
                logging.info(f"Result of '{scenario_id_str}' trial run - GEH: {geh_val}, Counts: {counts_val}")

            except IOError as e_io_bounds:
                logging.error(f"IOError during bounds testing for '{scenario_id_str}': {e_io_bounds}")
                raise
            except ValueError as e_val_bounds: # Catch the NaN error from above.
                print(f"ValueError during {scenario_id_str} testing: {e_val_bounds}")
                raise # Re-raise critical errors.