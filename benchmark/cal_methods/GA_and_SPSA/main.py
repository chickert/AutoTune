import os
import logging # For basic logging before optimizer's logger is set up.
from optimizer_core import Optimizer # Main optimizer class.
import sys # For exiting on critical errors.

def main_workflow():
    # Main entry point to initialize and run the optimization process or specific tasks.
    
    # Initialize the Optimizer with the path to the main configuration YAML file.
    # The Optimizer's __init__ method handles loading the config, setting up its own detailed logging,
    # and initializing all calibration blocks and parameters.
    try:
        # TODO: update to get path from CLI or config file
        optimizer_instance = Optimizer(main_config_yaml_path="./run_params/input_param_mediumnet.yaml") # Or get path from command-line arguments.
    except Exception as e_init:
        # If Optimizer initialization fails (e.g., config file not found, critical YAML parsing error),
        # log the error (if possible) and exit, as the system cannot proceed.
        logging.critical(f"CRITICAL ERROR: Failed to initialize the Optimizer. Details: {e_init}", exc_info=True)
        print(f"CRITICAL ERROR: Optimizer initialization failed. Please check logs. Error: {e_init}")
        sys.exit(1) # Exit the script with an error code.

    # --- Workflow Options ---
    # Uncomment or select the operations you wish to perform.

    # # Option 1: Test Parameter Bounds (Highly Recommended Before Full Optimization)
    # # This runs simulations with parameters at their defined lower and upper limits
    # # to catch potential issues (e.g., simulation crashes, NaN results) early.
    # try:
    #     print("\n--- Stage 1: Testing Parameter Bounds ---")
    #     optimizer_instance.execute_parameter_bounds_test()
    #     print("--- Parameter Bounds Test Completed Successfully ---")
    # except Exception as e_bounds_test:
    #     logging.error(f"ERROR during parameter bounds testing: {e_bounds_test}", exc_info=True)
    #     print(f"ERROR: Parameter bounds testing failed critically: {e_bounds_test}. Aborting further operations.")
    #     sys.exit(1) # Stop if bounds test indicates critical issues.

    # # Option 2: Run a Single Evaluation with a Specific Parameter Set
    # # Useful for testing a known configuration (e.g., from a previous run or manual setup).
    # # Ensure the specified YAML file (e.g., 'chengyuan_params.yaml') exists and is correctly formatted.
    # custom_eval_yaml = 'chengyuan.yaml' # Example filename.
    # try:
    #     print(f"\n--- Stage 2: Running Single Evaluation (using '{custom_eval_yaml}') ---")
    #     # The `perform_single_evaluation_run` method uses parameters from the specified YAML
    #     # to configure and run the SUMO simulation, then reports error metrics.
    #     optimizer_instance.perform_single_evaluation_run(intermediate_yaml_for_eval=custom_eval_yaml)
    #     print(f"--- Single Evaluation with '{custom_eval_yaml}' Completed ---")
    # except FileNotFoundError:
    #     logging.warning(f"Evaluation YAML file '{custom_eval_yaml}' not found. Skipping this single evaluation step.")
    #     print(f"Warning: Evaluation YAML file '{custom_eval_yaml}' not found. Skipping.")
    # except Exception as e_single_eval:
    #     logging.error(f"ERROR during single evaluation with '{custom_eval_yaml}': {e_single_eval}", exc_info=True)
    #     print(f"ERROR: Single evaluation with '{custom_eval_yaml}' failed: {e_single_eval}")

    # Option 3: Run the Full Optimization Cycle
    # This iterates through all defined calibration blocks, applying their respective optimization algorithms.
    try:
        print("\n--- Stage 3: Starting Full Optimization Cycle ---")
        # `num_optimizer_iterations` is the number of times the optimizer cycles through *all* blocks.
        optimizer_instance.run_full_optimization_process(num_optimizer_iterations=5) # Example: 2 master iterations.
        print("--- Full Optimization Cycle Completed ---")
    except Exception as e_full_opt:
        logging.error(f"ERROR during the full optimization cycle: {e_full_opt}", exc_info=True)
        print(f"An error occurred during the full optimization cycle: {e_full_opt}")

    # # Option 4: Run a Final Evaluation using the Parameters in the Default 'intermediate.yaml'
    # # This file should reflect the best parameters found after the optimization cycle (if run).
    # try:
    #     print("\n--- Stage 4: Running Final Evaluation (with optimized parameters) ---")
    #     # optimizer_instance.perform_single_evaluation_run(intermediate_yaml_for_eval=optimizer_instance.global_config_data["checkpoints_dir"] + optimizer_instance.global_config_data["checkpoint_to_eval"]) 
        
    #     # TODO: Consider if this is the best place for these, or should go in sumo_utils.py
    #     if optimizer_instance.global_config_data["error_func_type"] == "smallnet":
    #         # Add the parent directory to the system path for viz utils imports
    #         current_script_dir = os.path.dirname(__file__)
    #         parent_dir = os.path.abspath(os.path.join(current_script_dir, '../../viz_error'))
    #         sys.path.insert(0, parent_dir)
    #         from smallmednet_viz_utils import gen_smallnet_plots

    #         # Build paths for loading data and saving
    #         target_dir_path = optimizer_instance.global_config_data["sim_dir_name"] + optimizer_instance.global_config_data["calibration_run"] + "/" + str(0)
    #         fcd_file_path = target_dir_path + optimizer_instance.global_config_data["fcd_output_file"]
    #         plot_subpath = optimizer_instance.global_config_data['checkpoint_to_eval'].split('/')[2].split('.')[0]

    #         # Generate visualizations
    #         gen_smallnet_plots(fcd_name=fcd_file_path.split('.xml')[0], plot_save_path=f"analysis/{optimizer_instance.global_config_data['error_func_type']}/{plot_subpath}.png")

    #     if optimizer_instance.global_config_data["error_func_type"] == "mediumnet":
    #         # Add the parent directory to the system path for viz utils imports
    #         current_script_dir = os.path.dirname(__file__)
    #         parent_dir = os.path.abspath(os.path.join(current_script_dir, '../../viz_error'))
    #         sys.path.insert(0, parent_dir)
    #         from smallmednet_viz_utils import gen_mednet_plots

    #         # Build paths for loading data and saving
    #         target_dir_path = optimizer_instance.global_config_data["sim_dir_name"] + optimizer_instance.global_config_data["calibration_run"] + "/" + str(0)
    #         fcd_file_path = target_dir_path + optimizer_instance.global_config_data["fcd_output_file"]
    #         plot_subpath = optimizer_instance.global_config_data['checkpoint_to_eval'].split('/')[2].split('.')[0]

    #         # Generate visualizations
    #         gen_mednet_plots(fcd_name=fcd_file_path.split('.xml')[0],
    #                          plot_save_path=f"analysis/{optimizer_instance.global_config_data['error_func_type']}/{plot_subpath}.png",
    #                          start_time=0, end_time=10800, hours=3)
 
        
    #     print("--- Final Evaluation Completed ---")
    # except Exception as e_final_eval:
    #     logging.error(f"ERROR during final evaluation: {e_final_eval}", exc_info=True)
    #     print(f"ERROR during final evaluation: {e_final_eval}")

    logging.info("Optimizer script execution finished.")
    print("\nOptimizer script has finished all planned operations. Please check logs for details.")

if __name__ == "__main__":
    # Setup basic console logging for messages that occur *before* the Optimizer's
    # file-based logging is configured. This helps catch very early issues,
    # such as if the main 'input_param.yaml' itself is missing or unreadable.
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: (pre-init) %(message)s')
    main_workflow()