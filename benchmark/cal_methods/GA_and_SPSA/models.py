from abc import ABC, abstractmethod
import ast
import shutil
import logging
import os
import pygad # For Genetic Algorithm.
import random
import subprocess
import numpy as np
import xml.etree.ElementTree as ET
# Assuming spsa_library is available in the Python environment.
from spsa_library.spsa import SPSAAlgorithm
from spsa_library.problem import ContinuousProblem

# Custom module imports.
from param_utils import set_intermediate_file
from sumo_utils import write_sumo_config_files, run_sim_and_calculate_error_metric

class Model(ABC):
    # Abstract base class for all optimization models (e.g., GA, SPSA).
    def __init__(self, parameter_names_to_optimize, parameter_config_for_model, # e.g., gene_space for GA, bounds for SPSA.
                 simulation_data_config, sim_start_seconds, sim_end_seconds,
                 block_identifier, full_current_solutions_dict, # For context of other blocks.
                 algorithm_specific_params):
        self.param_names = parameter_names_to_optimize    # List of parameter names this model instance will optimize.
        self.param_config = parameter_config_for_model    # Structure defining search space (e.g., GA's gene_space).
        self.sim_data = simulation_data_config            # General simulation config from main YAML.
                                  # Threading lock, if needed for shared resources.
        self.sim_start = sim_start_seconds                # Simulation start time in seconds.
        self.sim_end = sim_end_seconds                    # Simulation end time in seconds.
        self.block_name = block_identifier                # Identifier for the current calibration block (e.g., "od").
        self.all_block_solutions = full_current_solutions_dict # Snapshots of solutions from other blocks.
        self.alg_params = algorithm_specific_params       # Dict of params specific to the algorithm (e.g., num_generations).
        self.run_fitness_cost_history = []                     # Stores fitness/cost values over the optimization run.

    @abstractmethod
    def run_optimization(self):
        # Abstract method to execute the optimization algorithm.
        pass

    @abstractmethod
    def _evaluate_solution_fitness(self, solution_candidate_values):
        # Abstract method to calculate fitness for a given candidate solution.
        # This typically involves running a SUMO simulation.
        pass

    def generate_sumo_od_matrix(self):
        """
        Uses SUMO's flowrouter and duarouter tools to generate an initial OD matrix from the provided detector data.
        """
        # Step 1: Run flowrouter to get SUMO's estimate for routes & flows based on detector data
        flowrouter_cmd = [
            "python3",
            f"{os.environ['SUMO_HOME']}/tools/detector/flowrouter.py",
            "-n", self.sim_data["sim_dir_name"] + self.sim_data['network_file'],
            "-d", self.sim_data["sim_dir_name"] + self.sim_data["detector_file"],
            "-f", self.sim_data["data_dir_name"] + self.sim_data["data_path"],
            "-o", self.sim_data["sim_dir_name"] + self.sim_data["calibration_run"] + self.sim_data["output_routes"],
            "-e", self.sim_data["sim_dir_name"] + self.sim_data["calibration_run"] + self.sim_data["flowrouter_output_flows"],
            "-i", str(self.sim_data["aggregation_interval"]), 
            "--respect-zero", 
            "--revalidate-detectors", 
            "--quiet",
            ]
        subprocess.run(flowrouter_cmd, capture_output=True, text=True, check=True) 

        # Step 2: Run duarouter to format routes & flows in format required by route2OD.py
        duarouter_cmd = [
            "duarouter",
            "-n", self.sim_data["sim_dir_name"] + self.sim_data['network_file'],
            "--route-files", self.sim_data["sim_dir_name"] + self.sim_data["calibration_run"] + self.sim_data["output_routes"] + "," + self.sim_data["sim_dir_name"] + self.sim_data["calibration_run"] + self.sim_data["flowrouter_output_flows"],
            "-o", self.sim_data["sim_dir_name"] + self.sim_data["calibration_run"] + self.sim_data['duarouter_output_routes'],
            # "--ignore-errors",
            "--no-step-log",
        ]
        subprocess.run(duarouter_cmd, capture_output=True, text=True, check=True) # 

        # Step 3: Convert routes to OD matrix using route2OD.py
        route2od_cmd = [
            "python3",
            f"{os.environ['SUMO_HOME']}/tools/route/route2OD.py",
            "-r", self.sim_data["sim_dir_name"] + self.sim_data["calibration_run"] + self.sim_data['duarouter_output_routes'],
            "-a", self.sim_data["sim_dir_name"] + self.sim_data["taz_file"],
            "-o", self.sim_data["sim_dir_name"] + self.sim_data["calibration_run"] + self.sim_data['od_initialization_file'],
        ]
        subprocess.run(route2od_cmd, capture_output=True, text=True, check=False) # Set check=False to allow for non-zero exit codes

    def create_od_counts_dict(self, xml_file_path):
        """
        Reads an XML file, extracts 'count' values for matching 'from' and 'to' TAZs
        based on self.param_names, and returns a new dictionary.
        For OD pairs not found in the XML, it defaults to 0.

        Args:
            xml_file_path (str): The path to the XML file.

        Returns:
            dict: A dictionary where keys are items from self.param_names
                  and values are the corresponding 'count' from the XML,
                  or None if not found.
        """
        od_counts = {}
        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()

            # Create a lookup for quick access to XML data
            xml_data_lookup = {}
            for interval in root.findall('interval'):
                for taz_relation in interval.findall('tazRelation'):
                    from_taz = taz_relation.get('from')
                    to_taz = taz_relation.get('to')
                    count = taz_relation.get('count')
                    xml_data_lookup[(from_taz, to_taz)] = count

            for param_name_str in self.param_names:
                # Convert the string representation of tuple to an actual tuple
                from_taz, to_taz = ast.literal_eval(param_name_str)
                key_tuple = (from_taz, to_taz)

                # Look up the count in our prepared dictionary
                count_value = xml_data_lookup.get(key_tuple)

                # Store the original string key and the found count if it exists, otherwise default to 0.
                od_counts[param_name_str] = int(count_value) if count_value is not None else 0

        except FileNotFoundError:
            print(f"Error: The file '{xml_file_path}' was not found.")
            return None
        except ET.ParseError:
            print(f"Error: Could not parse the XML file '{xml_file_path}'. Check for well-formedness.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

        return od_counts

    @staticmethod
    def fixed_sum_flow_redistribution(od_array, redistribution_percentage, max_num_shifts, seed=None):
        """
        Applies fixed-sum flow redistribution to an Origin-Destination (OD) array.

        This function redistributes a specified percentage of the total traffic flow
        among randomly selected OD pairs. It ensures that the overall sum of all OD
        entries remains approximately constant, which is useful for generating
        variations while preserving total demand.

        Args:
            od_array (np.ndarray): The input NumPy array of OD entries.
                                This can be 1D (flattened) or 2D.
                                Assumed to contain non-negative integer values.
            redistribution_percentage (float): The percentage of the total flow
                                            to be redistributed (e.g., 5 for 5%).
                                            This amount will be distributed across
                                            `num_shifts` operations.
            max_num_shifts (int): The maximum number of individual flow transfers (shifts)
                            that could be performed. A higher number leads to more granular
                            and distributed changes.
            seed (int): Random seed for reproducibility of the redistribution process.

        Returns:
            np.ndarray: A new NumPy array with the redistributed OD entries.
                        Values are rounded to the nearest integer and remain non-negative.
        """

        # Set the random seed for reproducibility
        if seed:
            random.seed(seed)

        # Create a copy to ensure the original array is not modified
        modified_od_array = od_array.copy()
        
        # Calculate the total sum of all OD entries in the current array
        total_current_flow = np.sum(modified_od_array)

        if total_current_flow == 0:
            print("Warning: Total flow in the array is zero. No redistribution can occur.")
            return modified_od_array

        # Calculate the total amount of flow to be redistributed across all shifts
        # This is the "budget" for redistribution
        total_redistribution_amount = total_current_flow * (redistribution_percentage / 100.0)
        
        # Track the remaining amount to be shifted to ensure we don't over-shift
        remaining_budget = total_redistribution_amount

        # Get all possible indices (flattened) for random selection
        # This works for both 1D and 2D arrays
        if od_array.ndim == 1:
            all_indices = list(range(od_array.size))
        else: # For 2D arrays, generate (row, col) tuples
            rows, cols = od_array.shape
            all_indices = [(r, c) for r in range(rows) for c in range(cols)]

        # Perform multiple individual shifts
        for _ in range(max_num_shifts):
            if remaining_budget <= 0:
                break # No more budget to redistribute

            # 1. Select a source OD pair to decrease flow from
            # It must have a positive flow to decrease
            non_zero_indices = np.where(modified_od_array > 0)
            
            # Convert non_zero_indices to a list of tuples for 2D, or list of ints for 1D
            if od_array.ndim == 1:
                possible_sources = non_zero_indices[0].tolist()
            else:
                possible_sources = list(zip(*non_zero_indices))

            if not possible_sources:
                # No more non-zero flows to redistribute from
                break

            source_idx = random.choice(possible_sources)
            current_source_flow = modified_od_array[source_idx]

            # 2. Select a destination OD pair to increase flow to
            # It should ideally be different from the source
            possible_destinations = [idx for idx in all_indices if idx != source_idx]
            if not possible_destinations:
                # No other OD pairs to shift to
                break
            dest_idx = random.choice(possible_destinations)

            # 3. Determine the actual amount to shift in this single operation
            # This shift value is capped by:
            #   a) The current flow at the source (cannot go below 0)
            #   b) The remaining redistribution budget
            #   c) Must be at least 1 to make a meaningful change
            
            # Calculate maximum possible shift for this operation
            max_shift_for_op = min(current_source_flow, remaining_budget)
            
            if max_shift_for_op < 1: # If max possible shift is less than 1, skip this iteration
                continue

            # Choose a random integer value to shift, between 1 and max_shift_for_op
            shift_value = random.randint(1, int(max_shift_for_op))

            # Apply the shift
            modified_od_array[source_idx] -= shift_value
            modified_od_array[dest_idx] += shift_value

            # Update the remaining budget
            remaining_budget -= shift_value
            
            # Ensure non-negativity and integer values after each shift
            # (though `min` and `randint` should largely handle this for individual shifts)
            modified_od_array[source_idx] = int(max(0, modified_od_array[source_idx]))
            modified_od_array[dest_idx] = int(modified_od_array[dest_idx])

        # Final check and rounding to ensure all values are integers
        modified_od_array = np.round(modified_od_array).astype(int)

        return modified_od_array

class GeneticAlgorithmModel(Model):
    # Implementation of the Genetic Algorithm optimization model.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_ga_instance()

    def _setup_ga_instance(self):
        # Initializes the PyGAD Genetic Algorithm instance with configured parameters.
        ga_instance_params = {
            "num_generations": self.alg_params.get('num_generations', 3),   # Total # of sims = (num_generations * sol_per_pop) + sol_per_pop (for initial gen)
            "num_parents_mating": self.alg_params.get('num_parents_mating', 2),
            "sol_per_pop": self.alg_params.get('sol_per_pop', 4),   # Total # of sims = (num_generations * sol_per_pop) + sol_per_pop (for initial gen)
            "num_genes": len(self.param_names),
            "gene_space": self.param_config, # Expects format like [{'low': X, 'high': Y}, ...] or lists for discrete.
            "parent_selection_type": self.alg_params.get('ga_parent_selection', "sss"),
            "keep_parents": self.alg_params.get('ga_keep_parents', 1),
            "crossover_type": self.alg_params.get('ga_crossover_type', "single_point"),
            "mutation_type": self.alg_params.get('ga_mutation_type', "random"),
            "mutation_percent_genes": self.alg_params.get('ga_mutation_percent', 10),
            "fitness_func": self._ga_fitness_function_adapter, # Adapter for PyGAD.
            "on_generation": self._ga_generation_callback,    # Called after each generation.
            "parallel_processing": self.alg_params.get('ga_parallel_config', ["process", self.sim_data["ga_workers"]]), # e.g., ["thread", 4] or 0 for auto process.
            "random_seed": self.alg_params.get('random_seed', 0), # Seed for reproducibility.
            "save_best_solutions": True, # Save best solutions each generation
        }
        # Initialize OD block w/ specific parameters
        if self.block_name == 'od':
            
            # Generate the initial OD matrix using SUMO tools.
            self.generate_sumo_od_matrix()

            # Load the OD matrix as one chromosome in the initial population for the GA.
            ga_instance_params["initial_population"] = [] 
            od_entries = self.create_od_counts_dict(self.sim_data["sim_dir_name"] + self.sim_data['calibration_run'] + self.sim_data['od_initialization_file'])
            ga_instance_params["initial_population"].append(np.array(list(od_entries.values()))) 

            # Generate slight variations on the OD entries to create a diverse initial population.
            for i in range(ga_instance_params["sol_per_pop"] - 1): 
                # print(f"Generating initial population variant {i+1} for OD block.")
                seed = i + self.alg_params.get('random_seed', 0)
                variant = self.fixed_sum_flow_redistribution(
                    np.array(list(od_entries.values())), 
                    redistribution_percentage=self.alg_params.get('od_initial_redistribution_percentage', 10), 
                    max_num_shifts=self.alg_params.get('od_initial_max_num_shifts', 10), 
                    seed=seed,
                    )
                ga_instance_params["initial_population"].append(variant)
            
        self.ga_model = pygad.GA(**ga_instance_params)

    def _ga_fitness_function_adapter(self, ga_inst, solution_values, solution_idx):
        # Adapts the call to the main fitness evaluation method for PyGAD.
        return self._evaluate_solution_fitness(solution_values, solution_idx)

    def _evaluate_solution_fitness(self, current_solution_values, solution_idx):
        # Calculates fitness for a GA solution by running SUMO and evaluating specified error metrics.
        param_log_str = ", ".join([f"{name}: {val:.3f}" for name, val in zip(self.param_names, current_solution_values)])
        # logging.info("GA (%s): Evaluating solution #%d: %s\n", self.block_name, solution_idx, param_log_str)

        # Update the intermediate YAML file with the current candidate solution values for THIS block's parameters.
        target_dir = self.sim_data["sim_dir_name"] + self.sim_data["calibration_run"] + "/" + str(solution_idx)
        target_file = os.path.join(target_dir, self.sim_data["intermediate_yaml_filename"])

        # Inject the current GA candidate values into the solution_idx's intermediate file, from which they will be used to build the sim
        for i, p_name in enumerate(self.param_names):

            # Round the warmup_time and step_length parameters for SUMO compatibility.
            if p_name == 'warmup_time':
                # Ensure warmup_time is an integer number of seconds.
                current_solution_values[i] = round(current_solution_values[i])
            if p_name == 'step_length':
                # Ensure step_length is a float, but rounded to nearest 0.1 seconds.
                current_solution_values[i] = round(current_solution_values[i] * 10) / 10.0

            set_intermediate_file(str(p_name), str(current_solution_values[i]), filename=target_file)
        
        # Write all SUMO configuration files. This uses the updated intermediate YAML for values,
        # and self.all_block_solutions for the structural context of OD keys, CFM/LC grouping.
        write_sumo_config_files(self.sim_data, self.all_block_solutions, solution_idx=solution_idx, intermediate_filename=target_file)

        # Run SUMO simulation(s) and get error metrics using the chosen error function.
        metric = run_sim_and_calculate_error_metric(
            self.sim_start, self.sim_end, self.sim_data, solution_idx=solution_idx, intermediate_filename=self.sim_data["intermediate_yaml_filename"],
        ) # Assumes error_func returns (e.g., geh, counts).
        
        # Fitness is typically inverse of error (higher is better). Avoid division by zero.
        if self.sim_data["fitness_criterion"] == "speed_iou":
            calculated_fitness = metric
        elif self.sim_data["fitness_criterion"] == "count_rmse" or self.sim_data["fitness_criterion"] == "headway_wass_dist" or self.sim_data["fitness_criterion"] == "mape" or self.sim_data["fitness_criterion"] == "speed_rmse":
             calculated_fitness = 1.0 / metric if metric != 0 else 0.0
        else:
            raise ValueError(f"Unknown fitness_criterion {self.sim_data['fitness_criterion']} in config.")

        # Be sure to save the intermediate file from the best solution (our GA maximizes)
        if calculated_fitness >= self.current_overall_best_fitness:
            checkpoint_fname = f"{self.sim_data['checkpoints_dir'] + self.sim_data['calibration_run']}/intermediate_best_checkpoint-GA-{solution_idx}.yaml"
            shutil.copyfile(target_file, checkpoint_fname)
            logging.info(f"New fitness {calculated_fitness:.4f} ({self.sim_data['fitness_criterion']}: {metric:.4f}) found at solution {solution_idx} matches/beats prior best {self.current_overall_best_fitness:.4f}, saving to {checkpoint_fname}")

        # logging.info("GA (%s) solution %d: Params: %s -> GEH: %.3f, Counts: %.1f, Fitness (%s): %.4f\n", self.block_name, solution_idx, param_log_str, metric1, metric2, fitness_criterion, calculated_fitness)
        return calculated_fitness

    def _ga_generation_callback(self, ga_instance):
        # Callback function executed by PyGAD after each generation.
        current_gen_best_fitness = ga_instance.best_solutions_fitness[-1]
        self.run_fitness_cost_history.append(current_gen_best_fitness) 
        # Log best fitness for this generation.
        logging.info(f"GA ({self.block_name}) Gen {ga_instance.generations_completed} - This generation's best fitness: {current_gen_best_fitness:.4f}")

    def run_optimization(self, current_overall_best_fitness):

        # Store the current best fitness for comparison across iterations (used to save best solution file for easy access at eval time, if desired)
        self.current_overall_best_fitness = current_overall_best_fitness

        # Executes the Genetic Algorithm optimization process.
        logging.info(f"Starting Genetic Algorithm for block: {self.block_name}")
        self.ga_model.run() # Run the GA.

        # Look back across all generations in this iteration (for this block) to find this iteration's best solution.
        lookback_idxs = self.ga_model.num_generations + 1 # Have the "+1" here for GA since it somehow logs the final gen twice in the best_solutions and best_solutions_fitness lists.
        # We get the best fitness across all generations (incl. generations from other iterations)
        this_iter_best_fitness = max(self.ga_model.best_solutions_fitness[-lookback_idxs:]) # Since our GA maximizes, best is max 
        this_iter_best_fitness_idx = self.ga_model.best_solutions_fitness[-lookback_idxs:].index(this_iter_best_fitness) # Get the index of the best fitness value.
        this_iter_best_solution_values = self.ga_model.best_solutions[-lookback_idxs:][this_iter_best_fitness_idx]  # Get the corresponding solution values for the best fitness.
        
        logging.info(f"GA for {self.block_name} finished. Best block solution this iteration: {this_iter_best_solution_values}, Fitness: {this_iter_best_fitness:.4f}")
        return self.run_fitness_cost_history, this_iter_best_solution_values, this_iter_best_fitness, self.param_names


class SPSAEvaluator:
    # Helper class for SPSA to submit and manage objective function evaluations.
    def __init__(self, spsa_model_ref):
        self.spsa_model_instance = spsa_model_ref # Reference to the SPSAModel instance.
        self.evaluation_log = []                  # Logs details of each evaluation.

    def submit_one(self, solution_candidate_array, metadata=None):
        # This method is called by the SPSA library to get the objective function value (cost).
        # SPSA aims to minimize this cost.
        cost_value = self.spsa_model_instance._evaluate_solution_fitness(solution_candidate_array)
        self.evaluation_log.append({'solution': solution_candidate_array, 'cost': cost_value, 'metadata': metadata}) # Saves across iterations
        return cost_value # Return the cost for SPSA.

    def get_logged_evaluations(self):
        # Retrieves the log of all evaluations performed.
        return self.evaluation_log


class SPSAModel(Model):
    # Implementation of the Simultaneous Perturbation Stochastic Approximation (SPSA) model.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.spsa_eval_helper = SPSAEvaluator(self) # Helper for SPSA library interaction.
        self._setup_spsa_instance()

    def _setup_spsa_instance(self):
        # Initializes the SPSAAlgorithm instance.
        spsa_initial_values = []
        spsa_param_bounds = [] # SPSA expects bounds as a list of (low, high) tuples.
        
        # Convert parameter configuration (from GA-like gene_space) to SPSA format.
        for p_conf in self.param_config:
            # TODO: Init w/ SUMO vals (?)
            if isinstance(p_conf, dict) and "low" in p_conf and "high" in p_conf: # For numeric ranges.
                if self.block_name == 'heterogeneity':
                    spsa_initial_values.append(p_conf["low"]) # Set initial value from low bound.
                elif self.block_name == 'sim':
                    if p_conf["high"] > 10.0: # Using 10.0 as a heuristic to identify warmup_time vs step_length, since step_length max is typically well below that
                        spsa_initial_values.append(p_conf["low"]) # Initialize warmup_time with low bound.
                    else:
                        assert p_conf['low'] <= 1.0 <= p_conf['high'], "step_length bounds must include 1.0"
                        spsa_initial_values.append(1.0) # Initialize step_length with 1.0s
                else:
                    spsa_initial_values.append((p_conf["low"] + p_conf["high"]) / 2.0) # Midpoint as initial.
                spsa_param_bounds.append((p_conf["low"], p_conf["high"]))
            elif isinstance(p_conf, list) and len(p_conf) == 2: 
                if self.block_name == 'cfm' or self.block_name == 'lc':
                    spsa_initial_values.append(p_conf[len(p_conf) // 2]) # Middle choice as initial.
                elif self.block_name == 'sim':
                    spsa_initial_values.append(p_conf[0])   # Set initial value from first element if list, or value itself.
                spsa_param_bounds.append((min(p_conf), max(p_conf))) # Min/max as bounds.
            elif isinstance(p_conf, list) and len(p_conf) != 2: # For discrete choices (SPSA is generally best for continuous).
                spsa_initial_values.append(p_conf[0]) # Set initial value from first element if list, or value itself.
                spsa_param_bounds.append((min(p_conf), max(p_conf))) # Min/max as bounds.
                logging.warning(f"SPSA ({self.block_name}): Using min/max of discrete list {p_conf} as bounds. SPSA is best for continuous parameters.")
            else: # Fallback for unexpected parameter config.
                raise ValueError(f"SPSA ({self.block_name}): Unexpected parameter config structure: {p_conf}. Expected dict with 'low'/'high' or list of values.")

        # Initialize OD block w/ SUMO-generated initial values.
        if self.block_name == 'od':
            # Generate the initial OD matrix using SUMO tools.
            self.generate_sumo_od_matrix()
            # Load the OD matrix as the initial values for the SPSA optimization.
            od_entries = self.create_od_counts_dict(self.sim_data["sim_dir_name"] + self.sim_data['calibration_run'] + self.sim_data['od_initialization_file'])
            spsa_initial_values = [float(od_entry) for od_entry in od_entries.values()]
        
        spsa_problem_def = ContinuousProblem(parameters=len(self.param_names),
                                             bounds=spsa_param_bounds,
                                             initial_values=spsa_initial_values)
        
        # NOTE: spsa_cost_scale_factor is another hyperparameter used below (defaults to 1.0).
        spsa_algo_config = { # SPSA algorithm hyperparameters.
            "perturbation_factor": self.alg_params.get('spsa_perturb_factor', 0.1),
            "gradient_factor": self.alg_params.get('spsa_grad_factor', 0.1),
            "perturbation_exponent": self.alg_params.get('spsa_perturb_exp', 0.101),
            "gradient_exponent": self.alg_params.get('spsa_grad_exp', 0.602),
            "gradient_offset": self.alg_params.get('spsa_grad_offset', 0),
            "compute_objective": True, # SPSA library will use SPSAEvaluator's 
            "seed": self.alg_params.get('random_seed', 0)
        }

        self.spsa_model = SPSAAlgorithm(problem=spsa_problem_def, **spsa_algo_config)
        self.spsa_num_generations = self.alg_params.get('num_generations', 100) # Number of SPSA generations.

    def _evaluate_solution_fitness(self, current_solution_values):
        # Calculates "cost" for SPSA by running SUMO. SPSA minimizes this cost.

        # Log the current solution values being evaluated.        
        param_log_str = ", ".join([f"{name}: {val:.3f}" for name, val in zip(self.param_names, current_solution_values)])
        logging.info(f"SPSA ({self.block_name}): Evaluating solution: {param_log_str}")
    
        # For the simulation block, we need to ensure the values are appropriately rounded 
        # NOTE: This is a workaround for the SPSA issue with discrete parameters.
        if self.block_name == 'sim':
            for i, p_name in enumerate(self.param_names):
                if p_name in ["num_chunks", "warmup_time"]:
                    # Ensure simulation parameters are integers.
                    current_solution_values[i] = round(current_solution_values[i])
                elif p_name in ["step_length"]:
                    # Ensure step_length is a float, but rounded to nearest 0.1 seconds.
                    current_solution_values[i] = round(current_solution_values[i] * 10) / 10.0
                else:
                    raise ValueError(f"SPSA ({self.block_name}): Unexpected parameter name for simulation block: {p_name}. Expected 'num_chunks', 'warmup_time', or 'step_length'.")
                
            param_log_str = ", ".join([f"{name}: {val:.3f}" for name, val in zip(self.param_names, current_solution_values)])
            logging.info(f"\t\t\t\t\tSolution rounded to: {param_log_str}")

        # Inject the current SPSA candidate values into the overall intermediate file, from which they will be used to build the sim
        intermediate_file_path = self.sim_data["sim_dir_name"] + self.sim_data["calibration_run"] + "/" + self.sim_data["intermediate_yaml_filename"]
        for i, p_name in enumerate(self.param_names): # Update intermediate file.
            set_intermediate_file(str(p_name), str(current_solution_values[i]), intermediate_file_path)

        write_sumo_config_files(self.sim_data, self.all_block_solutions, solution_idx=0, intermediate_filename=intermediate_file_path) # Setup SUMO files.

        metric = run_sim_and_calculate_error_metric( # Run simulation and get metrics.
            self.sim_start, self.sim_end, self.sim_data, solution_idx=0, intermediate_filename=self.sim_data["intermediate_yaml_filename"],
        )

        # Cost is the error metric itself (SPSA minimizes).
        if self.sim_data["fitness_criterion"] == "count_rmse" or self.sim_data["fitness_criterion"] == "headway_wass_dist" or self.sim_data["fitness_criterion"] == "mape" or self.sim_data["fitness_criterion"] == "speed_rmse":
            calculated_cost = metric # Lower RMSE or Wasserstein distance is better.
        elif self.sim_data["fitness_criterion"] == "speed_iou":
            calculated_cost = 1 - metric # Higher IOU is better, so cost is 1 - IOU.
        else:
            raise ValueError(f"Unknown fitness_criterion {self.sim_data['fitness_criterion']} in config.")
        
        # spsa_cost_scale_factor can be used to scale very large or small costs to a more suitable range
        calculated_cost *= self.alg_params.get("spsa_cost_scale_factor", 1.0) 

        # Be sure to save the intermediate file from the best solution (SPSA minimizes)
        if calculated_cost <= self.current_overall_best_cost:
            checkpoint_fname = f"{self.sim_data['checkpoints_dir'] + self.sim_data['calibration_run']}/intermediate_best_checkpoint-SPSA.yaml"
            shutil.copyfile(intermediate_file_path, checkpoint_fname)
            logging.info(f"New fitness {calculated_cost:.4f} matches/beats prior best {self.current_overall_best_cost:.4f}, saving to {checkpoint_fname}")

        # logging.info(f"SPSA ({self.block_name}): Params: {param_log_str} -> GEH: {metric1:.3f}, Counts: {metric2:.1f}, Cost ({cost_criterion}): {calculated_cost:.4f}")
        self.run_fitness_cost_history.append(calculated_cost) # Log the cost for this SPSA step.
        return calculated_cost

    def run_optimization(self, current_overall_best_cost):
        ### Executes the SPSA optimization process.

        # Store the current best fitness for comparison across iterations (used to save best solution file for easy access at eval time, if desired)
        self.current_overall_best_cost = current_overall_best_cost

        initial_spsa_state = self.spsa_model.get_state()['values']
        logging.info(f"Starting SPSA for block: {self.block_name}. Initial state: {initial_spsa_state}")

        # Total # of sims = num_generations * 3 (3 sims per gen)
        for i in range(self.spsa_num_generations):
            logging.info(f"SPSA ({self.block_name}) Generation {i+1}/{self.spsa_num_generations} starting. Starting soln values: {self.spsa_model.get_state()['values']}")
            self.spsa_model.advance(self.spsa_eval_helper) # Perform one SPSA step (runs 3 sims).
            lookback_idxs = 3 # Get the last 3 evaluations (since 3 sims per gen).
            this_gen_results = [(sim_info['cost'], sim_info['solution']) for sim_info in self.spsa_eval_helper.evaluation_log[-lookback_idxs:]] 
            this_gen_best_cost, this_gen_best_solution_values = min(this_gen_results, key=lambda item: item[0]) # Since our SPSA minimizes, best is min
            this_gen_best_fitness = 1 - this_gen_best_cost if self.alg_params.get("fitness_criterion") == "speed_iou" else this_gen_best_cost # Fitness is inverse of cost for speed_iou, else cost itself.
            logging.info(f"SPSA ({self.block_name}) Generation {i+1} done. This generation's best fitness: {this_gen_best_fitness}, Resulting soln values: {this_gen_best_solution_values}")

        # Look back across all generations in this iteration (for this block) to find this iteration's best solution and associated cost.
        # self.spsa_eval_helper.evaluation_log saves history across all generations/iterations in this block, so just grab the ones from this iteration.
        lookback_idxs = self.spsa_num_generations * 3 
        this_iter_results = [(sim_info['cost'], sim_info['solution']) for sim_info in self.spsa_eval_helper.evaluation_log[-lookback_idxs:]] 
        this_iter_best_cost, this_iter_best_solution_values = min(this_iter_results, key=lambda item: item[0]) # Since our SPSA minimizes, best is min
        this_iter_best_fitness = 1 - this_iter_best_cost if self.alg_params.get("fitness_criterion") == "speed_iou" else this_iter_best_cost # Fitness is inverse of cost for speed_iou, else cost itself.

        logging.info(f"SPSA for {self.block_name} finished. Best block solution this iteration: {this_iter_best_solution_values}, Fitness: {this_iter_best_fitness:.4f}")
        return self.run_fitness_cost_history, this_iter_best_solution_values, this_iter_best_fitness, self.param_names


class NeuralNetworkModel(Model): # Stub from original code, needs full implementation.
    # Placeholder for a Neural Network based optimization model.
    def __init__(self, **kwargs):
        # super().__init__(**kwargs) # NN model might have a different initialization pattern.
        self.param_names = kwargs.get("parameter_names_to_optimize", []) # Ensure compatibility.
        self.run_fitness_cost_history = []
        logging.warning(f"NeuralNetworkModel for block '{kwargs.get('block_identifier', 'N/A')}' is a STUB and not implemented.")

    def run_optimization(self):
        logging.info(f"NeuralNetworkModel run method called for block '{self.block_name}', but it's not implemented.")
        # Return dummy values to match the expected signature by the Optimizer class.
        dummy_solution = [0.0] * len(self.param_names) # Create a list of zeros.
        return self.run_fitness_cost_history, dummy_solution, self.param_names

    def _evaluate_solution_fitness(self, solution_candidate_values):
        # Fitness evaluation for an NN model would be very different (e.g., model prediction, error).
        logging.warning(f"NeuralNetworkModel _evaluate_solution_fitness for block '{self.block_name}' called, but not implemented.")
        return 0.0 # Return a dummy fitness/cost.