''' This model holds a Block class and functionality needed to optimize a block.
A block refers to one of OD, CFM, LC, heterogeneity, and simulation paraneters.
A block can hold a model of type Genetic Algorithm or SPSA'''

import logging
# Import available model implementations.
from models import GeneticAlgorithmModel, SPSAModel, NeuralNetworkModel

class Block:
    ''' Represents a single calibration block (e.g., OD, CFM, LC) and manages its 
    optimization model. Instantiates a model class based on the value of model_type_name'''
    def __init__(self, block_type_id, parameter_names_list, parameter_config_for_model,
                 chosen_model_type_str, # String identifier like "genetic_alg", "spsa".
                 sim_start_secs, sim_end_secs, global_data_config,
                 all_current_block_solutions, block_specific_alg_params):
        self.block_id = block_type_id         # e.g., "od", "cfm".
        self.model_type_name = chosen_model_type_str # Algorithm type for this block.
        # Arguments common to all model constructors.
        common_model_constructor_args = {
            "parameter_names_to_optimize": parameter_names_list,
            "parameter_config_for_model": parameter_config_for_model,
            "simulation_data_config": global_data_config,
            "sim_start_seconds": sim_start_secs,
            "sim_end_seconds": sim_end_secs,
            "block_identifier": self.block_id,
            "full_current_solutions_dict": all_current_block_solutions, # For context.
            "algorithm_specific_params": block_specific_alg_params
        }
        # Instantiate the appropriate optimization model based on 'chosen_model_type_str'.
        if self.model_type_name == "genetic_alg":
            self.optimization_model = GeneticAlgorithmModel(**common_model_constructor_args)
        elif self.model_type_name == "spsa":
            self.optimization_model = SPSAModel(**common_model_constructor_args)
        elif self.model_type_name == "neural_network": # As per original code's structure.
            self.optimization_model = NeuralNetworkModel(**common_model_constructor_args)
        else:
            logging.error("Invalid model type specified, block '%d': %s\n", self.block_id, self.model_type_name)
            raise ValueError(f"Unsupported model type: {self.model_type_name} for block {self.block_id}")
        logging.info("Successfully created '%s' model for block '%s'.\n", self.model_type_name, self.block_id)

    def execute_block_optimization(self, current_overall_best_fitness_cost):
        """execute_block_optimization()
        This function executes block optimization via calling the block's model object
        to execute run_optimization. If the block does not have a block_optimization object initialized
        it will log an error."""
        if self.optimization_model:
            # Model's run_optimization method should return: (list_of_fitness_history, best_solution_array, optimized_param_names_list)
            return self.optimization_model.run_optimization(current_overall_best_fitness_cost)
        # Should not happen if constructor raised ValueError for invalid model type.
        logging.error("No optimization model initialized for block '%d' (type: %s).\n", self.block_id, self.model_type_name)
        # Return empty/dummy results to prevent crashes if error was somehow bypassed.
        return [], [], []