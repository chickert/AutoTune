import numpy as np

from spsa_library.algorithm import Algorithm
from spsa_library.evaluator import Evaluator
from spsa_library.problem import Problem

import logging
logger = logging.getLogger(__name__)

# https://www.jhuapl.edu/spsa/PDF-SPSA/Spall_Implementation_of_the_Simultaneous.PDF

class SPSAAlgorithm(Algorithm):
    def __init__(self, problem: Problem, perturbation_factor, gradient_factor, perturbation_exponent = 0.101, gradient_exponent = 0.602, gradient_offset = 0, compute_objective = True, seed = 0):
        # Problem
        self.problem = problem
        self._require_initial_values(self.problem.get_parameters())

        # Algorithm settings
        self.compute_objective = compute_objective

        self.perturbation_factor = perturbation_factor
        self.perturbation_exponent = perturbation_exponent

        self.gradient_factor = gradient_factor
        self.gradient_exponent = gradient_exponent
        self.gradient_offset = gradient_offset

        self.seed = seed
        self.dimensions = self._get_dimensions(problem)

        # Algorithm state
        self.random = np.random.RandomState(self.seed)
        self.iteration = 0
        self.values = self._require_initial_values(self.problem.get_parameters())

        self.saved_values = [] # need to save {"guess": val}, {"positive": val}, {"negative": val}

        # Ephemeral state
        self.gradient_length = np.nan
        self.perturbation_length = np.nan

        self._warn_ignore_bounds(self.problem.get_parameters(), logger)

    def set_state(self, state):
        assert self.dimensions == len(state["values"])

        self.iteration = state["iteration"]
        self.random.set_state(state["random"])
        self.values = state["values"]

    def get_state(self):
        return {
            "iteration": self.iteration,
            "random": self.random.get_state(),
            "values": self.values,

            # Just for information
            "gradient_length": self.gradient_length,
            "perturbation_length": self.perturbation_length
        }

    def get_settings(self):
        return {
            "compute_objective": self.compute_objective,
            "perturbation_factor": self.perturbation_factor,
            "perturbation_exponent": self.perturbation_exponent,
            "gradient_factor": self.gradient_factor,
            "gradient_exponent": self.gradient_exponent,
            "gradient_offset": self.gradient_offset,
            "seed": self.seed
        }
    
    def normalize(self, values):
        bounds = self.problem.get_parameters()
        norm_val = []
        for i in range(len(values)):
            val = values[i]
            bound = bounds[i].get_bounds()
            var = bound[1] - bound[0]
            offset = bound[0]
            norm_val.append((val-offset)/var)
        # print(f"Normalized Values: {norm_val}")
        return norm_val

    def unnormalize(self, values):
        bounds = self.problem.get_parameters()
        unnorm_val = []
        for i in range(len(values)):
            val = values[i]
            bound = bounds[i].get_bounds()
            
            var = bound[1] - bound[0]
            offset = bound[0]
            unnorm_val.append((val*var)+offset)
        # print(f"Unnormalized Values: {unnorm_val}")
        return unnorm_val

    def advance(self, evaluator: Evaluator):
        self.iteration += 1

        # Update lengths
        self.gradient_length = self.gradient_factor / (self.iteration + self.gradient_offset)**self.gradient_exponent
        self.perturbation_length = self.perturbation_factor / self.iteration**self.perturbation_exponent

        logger.debug("SPSA Iteration {} (Gradient {}, Perturbation {})".format(
            self.iteration, self.gradient_length, self.perturbation_length
        ))

        # Calculate objective
        if self.compute_objective:
            objective_identifier = evaluator.submit_one(self.values, { "type": "objective" })

        # Sample direction from Rademacher distribution
        direction = self.random.randint(0, 2, self.dimensions) - 0.5

        # Schedule samples
        positive_values = np.copy(self.values)
        norm_positive_values = self.normalize(positive_values)
        norm_positive_values += direction * self.perturbation_length
        norm_positive_values = np.clip(norm_positive_values, 0, 1)
        # print("positive values before unnormalize: ", norm_positive_values)
        positive_values = self.unnormalize(norm_positive_values)
        positive_objective = evaluator.submit_one(positive_values, { "type": "positive_gradient" })
        # print("Positive Values: ", positive_values)

        negative_values = np.copy(self.values)
        norm_negative_values = self.normalize(negative_values)
        norm_negative_values -= direction * self.perturbation_length
        norm_negative_values = np.clip(norm_negative_values, 0, 1)
        negative_values = self.unnormalize(norm_negative_values)
        negative_objective = evaluator.submit_one(negative_values, { "type": "negative_gradient" })
        # print("Negative Values: ", negative_values)

        # if self.compute_objective:
        #     evaluator.clean_one(objective_identifier)

        # positive_objective = evaluator.get_one(positive_identifier).get_objective()
        # evaluator.clean_one(positive_identifier)

        # negative_objective = evaluator.get_one(negative_identifier).get_objective()
        # evaluator.clean_one(negative_identifier)

        gradient = (positive_objective - negative_objective) / (2.0 * self.perturbation_length)
        gradient *= direction**-1

        # Update state
        #self.values -= self.gradient_length * gradient # need to normalize the gradient update
        normalized_values = self.normalize(self.values)

        gradient_norm = np.linalg.norm(gradient)

        # Prevent division by zero
        if gradient_norm != 0:
            gradient = gradient / gradient_norm
        else:
            gradient = np.zeros_like(gradient)  # or leave unchanged depending on your design

        # Update state with normalized gradient
        normalized_values -= self.gradient_length * gradient
        self.values = self.unnormalize(normalized_values)
        # print(f"NEW UPDATED VALUES {self.values}")
