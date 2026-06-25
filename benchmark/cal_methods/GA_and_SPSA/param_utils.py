import yaml
import time # Retained from original if any function still subtly depends on it.

def set_intermediate_file(parameter_name, parameter_value, filename):
    # Sets a specific parameter and its value in the intermediate YAML file.
    try:
        with open(filename, 'r') as file:
            intermediate_file = yaml.safe_load(file) or {}
    except FileNotFoundError:
        intermediate_file = {}
    intermediate_file[parameter_name] = parameter_value
    with open(filename, 'w') as file:
        yaml.dump(intermediate_file, file)

def get_intermediate_file(filename):
    # Retrieves all parameters and their values from the intermediate YAML file.
    with open(filename, 'r') as file:
        intermediate_file = yaml.safe_load(file)
    return intermediate_file