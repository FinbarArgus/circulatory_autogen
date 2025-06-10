import numpy as np
import pandas as pd
import yaml
import json
import os, sys
root_dir = os.path.join(os.path.dirname(__file__), '../..')

class ObsDataCreator:
    def __init__(self):
        self.obs_data_dict = {}
        self.obs_data_dict['protocol_info'] = {}
        self.obs_data_dict['prediction_items'] = []
        self.obs_data_dict['data_items'] = []

    def add_protocol_info(self, pre_times, sim_times, params_to_change, experiment_labels=None):
        """
        Add protocol information to the dictionary.
        pre_times: list of pre-simulation times for each experiment
        sim_times: 2D list of lists of simulation times for each experiment and subexperiment
        params_to_change: dictionary with parameter names as keys and list of lists of values Each parameter should have a value
        entry the same shape as sim_times.
        experiment_labels: list of labels for each experiment
        """
        # check sizes of lists are correct
        num_exps = len(sim_times)
        if len(pre_times) != num_exps:
            raise ValueError("pre_times should have the same length as the number of experiments (number of rows of sim_times).")
        if not all(isinstance(entry, list) for entry in sim_times):
            raise ValueError("sim_times should be a 2D list with one row for each experiment and a column for each subexperiment.")
        if experiment_labels is not None:
            if len(experiment_labels) != num_exps:
                raise ValueError("experiment_labels should have the same length as the number of experiments (number of rows of sim_times).")
        else:
            # if experiment_labels is not provided, create default labels
            experiment_labels = [f'exp_{i}' for i in range(num_exps)]
        if type(params_to_change) is not dict:
            raise ValueError("params_to_change should be a dictionary with parameter names as keys and lists of values as values.")
        for param, values in params_to_change.items():
            if len(values) != num_exps:
                raise ValueError(f"Parameter {param} should have the same number of values as the number of experiments ({num_exps}).")
            if not all(isinstance(v, list) for v in values):
                raise ValueError(f"Parameter {param} should have a list of values for each subexperiment.")
            if not all(len(v) == len(sim_times[i]) for i, v in enumerate(values)):
                raise ValueError(f"Parameter {param} should have the same number of values as the number of subexperiments for each experiment.")
        
        # now add to dict
        self.obs_data_dict['protocol_info']['pre_times'] = pre_times
        self.obs_data_dict['protocol_info']['sim_times'] = sim_times
        self.obs_data_dict['protocol_info']['params_to_change'] = params_to_change
        self.obs_data_dict['protocol_info']['experiment_labels'] = experiment_labels

    def add_prediction_item(self, variable, unit, experiment_idx):
        """
        Add a prediction item to the dictionary.
        variable: name of the variable to predict
        unit: unit of the variable
        experiment_idx: index of the experiment this prediction item belongs to
        """
        # check that experiment_idx is valid
        if experiment_idx < 0 or experiment_idx >= len(self.obs_data_dict['protocol_info']['sim_times']):
            raise ValueError(f"experiment_idx {experiment_idx} is out of bounds for the number of experiments ({len(self.obs_data_dict['protocol_info']['sim_times'])}).")

        prediction_item = {
            'variable': variable,
            'unit': unit,
            'experiment_idx': experiment_idx
        }
        self.obs_data_dict['prediction_items'].append(prediction_item)

    #TODO Create functions for adding entries to the data_items
    def add_data_item(self, entry):
        """
        Add a data item to the dictionary.
        entry: dictionary containing the data item
        """
        required_keys = ['variable', 'name_for_plotting', 'data_type', 'operation', 'operands', 
                         'unit', 'weight', 'value', 'std']
                         
        if 'subexperiment_idx' not in entry:
            entry['subexperiment_idx'] = 0  # default to 0 if not provided
        if 'experiment_idx' not in entry:
            entry['experiment_idx'] = 0  # default to 0 if not provided
        for key in required_keys:
            if key not in entry:
                raise ValueError(f"Entry is missing required key: {key}")
        if entry["data_type"] == 'series':
            if 'obs_dt' in entry.keys():
                pass
            elif 'dt' in entry.keys():
                print("Warning: 'dt' for the time step of series data items is deprecated, ",
                      "please use 'obs_dt' instead. Setting 'obs_dt' to 'dt'.")
                pass
            else:
                raise ValueError(f"obs_dt is required for series entries")

        if self.obs_data_dict['protocol_info'] != {}:
            # check that experiment_idx and subexperiment_idx are valid if there is a protocol_info
            if entry['experiment_idx'] < 0 or entry['experiment_idx'] >= len(self.obs_data_dict['protocol_info']['sim_times']):
                raise ValueError(f"experiment_idx {entry['experiment_idx']} is out of bounds for the number of experiments ({len(self.obs_data_dict['protocol_info']['sim_times'])}).")
            if entry['subexperiment_idx'] < 0 or entry['subexperiment_idx'] >= len(self.obs_data_dict['protocol_info']['sim_times'][entry['experiment_idx']]):
                raise ValueError(f"subexperiment_idx {entry['subexperiment_idx']} is out of bounds for the number of subexperiments in experiment {entry['experiment_idx']} ({len(self.obs_data_dict['protocol_info']['sim_times'][entry['experiment_idx']])}).")

        self.obs_data_dict['data_items'].append(entry)
    
    def get_obs_data_dict(self):
        """
        Returns the observation data dictionary.
        """
        return self.obs_data_dict

    def dump_to_path(self, output_path):
        """
        Dumps the observation data dictionary to a JSON file.
        """
        with open(output_path, 'w') as f:
            json.dump(self.obs_data_dict, f, indent=2)
        print(f"Observation data dumped to {output_path}")
