import numpy as np
import pandas as pd
import yaml
import json
import os
root_dir = os.path.join(os.path.dirname(__file__), '../..')
scripts_dir = os.path.join(root_dir, 'src/scripts')
example_data_dir = os.path.join(scripts_dir, 'example_data')

class ObsDataCreator:
    def __init__(self):
        self.obs_data_dict = {}
        self.obs_data_dict['protocol_info'] = {}
        self.obs_data_dict['prediction_items'] = []
        self.obs_data_dict['data_items'] = []

    def add_protocol_info(self, pre_times, sim_times, params_to_change, experiment_labels):
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
        if len(experiment_labels) != num_exps:
            raise ValueError("experiment_labels should have the same length as the number of experiments (number of rows of sim_times).")
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
                         'unit', 'weight', 'value', 'std', 'dt', 
                         'experiment_idx', 'subexperiment_idx']
                         
        for key in required_keys:
            if key not in entry:
                raise ValueError(f"Entry is missing required key: {key}")
        
        # check that experiment_idx and subexperiment_idx are valid
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

if __name__ == "__main__":
    # here is an exaple of using the above class with an example data file

    # Load the data that you want to use as ground truth
    # change this to the path of your data file
    data_file = os.path.join(example_data_dir, 'example_data_for_conversion.csv')

    # output path for the JSON file
    # change this to the desired output path
    output_path = os.path.join(example_data_dir, 'NKE_pump_obs_data.json')
    data = pd.read_csv(data_file)

    # access the data in the way you want it
    print(data.columns)
    time = data['environment | t (second)'].values

    # create obs_data_creator instance
    obs_data_creator = ObsDataCreator()

    # first create protocol_info to define subexperiments and parameter changes for those subexperiments
    pre_times = [1]
    sim_times = [[100, float(time[-1])]]
    params_to_change = {}
    params_to_change['NKE_pump/flag_0'] = [[0.0, 1.0]]  # example parameter change
    experiment_labels = ['exp_0_subexp_0', 'exp_0_subexp_1']  # example labels

    obs_data_creator.add_protocol_info(pre_times, sim_times, params_to_change, experiment_labels)

    # now create the prediction items
    obs_data_creator.add_prediction_item('NKE_pump/v_2', 'fmol_per_s', 0)
    obs_data_creator.add_prediction_item('NKE_pump/u_e', 'fmol_per_s', 0)  

    ## Now fill the data items with the data you want to use for the parameter estimation
    # get data vecs from loaded data
    dt = time[1] - time[0]
    V_R1 = data['environment | v_R1 (fmol_per_sec)'].values

    # and make an entry
    entry = {}
    entry['variable'] = 'NKE_pump/v_1'
    entry['name_for_plotting'] = 'v1_{01}'
    entry['data_type'] = 'series'
    entry['operation'] = None
    entry['operands'] = ['NKE_pump/v_1']
    entry['unit'] = 'fmol_per_s'
    entry['weight'] = 1.0
    entry['value'] = V_R1.tolist()
    entry['std'] = (0.1*V_R1).tolist()
    entry['dt'] = dt
    entry['experiment_idx'] = 0
    entry['subexperiment_idx'] = 1
    obs_data_creator.add_data_item(entry)

    # a made up entry to test different dt
    dt = 0.5

    entry = {}
    entry['variable'] = 'NKE_pump/v_1'
    entry['name_for_plotting'] = 'v1_{00}'
    entry['data_type'] = 'series'
    entry['operation'] = None
    entry['operands'] = ['NKE_pump/v_1']
    entry['unit'] = 'fmol_per_s'
    entry['weight'] = 0.01

    data_vec = 13*np.ones((int(100/dt)))

    entry['value'] = data_vec.tolist()
    entry['std'] = (data_vec/10).tolist()
    entry['dt'] = dt
    entry['experiment_idx'] = 0
    entry['subexperiment_idx'] = 0
    obs_data_creator.add_data_item(entry)

    # add more entries to data_items if you have more data to add

    obs_data_dict = obs_data_creator.get_obs_data_dict()
    print(obs_data_dict)
    obs_data_creator.dump_to_path(output_path)

