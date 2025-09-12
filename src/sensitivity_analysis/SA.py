'''
@author: Mohammad H. Shafieizadegan
@ reference: https://salib.readthedocs.io/en/latest/index.html
'''

import json
import os
import sys
from sys import exit
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utilities'))
import math as math
import opencor as oc
from opencor_helper import SimulationHelper
from SALib.sample import saltelli, sobol
import pandas as pd
from SALib.analyze import sobol
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from parsers.PrimitiveParsers import scriptFunctionParser
from mpi4py import MPI

class CVS0D_SA():

    """
        A class for performing sensitivity analysis
        How to use:
        1. Initialize the class with the model path, output names, solver info, sensitivity analysis configuration, protocol info, time step, and save path.
        2. Call the `run` method with a feature extractor function and any additional arguments needed by that function.
        3. The 'run' method will generate sobol indices
        4. You can plot the results using the `plot_sobol_first_order_idx` and `plot_sobol_S2_idx` methods.
    """
        
    def __init__(self, model_path, model_out_names, solver_info, SA_cfg, protocol_info, dt, 
                 save_path, param_id_path = None, use_MPI = False, verbose=False):

        """
        Initializes the Sensitivity_analysis class.
        Parameters:
            model_path (str): Path to the model file.
            model_out_names (list): Names of the model outputs to be analyzed.
            solver_info (dict): Solver configuration parameters.
            SA_cfg (dict): Configuration for sensitivity analysis, including sample type, number of samples,
                           parameter names, and their bounds.
            protocol_info (dict): Information about the simulation protocol, including simulation times and pre-times.
            dt (float): Time step for the simulation.
            save_path (str): Directory where results will be saved.
            verbose (bool): If True, prints additional information during execution.
        """

        self.model_path = model_path
        self.output_dir = None
        self.verbose = verbose
        self.save_path = save_path

        self.solver_info = solver_info
        self.SA_cfg = SA_cfg
        self.sample_type = self.SA_cfg["sample_type"]
        self.num_params = len(self.SA_cfg["param_names"])
        self.model_output_names = model_out_names

        # set up opencor simulation
        self.dt = dt
        if protocol_info['sim_times'][0][0] is not None:
            self.sim_time = protocol_info['sim_times'][0][0]
        else:
            # set temporary sim time, just to initialise the sim_helper
            self.sim_time = 0.001
        if protocol_info['pre_times'][0] is not None:
            self.pre_time = protocol_info['pre_times'][0]
        else:
            # set temporary pre time, just to initialise the sim_helper
            self.pre_time = 0.001

        self.sim_helper = self.initialise_sim_helper()
        self.sim_helper.update_times(self.dt, 0.0, self.sim_time, self.pre_time)
        self.n_steps = int(self.sim_time/self.dt)

        # set up observables functions
        sfp = scriptFunctionParser()
        self.operation_funcs_dict = sfp.get_operation_funcs_dict()
        self.__set_obs_names_and_df(param_id_path, self.pre_time, self.sim_time)

        self.set_output_dir(save_path)

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.num_procs = self.comm.Get_size()
        self.use_mpi = use_MPI

    def initialise_sim_helper(self):
        return SimulationHelper(self.model_path, self.dt, self.sim_time,
                                solver_info=self.solver_info, pre_time=self.pre_time)



    def __set_obs_names_and_df(self, param_id_obs_path, pre_time=None, sim_time=None):
        # TODO this function should be in the parsing section. as it parses the 
        # ground truth data.
        # TODO it should also be cleaned up substantially.
        """_summary_

        Args:
            param_id_obs_path (_type_): _description_
            pre_time (_type_): _description_
            sim_time (_type_): _description_
        """
        with open(param_id_obs_path, encoding='utf-8-sig') as rf:
            json_obj = json.load(rf)
        if type(json_obj) == list:
            self.gt_df = pd.DataFrame(json_obj)
            self.protocol_info = {"pre_times": [pre_time], 
                                    "sim_times": [[sim_time]],
                                    "params_to_change": [[None]]}
            self.prediction_info = {'names': [],
                                    'units': [],
                                    'names_for_plotting': [],
                                    'experiment_idxs': []}
        elif type(json_obj) == dict:
            if 'data_items' in json_obj.keys():
                self.gt_df = pd.DataFrame(json_obj['data_items'])
            elif 'data_item' in json_obj.keys():
                self.gt_df = pd.DataFrame(json_obj['data_item']) # should be data_items but accept this
            else:
                print("data_items not found in json object. ",
                      "Please check that data_items is the key for the list of data items")
            if 'protocol_info' in json_obj.keys():
                self.protocol_info = json_obj['protocol_info']
                if "sim_times" not in self.protocol_info.keys():
                    self.protocol_info["sim_times"] = [[sim_time]]
                if "pre_times" not in self.protocol_info.keys():
                    self.protocol_info["pre_times"] = [pre_time]
            else:
                if pre_time is None or sim_time is None:
                    print("protocol_info not found in json object. ",
                          "If this is the case sim_time and pre_time must be set",
                          "in the user_inputs.yaml file")
                    exit()

                self.protocol_info = {"pre_times": [pre_time], 
                                      "sim_times": [[sim_time]],
                                      "params_to_change": [[None]]}
            if 'prediction_items' in json_obj.keys():
                self.prediction_info = {'names': [],
                                        'units': [],
                                        'names_for_plotting': [],
                                        'experiment_idxs': []}

                for entry in json_obj['prediction_items']:
                    if 'variable' in entry.keys():
                        self.prediction_info['names'].append(entry['variable'])
                    else:
                        print('"variable" not found in prediction item in obs_data.json file, ',
                              'exitiing') 
                        exit()
                    if 'unit' in entry.keys():
                        self.prediction_info['units'].append(entry['unit'])
                    else:
                        print('"unit" not found in prediction item in obs_data.json file, ',
                              'exitiing') 
                        exit()
                    if 'name_for_plotting' in entry.keys():
                        self.prediction_info['names_for_plotting'].append(entry['name_for_plotting'])
                    else:
                        self.prediction_info['names_for_plotting'].append(entry['variable'])
                    if 'experiment_idx' in entry.keys():
                        self.prediction_info['experiment_idxs'].append(entry['experiment_idx'])
                    else:
                        self.prediction_info['experiment_idxs'].append(0)
            else:
                self.prediction_info = None
        else:
            print(f"unknown data type for imported json object of {type(json_obj)}")
        
        self.obs_info = {}
        self.obs_info["obs_names"] = [self.gt_df.iloc[II]["variable"] for II in range(self.gt_df.shape[0])]

        # OBSOLETE self.obs_types = [self.gt_df.iloc[II]["obs_type"] for II in range(self.gt_df.shape[0])]
        self.obs_info["data_types"] = [self.gt_df.iloc[II]["data_type"] for II in range(self.gt_df.shape[0])]
        self.obs_info["units"] = [self.gt_df.iloc[II]["unit"] for II in range(self.gt_df.shape[0])]
        self.obs_info["experiment_idxs"] = [self.gt_df.iloc[II]["experiment_idx"] if "experiment_idx" in 
                                            self.gt_df.iloc[II].keys() else 0 for II in range(self.gt_df.shape[0])]
        self.obs_info["subexperiment_idxs"] = [self.gt_df.iloc[II]["subexperiment_idx"] if "subexperiment_idx" in
                                               self.gt_df.iloc[II].keys() else 0 for II in range(self.gt_df.shape[0])]

        # get plotting color, asign to randomish color if not defined
        # list of all possible colors
        possible_colors = ['b', 'g', 'c', 'm', 'y', 
                           'tab:brown', 'tab:pink', 'tab:olive', 'tab:orange'] # don't include red or black, 
                                                    # because they are used for plotting the series
        self.obs_info["plot_colors"] = [self.gt_df.iloc[II]["plot_color"] if "plot_color" in 
                                        self.gt_df.iloc[II].keys() else possible_colors[II%len(possible_colors)] 
                                        for II in range(self.gt_df.shape[0])]
        self.obs_info["plot_type"] = []

        # get plotting type
        # TODO make the plot_types operation_funcs so the user can defined how they are plotted.
        warning_printed = False
        for II in range(self.gt_df.shape[0]):
            if "plot_type" not in self.gt_df.iloc[II].keys():
                if self.gt_df.iloc[II]["data_type"] == "constant":
                    if not warning_printed:
                        print('constant data types plot type defaults to horizontal lines',
                            'change "plot_type" in obs_data.json to change this')
                        warning_printed = True
                    self.obs_info["plot_type"].append("horizontal")
                elif self.gt_df.iloc[II]["data_type"] == "prob_dist":
                    if not warning_printed:
                        print('prob_dist data types plot type defaults to horizontal lines',
                            'change "plot_type" in obs_data.json to change this')
                        warning_printed = True
                    self.obs_info["plot_type"].append("horizontal")
                elif self.gt_df.iloc[II]["data_type"] == "series":
                    self.obs_info["plot_type"].append("series")
                elif self.gt_df.iloc[II]["data_type"] == "frequency":
                    self.obs_info["plot_type"].append("frequency")
                elif self.gt_df.iloc[II]["data_type"] == "plot_dist":
                    self.obs_info["plot_type"].append("horizontal")
                else:
                    print(f'data type {self.gt_df.iloc[II]["data_type"]} not recognised')
            else:
                self.obs_info["plot_type"].append(self.gt_df.iloc[II]["plot_type"])
                if self.obs_info["plot_type"][II] in ["None", "null", "Null", "none", "NONE"]:
                    self.obs_info["plot_type"][II] = None

        self.obs_info["operations"] = []
        self.obs_info["names_for_plotting"] = []
        self.obs_info["operands"] = []
        self.obs_info["freqs"] = []
        self.obs_info["operation_kwargs"] = []
        # below we remove the need for obs_types, but keep it backwards compatible so 
        # previous specifications of obs_type = mean etc should still work
        for II in range(self.gt_df.shape[0]):
            if "operation" not in self.gt_df.iloc[II].keys() or \
                    self.gt_df.iloc[II]["operation"] in ["Null", "None", "null", "none", "", "nan", np.nan]:
                if "obs_type" in self.gt_df.iloc[II].keys():
                    if self.gt_df.iloc[II]["obs_type"] == "series":
                        self.obs_info["operations"].append(None)
                        if "operands" in self.gt_df.iloc[II].keys():
                            self.obs_info["operands"].append(self.gt_df.iloc[II]["operands"])
                        else:
                            self.obs_info["operands"].append(None)
                    elif self.gt_df.iloc[II]["obs_type"] == "frequency":
                        self.obs_info["operations"].append(None)
                        if "operands" in self.gt_df.iloc[II].keys():
                            self.obs_info["operands"].append(self.gt_df.iloc[II]["operands"])
                        else:
                            self.obs_info["operands"].append(None)
                    # TODO remove these eventually when I get rid of obs_type
                    elif self.gt_df.iloc[II]["obs_type"] == "min":
                        self.obs_info["operations"].append("min")
                        self.obs_info["operands"].append([self.gt_df.iloc[II]["variable"]])
                    elif self.gt_df.iloc[II]["obs_type"] == "max":
                        self.obs_info["operations"].append("max")
                        self.obs_info["operands"].append([self.gt_df.iloc[II]["variable"]])
                    elif self.gt_df.iloc[II]["obs_type"] == "mean":
                        self.obs_info["operations"].append("mean")
                        self.obs_info["operands"].append([self.gt_df.iloc[II]["variable"]])
                else:
                    self.obs_info["operations"].append(None)
                    if "operands" in self.gt_df.iloc[II].keys():
                        self.obs_info["operands"].append(self.gt_df.iloc[II]["operands"])
                    else:
                        self.obs_info["operands"].append(None)
            elif self.gt_df.iloc[II]["operation"] in ["Null", "None", "null", "none", ""]:
                self.obs_info["operations"].append(None)
                self.obs_info["operands"].append(None)
            else:
                self.obs_info["operations"].append(self.gt_df.iloc[II]["operation"])
                self.obs_info["operands"].append(self.gt_df.iloc[II]["operands"])

            if "frequencies" not in self.gt_df.iloc[II].keys():
                self.obs_info["freqs"].append(None)
            else:
                self.obs_info["freqs"].append(self.gt_df.iloc[II]["frequencies"])

            if "name_for_plotting" in self.gt_df.iloc[II].keys():
                self.obs_info['names_for_plotting'].append(self.gt_df.iloc[II]["name_for_plotting"])
            else:
                self.obs_info['names_for_plotting'].append(self.obs_info["obs_names"][II])

            if "operation_kwargs" in self.gt_df.iloc[II].keys() and self.gt_df.iloc[II]["operation_kwargs"] \
                    not in ["Null", "None", "null", "none", "", np.nan]:
                self.obs_info["operation_kwargs"].append(self.gt_df.iloc[II]["operation_kwargs"])
            else:
                self.obs_info["operation_kwargs"].append({})

        self.obs_info["num_obs"] = len(self.obs_info["obs_names"])

        # how much to weight the different observable errors by
        self.obs_info["weight_const_vec"] = np.array([self.gt_df.iloc[II]["weight"] for II in range(self.gt_df.shape[0])
                                          if self.gt_df.iloc[II]["data_type"] == "constant"])

        self.obs_info["weight_series_vec"] = np.array([self.gt_df.iloc[II]["weight"] for II in range(self.gt_df.shape[0])
                                           if self.gt_df.iloc[II]["data_type"] == "series"])

        self.obs_info["weight_amp_vec"] = np.array([self.gt_df.iloc[II]["weight"] for II in range(self.gt_df.shape[0])
                                           if self.gt_df.iloc[II]["data_type"] == "frequency"])
        
        self.obs_info["weight_prob_dist_vec"] = np.array([self.gt_df.iloc[II]["weight"] for II in range(self.gt_df.shape[0])
                                          if self.gt_df.iloc[II]["data_type"] == "prob_dist"])

        weight_phase_list = [] 
        for II in range(self.gt_df.shape[0]):
            if self.gt_df.iloc[II]["data_type"] == "frequency":
                if "phase_weight" not in self.gt_df.iloc[II].keys():
                    weight_phase_list.append(1)
                else:
                    weight_phase_list.append(self.gt_df.iloc[II]["phase_weight"])
        self.obs_info["weight_phase_vec"] = np.array(weight_phase_list)

        # set the cost type for each observable
        self.obs_info["cost_type"] = []
        for II in range(self.gt_df.shape[0]):
            if "cost_type" in self.gt_df.iloc[II].keys() and self.gt_df.iloc[II]["cost_type"] not in [np.nan, None, "None", ""]:
                self.obs_info["cost_type"].append(self.gt_df.iloc[II]["cost_type"])
            else:
                if self.ga_options is not None:
                    if "cost_type" in self.ga_options.keys():
                        self.obs_info["cost_type"].append(self.ga_options["cost_type"]) # default to cost type in ga_options
                    else:
                        self.obs_info["cost_type"].append("MSE") # default to mean squared error
                elif self.mcmc_options is not None:
                    if "cost_type" in self.mcmc_options.keys():
                        self.obs_info["cost_type"].append(self.mcmc_options["cost_type"]) # default to cost type in mcmc_options
                    else:
                        self.obs_info["cost_type"].append("MSE") # default to mean squared error
                else:
                    print("cost_type not found in obs_data.json, ga_options, or mcmc_options, exiting")
                    exit()



        # preprocess information in the protocol_info dataframe
        self.protocol_info['num_experiments'] = len(self.protocol_info["sim_times"])
        self.protocol_info['num_sub_per_exp'] = [len(self.protocol_info["sim_times"][II]) for II in range(self.protocol_info["num_experiments"])]
        self.protocol_info['num_sub_total'] = sum(self.protocol_info['num_sub_per_exp'])

        # calculate total experiment sim times
        self.protocol_info["total_sim_times_per_exp"] = []
        self.protocol_info["tSims_per_exp"] = []
        self.protocol_info["num_steps_total_per_exp"] = []

        for exp_idx in range(self.protocol_info['num_experiments']):
            total_sim_time = np.sum([self.protocol_info["sim_times"][exp_idx][II] for
                            II in range(self.protocol_info["num_sub_per_exp"][exp_idx])])
            num_steps_total = int(total_sim_time/self.dt)
            tSim_per_exp = np.linspace(0.0, total_sim_time, num_steps_total + 1)
            self.protocol_info["total_sim_times_per_exp"].append(total_sim_time)
            self.protocol_info["tSims_per_exp"].append(tSim_per_exp)
            self.protocol_info["num_steps_total_per_exp"].append(num_steps_total)
            

        if "experiment_colors" not in self.protocol_info.keys():
            self.protocol_info["experiment_colors"] = ['r']
            if self.protocol_info['num_experiments'] > 1:
                self.protocol_info["experiment_colors"] = ['r']*self.protocol_info['num_experiments']
        else:
            if len(self.protocol_info["experiment_colors"]) != self.protocol_info['num_experiments']:
                print('experiment_colors in obs_data.json not the same length as num_experiments, exiting')
                exit()

        if "experiment_labels" in self.protocol_info.keys():
            if len(self.protocol_info["experiment_labels"]) != self.protocol_info['num_experiments']:
                print('experiment_labels in obs_data.json not the same length as num_experiments, exiting')
                exit()
        else:
            self.protocol_info["experiment_labels"] = [None]
            if self.protocol_info['num_experiments'] > 1:
                self.protocol_info["experiment_labels"] = [None]*self.protocol_info['num_experiments']
        
        # set experiment and subexperiment idxs to 0 if they are not defined. print warning if multiple subexperiments
        for II in range(self.gt_df.shape[0]):
            if "experiment_idx" not in self.gt_df.iloc[II].keys():
                self.gt_df["experiment_idx"] = 0
                if self.protocol_info['num_sub_total'] > 1:
                    print(f'experiment_idx not found in obs_data.json entry {self.gt_df.iloc[II]["variable"]}, '
                          'but multiple experiments are defined.',
                          'Setting experiment_idx to 0 for all data points')
            if "subexperiment_idx" not in self.gt_df.iloc[II].keys():
                self.gt_df["subexperiment_idx"] = 0
                if self.protocol_info['num_sub_total'] > 1:
                    print(f'subexperiment_idx not found in obs_data.json entry {self.gt_df.iloc[II]["variable"]}, '
                          'but multiple subexperiments are defined.',
                          'Setting subexperiment_idx to 0 for all data points')
        
        # calculate the mapping from sub and experiment idx to the weight of the observable for that subexperiment
        const_map = [[[] for sub_idx in range(self.protocol_info['num_sub_per_exp'][exp_idx])]
                     for exp_idx in range(self.protocol_info['num_experiments'])]
        series_map = [[[] for sub_idx in range(self.protocol_info['num_sub_per_exp'][exp_idx])]
                     for exp_idx in range(self.protocol_info['num_experiments'])]
        amp_map = [[[] for sub_idx in range(self.protocol_info['num_sub_per_exp'][exp_idx])]
                     for exp_idx in range(self.protocol_info['num_experiments'])]
        phase_map = [[[] for sub_idx in range(self.protocol_info['num_sub_per_exp'][exp_idx])]
                     for exp_idx in range(self.protocol_info['num_experiments'])]
        prob_dist_map = [[[] for sub_idx in range(self.protocol_info['num_sub_per_exp'][exp_idx])]
                     for exp_idx in range(self.protocol_info['num_experiments'])]

        for exp_idx in range(self.protocol_info['num_experiments']):
            for this_sub_idx in range(self.protocol_info['num_sub_per_exp'][exp_idx]):

                for II in range(self.gt_df.shape[0]):
                    if self.gt_df.iloc[II]["data_type"] == "constant":
                        if self.gt_df.iloc[II]["experiment_idx"] == exp_idx and \
                            self.gt_df.iloc[II]["subexperiment_idx"] == this_sub_idx:
                            const_map[exp_idx][this_sub_idx].append(self.gt_df.iloc[II]["weight"])
                        else:
                            # if the data point is not in assigned to this experiment/subexperiment, 
                            # set the weight mapping to 0, so it doesn't influence the cost in this 
                            # subexperiment
                            const_map[exp_idx][this_sub_idx].append(0.0)
                    if self.gt_df.iloc[II]["data_type"] == "series":
                        if self.gt_df.iloc[II]["experiment_idx"] == exp_idx and \
                            self.gt_df.iloc[II]["subexperiment_idx"] == this_sub_idx:
                            series_map[exp_idx][this_sub_idx].append(self.gt_df.iloc[II]["weight"])
                        else:
                            series_map[exp_idx][this_sub_idx].append(0.0)

                    if self.gt_df.iloc[II]["data_type"] == "frequency":
                        if self.gt_df.iloc[II]["experiment_idx"] == exp_idx and \
                            self.gt_df.iloc[II]["subexperiment_idx"] == this_sub_idx:
                            amp_map[exp_idx][this_sub_idx].append(self.gt_df.iloc[II]["weight"])
                            if "phase_weight" not in self.gt_df.iloc[II].keys():
                                # if there is no phase weight, weight it the same as the amplitude
                                phase_map[exp_idx][this_sub_idx].append(self.gt_df.iloc[II]["weight"])
                            else:
                                phase_map[exp_idx][this_sub_idx].append(self.gt_df.iloc[II]["phase_weight"])
                        else:
                            amp_map[exp_idx][this_sub_idx].append(0.0)
                            phase_map[exp_idx][this_sub_idx].append(0.0)

                    if self.gt_df.iloc[II]["data_type"] == "prob_dist":
                        if self.gt_df.iloc[II]["experiment_idx"] == exp_idx and \
                            self.gt_df.iloc[II]["subexperiment_idx"] == this_sub_idx:
                            prob_dist_map[exp_idx][this_sub_idx].append(self.gt_df.iloc[II]["weight"])
                        else:
                            prob_dist_map[exp_idx][this_sub_idx].append(0.0)

                # make each weight vector a numpy array
                const_map[exp_idx][this_sub_idx] = np.array(const_map[exp_idx][this_sub_idx])
                series_map[exp_idx][this_sub_idx] = np.array(series_map[exp_idx][this_sub_idx])
                amp_map[exp_idx][this_sub_idx] = np.array(amp_map[exp_idx][this_sub_idx])
                phase_map[exp_idx][this_sub_idx] = np.array(phase_map[exp_idx][this_sub_idx])
                prob_dist_map[exp_idx][this_sub_idx] = np.array(prob_dist_map[exp_idx][this_sub_idx])

        self.protocol_info["scaled_weight_const_from_exp_sub"] = const_map
        self.protocol_info["scaled_weight_series_from_exp_sub"] = series_map
        self.protocol_info["scaled_weight_amp_from_exp_sub"] = amp_map
        self.protocol_info["scaled_weight_phase_from_exp_sub"] = phase_map
        self.protocol_info["scaled_weight_prob_dist_from_exp_sub"] = prob_dist_map
        return
    
    def set_output_dir(self, path):
        
        self.output_dir = path
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def generate_samples(self):

        problem = {
            'num_vars': self.num_params,
            'names': self.SA_cfg["param_names"],
            'bounds': list(zip(self.SA_cfg["param_mins"], self.SA_cfg["param_maxs"]))
        }
        self.problem = problem

        self.num_samples = self.SA_cfg["num_samples"]

        if self.SA_cfg["sample_type"] == "saltelli":
            samples = saltelli.sample(problem, self.num_samples, calc_second_order=True)  # Enable second-order interactions
        elif self.SA_cfg["sample_type"] == "sobol":
            samples = sobol.sample(problem, self.num_samples, calc_second_order=True)  # Enable second-order interactions
        else:
            raise ValueError(f"Unsupported sample type: {self.SA_cfg['sample_type']}")
        
        return samples
    
    def run_model_and_get_results(self, param_vals):
        self.sim_helper.set_param_vals(self.SA_cfg["param_names"], param_vals)
        self.sim_helper.reset_states()
        self.sim_helper.run()
        
        # y = self.sim_helper.get_results(self.model_output_names)
        # t = self.sim_helper.tSim - self.pre_time
        # return y, t
        return self.sim_helper.get_results(self.obs_info["operands"])

    def generate_outputs(self, samples):
        
        outputs = []
        for i in range(len(samples)):

            current_param_val = samples[i, :]
            operands_outputs = self.run_model_and_get_results(current_param_val)
            
            # Extract features using the provided feature_extractor function with additional arguments
            features = []
            for j in range(len(self.obs_info["operands"])):
                 
                feature = self.operation_funcs_dict[self.obs_info["operations"][j]](*operands_outputs[j], **self.obs_info["operation_kwargs"][j]) 
                features.append(feature)

            outputs.append(features)
            self.sim_helper.reset_and_clear()

            if self.verbose:
                print(f"Iteration {i+1}/{len(samples)}: Features extracted.")

        outputs = np.array(outputs)  # Convert to 2D numpy array: (n_samples, n_features)
        return outputs
    
    def generate_outputs_mpi(self, samples):
        # Split the samples across ranks
        n_samples = len(samples)
        samples_per_rank = n_samples // self.num_procs
        remainder = n_samples % self.num_procs

        # Determine the start and end indices for this rank
        if self.rank < remainder:
            start = self.rank * (samples_per_rank + 1)
            end = start + samples_per_rank + 1
        else:
            start = self.rank * samples_per_rank + remainder
            end = start + samples_per_rank

        local_samples = samples[start:end]

        print(f"[MPI Rank {self.rank}] Processing samples {start}:{end} (total {len(local_samples)})")

        # Each rank computes its chunk
        local_outputs = []
        for i, param_vals in enumerate(local_samples):
            # if self.verbose:
            #     print(f"[MPI Rank {self.rank}] Processing local sample {i+1}/{len(local_samples)}")
            operands_outputs = self.run_model_and_get_results(param_vals)
            features = []
            for j in range(len(self.obs_info["operands"])):
                feature = self.operation_funcs_dict[self.obs_info["operations"][j]](
                    *operands_outputs[j], **self.obs_info["operation_kwargs"][j]
                )
                features.append(feature)
            local_outputs.append(features)

        print(f"[MPI Rank {self.rank}] Processing samples {start}:{end} (total {len(local_samples)})")
        print(f"[MPI Rank {self.rank}] Finished processing local samples.")

        # Gather results at rank 0
        all_outputs = self.comm.gather(local_outputs, root=0)

        if self.rank == 0:
            # Flatten the list of lists
            outputs = [item for sublist in all_outputs for item in sublist]
            outputs = np.array(outputs)
            print(f"[MPI Rank 0] Gathered and flattened all outputs. Total outputs: {outputs.shape}")
            return outputs
        else:
            return None
    
    def sobol_index(self, outputs):

        outputs = np.array(outputs)
    
        if outputs.ndim == 1:
            outputs = outputs[:, np.newaxis]  # convert to (n_samples, 1)

        n_outputs = outputs.shape[1]
        S1_all = np.zeros((n_outputs, self.num_params))
        ST_all = np.zeros((n_outputs, self.num_params))
        S2_all = np.zeros((n_outputs, self.num_params, self.num_params))

        for i in range(n_outputs):
            print(f">>>>>>>>>>>>>>>>    outputs: {len(outputs[:,i])}")
            Si = sobol.analyze(self.problem, outputs[:,i], print_to_console=self.verbose)
            S1_all[i, :] = Si['S1']
            ST_all[i, :] = Si['ST']
            S2_all[i, :] = np.array(Si['S2'])

        return S1_all, ST_all, S2_all

    def plot_sobol_first_order_idx(self, S1_all, ST_all):
        """
        Plot first-order and total-order Sobol indices for multiple outputs.

        Parameters:
            S1_all (np.ndarray): First-order Sobol indices, shape (n_outputs, n_params)
            ST_all (np.ndarray): Total-order Sobol indices, shape (n_outputs, n_params)
        """
        n_outputs = S1_all.shape[0]
        x = np.arange(self.num_params)

        for i in range(n_outputs):
            S1 = S1_all[i]
            ST = ST_all[i]
            output_name = self.output_names[i] if hasattr(self, "output_names") else f"Output_{i}"

            plt.figure(figsize=(8, 5))
            plt.bar(x - 0.2, S1, width=0.4, label='First-order', color='blue', alpha=0.7)
            plt.bar(x + 0.2, ST, width=0.4, label='Total-order', color='red', alpha=0.7)

            plt.xticks(x, self.SA_cfg["param_names"], rotation=45)
            plt.ylabel('Sensitivity Index')
            plt.title(f'Sobol Sensitivity - {output_name}')
            plt.legend()
            plt.tight_layout()

            file_name = f"{output_name}_First_order_idx.png"
            plt.savefig(os.path.join(self.save_path, file_name))
            plt.clf()

    def plot_sobol_S2_idx(self, S2_all):
        """
        Plot second-order Sobol interaction indices for multiple outputs.

        Parameters:
            S2_all (np.ndarray): Second-order indices, shape (n_outputs, n_params, n_params)
        """
        n_outputs = S2_all.shape[0]

        for i in range(n_outputs):
            S2 = S2_all[i]
            output_name = self.output_names[i] if hasattr(self, "output_names") else f"Output_{i}"

            plt.figure(figsize=(6, 5))
            sns.heatmap(S2, annot=True, fmt=".2f", xticklabels=self.SA_cfg["param_names"], yticklabels=self.SA_cfg["param_names"], cmap="coolwarm")
            plt.title(f"Second-order Sobol Indices - {output_name}")
            plt.tight_layout()

            filename = f"{output_name}_n{self.num_samples}_second_order_idx.png"
            plt.savefig(os.path.join(self.save_path, filename))
            plt.clf()

    def run(self):
        samples = self.generate_samples()
        if self.use_mpi:
            outputs = self.generate_outputs_mpi(samples)
            if self.rank == 0:
                S1_all, ST_all, S2_all = self.sobol_index(outputs)
                return S1_all, ST_all, S2_all
            else:
                return None, None, None
        else:
            outputs = self.generate_outputs(samples)
            S1_all, ST_all, S2_all = self.sobol_index(outputs)
            return S1_all, ST_all, S2_all

