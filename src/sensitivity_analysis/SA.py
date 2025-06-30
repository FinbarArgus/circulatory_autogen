'''
@author: Mohammad H. Shafieizadegan
@ reference: https://salib.readthedocs.io/en/latest/index.html
'''

import json
import os
from Functions.OpenCor_Py.opencor_helper import SimulationHelper
from SALib.sample import saltelli, sobol
import pandas as pd
from SALib.analyze import sobol
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Sensitivity_analysis():

    """
        A class for performing sensitivity analysis
        How to use:
        1. Initialize the class with the model path, output names, solver info, sensitivity analysis configuration, protocol info, time step, and save path.
        2. Call the `run` method with a feature extractor function and any additional arguments needed by that function.
        3. The 'run' method will generate sobol indices
        4. You can plot the results using the `plot_sobol_first_order_idx` and `plot_sobol_S2_idx` methods.
    """
        
    def __init__(self, model_path, model_out_names, solver_info, SA_cfg, protocol_info, dt, save_path, verbose=False):

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

        self.set_output_dir(save_path)

    def initialise_sim_helper(self):
        return SimulationHelper(self.model_path, self.dt, self.sim_time,
                                solver_info=self.solver_info, pre_time=self.pre_time)

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
        
        y = self.sim_helper.get_results(self.model_output_names)
        t = self.sim_helper.tSim - self.pre_time
        return y, t

    def generate_outputs(self, samples, feature_extractor, *f_args, **f_kwargs):
        
        outputs = []
        for i in range(len(samples)):

            current_param_val = samples[i, :]
            y, t = self.run_model_and_get_results(current_param_val)
            
            # Extract features using the provided feature_extractor function with additional arguments
            features = feature_extractor(t, np.squeeze(y), *f_args, **f_kwargs)
            outputs.append(features)

            if self.verbose:
                print(f"Iteration {i+1}/{len(samples)}: Features extracted.")

        return outputs
    
    def sobol_index(self, outputs):

        outputs = np.array(outputs)
    
        if outputs.ndim == 1:
            outputs = outputs[:, np.newaxis]  # convert to (n_samples, 1)

        n_outputs = outputs.shape[1]
        S1_all = np.zeros((n_outputs, self.num_params))
        ST_all = np.zeros((n_outputs, self.num_params))
        S2_all = np.zeros((n_outputs, self.num_params, self.num_params))

        for i in range(n_outputs):
            print(outputs[:,i])
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

    def run(self, feature_extractor, *f_args, **f_kwargs):
        samples = self.generate_samples()
        outputs = self.generate_outputs(samples, feature_extractor, f_args, **f_kwargs)
        S1_all, ST_all, S2_all = self.sobol_index(outputs)
        
        return S1_all, ST_all, S2_all
    



