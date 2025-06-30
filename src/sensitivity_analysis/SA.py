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