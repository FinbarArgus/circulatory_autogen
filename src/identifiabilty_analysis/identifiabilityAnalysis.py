'''
@author: Finbar J. Argus
'''

import numpy as np
import os
import sys
from sys import exit
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utilities'))
import math as math
import opencor as oc
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import paperPlotSetup
import diagnostics
import utility_funcs
from utility_funcs import calculate_hessian
import traceback
from utility_funcs import Normalise_class
paperPlotSetup.Setup_Plot(3)
from opencor_helper import SimulationHelper
from parsers.PrimitiveParsers import scriptFunctionParser
from mpi4py import MPI
import re
from numpy import genfromtxt
from importlib import import_module
import csv
from datetime import date
# from skopt import gp_minimize, Optimizer
from parsers.PrimitiveParsers import CSVFileParser
import pandas as pd
import json
import math
# from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib/..*" )

class IdentifiabilityAnalysis():
    """
    Class for doing identifiability analysis on a 0D model
    """
    def __init__(self, model_path, model_type, file_name_prefix, DEBUG=False,
                 param_id_output_dir=None, resources_dir=None, param_id=None):

        self.model_path = model_path
        self.model_type = model_type
        self.file_name_prefix = file_name_prefix
        self.DEBUG = DEBUG
        self.param_id_output_dir = param_id_output_dir
        self.resources_dir = resources_dir
        self.best_param_vals = None
        self.param_id = param_id
        if self.param_id is None:
            # TODO intialise the param_id_object here
            raise ValueError("param_id object must be provided to IdentifiabilityAnalysis")
        
        # TODO
        pass
    
    def set_best_param_vals(self, best_param_vals):
        self.best_param_vals = best_param_vals

    def run_identifiability_analysis(self, ia_options):
        """
        Run the identifiability analysis based on the chosen method
        """
        if ia_options['method'] == 'profile_likelihood':
            self.run_profile_likelihood(ia_options)
        elif ia_options['method'] == 'Laplace':
            self.run_laplace_approximation(ia_options)

    def run_profile_likelihood(self, ia_options):
        # TODO
        pass

    def run_laplace_approximation(self, ia_options):

        Hessian = calculate_hessian(self.param_id)
        print("Hessian Matrix:\n", Hessian)
        covariance_matrix = np.linalg.inv(Hessian)
        mean = self.best_param_vals
        print("Laplace Approximation Results:")
        print("Mean (Best Parameter Values):", mean)
        print("Covariance Matrix:\n", covariance_matrix)
        return mean, covariance_matrix
