import numpy as np
import os
import sys
import sympy

"""
These functions can be used as cost functions. Specify a name of one of these functions as the "cost_type" in obs_data.json to
use it as the cost.

When making your own cost function make sure it works for scalars and vectors. Otherwise put an error message so that if it is used
for the wrong data type it gets called out and stopped.
"""

def gaussian_MLE(output, desired_mean, std, weight):
    # cost = np.sum(np.power(updated_weight_const_vec*(const -
    #                            self.obs_info["ground_truth_const"])/self.obs_info["std_const_vec"], 2))

    # cost = series_cost = np.sum(np.power((series[:, :min_len_series] -
    #                                            self.obs_info["ground_truth_series"][:,
    #                                            :min_len_series]) * updated_weight_series_vec.reshape(-1, 1) /
    #                                           self.obs_info["std_series_vec"].reshape(-1, 1), 2)) / min_len_series

    cost = np.power((output - desired_mean)/std, 2)*weight
    if hasattr(output, '__len__'):
        # if entry is a vector then turn the vector of costs for each data point into a average cost
        cost = np.sum(cost)/len(output)
    
    return cost
# TODO we need to create derivative functions for each cost with respect to the outputs so that we can pass 

def MSE(*args, **kwargs):
    # The mean squared error cost is the same as the 
    return gaussian_MLE(*args, **kwargs)

def multimodal_gaussian_mix():
    # TODO
    pass

def AE(output, desired_mean, std, weight):
    cost = np.abs((output - desired_mean)/std)*weight
    if hasattr(output, '__len__'):
        # if entry is a vector then turn the vector of costs for each data point into a average cost
        cost = np.sum(cost)/len(output)









