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

def multimodal_gaussian(output, prob_dist_params, weight):
    
    # TODO make it so the below checks are only done once.
    # TODO checks should be done when parsing rather than every time this is called.
    allowable_keys = ["means", "stds", "scales"].sort()
    if not isinstance(prob_dist_params, dict):
        print("prob_dist_params needs to be a dict with entries :")
        print(allowable_keys)
        exit()

    if prob_dist_params.keys().sort() != allowable_keys:
        print("prob_dist_params needs to be a dict with entries :")
        print(allowable_keys)
        exit()

    cost_exp = 0
    for desired_mean, std, scales in zip(dist_params["means"], dist_params["stds"], dist_params["weights"]):
        cost_mode = np.power((output - desired_mean)/std, 2)*weight
        if hasattr(output, '__len__'):
            # if entry is a vector then turn the vector of costs for each data point into a average cost
            cost_mode = np.sum(cost_mode)/len(output)

        cost_exp += np.exp(cost_mode)
    
    cost = np.log(cost_exp)*weight

    return cost

def AE(output, desired_mean, std, weight):
    cost = np.abs((output - desired_mean)/std)*weight
    if hasattr(output, '__len__'):
        # if entry is a vector then turn the vector of costs for each data point into a average cost
        cost = np.sum(cost)/len(output)









