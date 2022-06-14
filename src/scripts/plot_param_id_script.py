'''
Created on 29/10/2021

@author: Finbar J. Argus
'''

import sys
import os
from mpi4py import MPI
from distutils import util

root_dir_path = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir_path, 'src'))

resources_dir_path = os.path.join(root_dir_path, 'resources')
param_id_dir_path = os.path.join(root_dir_path, 'src/param_id')
generated_models_dir_path = os.path.join(root_dir_path, 'generated_models')

from param_id.paramID import CVS0DParamID
from utilities import obj_to_string
import traceback
from distutils import util

if __name__ == '__main__':

    try:

        if len(sys.argv) != 5:
            print(f'incorrect number of inputs to plot_param_id.py script')
            exit()
        param_id_method = sys.argv[1]
        file_name_prefix = sys.argv[2]
        model_path = os.path.join(generated_models_dir_path, f'{file_name_prefix}.cellml')
        param_id_model_type = 'CVS0D' # TODO make this an input variable eventually

        input_params_path = os.path.join(resources_dir_path, f'{file_name_prefix}_params_for_id.csv')
        if not os.path.exists(input_params_path):
            print(f'input_params_path of {input_params_path} doesn\'t exist, user must create this file')
            exit()

        param_id_obs_path = os.path.join(resources_dir_path, sys.argv[3])
        run_sensitivity = sys.argv[4] in ['True', 'true']

        # set the simulation time where the cost is calculated (sim_time) and the amount of 
        # simulation time it takes to get to an oscilating steady state before that (pre_time)
        if file_name_prefix == '3compartment':
            pre_time = 20.0
        else: 
            pre_time = 20.0
        sim_time = 1.0


        param_id = CVS0DParamID(model_path, param_id_model_type, param_id_method, False, file_name_prefix,
                                input_params_path=input_params_path,
                                param_id_obs_path=param_id_obs_path,
                                sim_time=sim_time, pre_time=pre_time, maximumStep=0.001)

        with open(os.path.join(param_id.output_dir, 'param_names_to_remove.csv'), 'r') as r:
            param_names_to_remove = []
            for row in r:
                name_list = row.split(',')
                name_list = [name.strip() for name in name_list]
                param_names_to_remove.append(name_list)
        param_id.remove_params_by_name(param_names_to_remove)

        # print(obj_to_string(param_id))
        # TODO I need to remove params here if there are params to remove from sequential_param_id

        param_id.simulate_with_best_param_vals()
        param_id.plot_outputs()
        param_id.plot_mcmc()
        param_id.save_prediction_data()
        if run_sensitivity:
            param_id.run_sensitivity(None)
        param_id.close_simulation()

    except:
        print(traceback.format_exc())
        print("Usage: param_id_method file_name_prefix input_params_to_id, param_id_obs_file")
        print("e.g. bayesian simple_physiological True simple_physiological_obs_data.json")
        exit()
