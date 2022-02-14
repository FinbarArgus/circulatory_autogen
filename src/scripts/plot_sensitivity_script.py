'''
Created on 23/12/2021

@author: Finbar J. Argus
Modified by M Savage
'''

import sys
import os
from mpi4py import MPI

root_dir_path = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir_path, 'src'))

resources_dir_path = os.path.join(root_dir_path, 'resources')
param_id_dir_path = os.path.join(root_dir_path, 'src/param_id')
generated_models_dir_path = os.path.join(root_dir_path, 'generated_models')

from param_id.paramID import CVS0DParamID
from utilities import obj_to_string
import traceback

if __name__ == '__main__':

    try:

        if len(sys.argv) != 4:
            print(f'incorrect number of inputs')
            exit()
        param_id_method = sys.argv[1]
        file_name_prefix = sys.argv[2]
        model_path = os.path.join(generated_models_dir_path, f'{file_name_prefix}.cellml')
        param_id_model_type = 'CVS0D' # TODO make this an input variable eventually

        input_params_path = os.path.join(resources_dir_path, f'{file_name_prefix}_params_for_id.csv')
        sensitivity_params_path = os.path.join(resources_dir_path, f'{file_name_prefix}_params_for_sensitivity.csv')
        param_id_obs_path = os.path.join(resources_dir_path, sys.argv[3])

        # set the simulation time where the cost is calculated (sim_time) and the amount of 
        # simulation time it takes to get to an oscilating steady state before that (pre_time)
        if file_name_prefix == '3compartment':
          pre_time = 6.0
        else: 
          pre_time = 20.0
        sim_time = 2.0


        param_id = CVS0DParamID(model_path, param_id_model_type, param_id_method, file_name_prefix,
                                input_params_path=input_params_path,
                                sensitivity_params_path=sensitivity_params_path,
                                param_id_obs_path=param_id_obs_path,
                                sim_time=sim_time, pre_time=pre_time, maximumStep=0.001)

        # print(obj_to_string(param_id))
        param_id.run_sensitivity()
        param_id.plot_sensitivity()
        param_id.close_simulation()

    except:
        print(traceback.format_exc())
        print("Usage: param_id_method file_name_prefix param_id_obs_file")
        print("e.g. bayesian simple_physiological simple_physiological_obs_data.json")
        exit()