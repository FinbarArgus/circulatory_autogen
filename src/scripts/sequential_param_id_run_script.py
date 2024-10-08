'''
Created on 26/04/2022

@author: Finbar J. Argus
'''

import sys
import os
from mpi4py import MPI
import time
import numpy as np
from distutils import util, dir_util

root_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir, 'src'))

user_inputs_dir = os.path.join(root_dir, 'user_run_files')

from param_id.sequential_paramID import SequentialParamID
import traceback
import yaml

if __name__ == '__main__':

    try:
        with open(os.path.join(user_inputs_dir, 'user_inputs.yaml'), 'r') as file:
            inp_data_dict = yaml.load(file, Loader=yaml.FullLoader)
        if inp_data_dict["user_inputs_path_override"]:
            if os.path.exists(inp_data_dict["user_inputs_path_override"]):
                with open(inp_data_dict["user_inputs_path_override"], 'r') as file:
                    inp_data_dict = yaml.load(file, Loader=yaml.FullLoader)
            else:
                print(f"User inputs file not found at {inp_data_dict['user_inputs_path_override']}")
                print("Check the user_inputs_path_override key in user_inputs.yaml and set it to False if "
                        "you want to use the default user_inputs.yaml location")
                exit()

        DEBUG = inp_data_dict['DEBUG']
        if DEBUG:
            print('WARNING: DEBUG IS ON, TURN THIS OFF IF YOU WANT TO DO ANYTHING QUICKLY')

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        num_procs = comm.Get_size()
        print(f'starting script for rank = {rank}')

        start_time = time.time()

        resources_dir = os.path.join(root_dir, 'resources')
        param_id_dir = os.path.join(root_dir, 'src/param_id')
        generated_models_dir = os.path.join(root_dir, 'generated_models')

        param_id_method = inp_data_dict['param_id_method']
        file_prefix = inp_data_dict['file_prefix']

        # overwrite dir paths if set in user_inputs.yaml
        if "resources_dir" in inp_data_dict.keys():
            resources_dir = inp_data_dict['resources_dir']
        if "param_id_output_dir" in inp_data_dict.keys():
            param_id_output_dir = inp_data_dict['param_id_output_dir']
        if "generated_models_dir" in inp_data_dict.keys():
            generated_models_dir = inp_data_dict['generated_models_dir']
        
        generated_models_subdir = os.path.join(generated_models_dir, file_prefix)
        model_path = os.path.join(generated_models_subdir, f'{file_prefix}.cellml')
        model_type = inp_data_dict['model_type']

        input_params_path = os.path.join(resources_dir, f'{file_prefix}_params_for_id.csv')
        if not os.path.exists(input_params_path):
            print(f'input_params_path of {input_params_path} doesn\'t exist, user must create this file')
            exit()

        param_id_obs_path = inp_data_dict['param_id_obs_path']
        if not os.path.exists(param_id_obs_path):
            print(f'param_id_obs_path={param_id_obs_path} does not exist')
            exit()


        if 'pre_time' in inp_data_dict.keys():
            pre_time = inp_data_dict['pre_time']
        else:
            pre_time = None
        if 'sim_time' in inp_data_dict.keys():
            sim_time = inp_data_dict['sim_time']
        else:
            sim_time = None

        solver_info = inp_data_dict['solver_info']
        dt = inp_data_dict['dt']
        ga_options = inp_data_dict['ga_options']

        if DEBUG:
            num_calls_to_function = inp_data_dict['debug_ga_options']['num_calls_to_function']
        else:
            num_calls_to_function = inp_data_dict['ga_options']['num_calls_to_function']

        if DEBUG:
            mcmc_options = inp_data_dict['debug_mcmc_options']
        else:
            mcmc_options = inp_data_dict['mcmc_options']

        seq_param_id = SequentialParamID(model_path, model_type, param_id_method, file_prefix,
                                input_params_path=input_params_path,
                                param_id_obs_path=param_id_obs_path,
                                num_calls_to_function=num_calls_to_function,
                                solver_info=solver_info, dt=dt, mcmc_options=mcmc_options, ga_options=ga_options,
                                DEBUG=DEBUG, 
                                param_id_output_dir=param_id_output_dir, resources_dir=resources_dir)

        seq_param_id.run()

        best_param_vals = seq_param_id.param_id.get_best_param_vals()
        best_param_names = seq_param_id.get_best_param_names()

        if rank == 0:
            wall_time = time.time() - start_time
            print(f'wall time = {wall_time}')
            np.save(os.path.join(seq_param_id.param_id.output_dir, 'wall_time.npy'), wall_time)

    except:
        print(traceback.format_exc())
        comm.Abort()
        exit
