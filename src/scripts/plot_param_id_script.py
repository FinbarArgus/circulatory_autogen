'''
Created on 29/10/2021

@author: Finbar J. Argus
'''

import sys
import os
from mpi4py import MPI
from distutils import util
import re

root_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir, 'src'))

user_inputs_dir = os.path.join(root_dir, 'user_run_files')

from param_id.paramID import CVS0DParamID
from param_id.sequential_paramID import SequentialParamID
from scripts.script_generate_with_new_architecture import generate_with_new_architecture
from utilities.utility_funcs import obj_to_string
import traceback
from distutils import util
import yaml


def plot_param_id(inp_data_dict=None):

    if inp_data_dict is None:
        with open(os.path.join(user_inputs_dir, 'user_inputs.yaml'), 'r') as file:
            inp_data_dict = yaml.load(file, Loader=yaml.FullLoader)
        if "user_inputs_path_override" in inp_data_dict.keys() and inp_data_dict["user_inputs_path_override"]:
            if os.path.exists(inp_data_dict["user_inputs_path_override"]):
                with open(inp_data_dict["user_inputs_path_override"], 'r') as file:
                    inp_data_dict = yaml.load(file, Loader=yaml.FullLoader)
            else:
                print(f"User inputs file not found at {inp_data_dict['user_inputs_path_override']}")
                print("Check the user_inputs_path_override key in user_inputs.yaml and set it to False if "
                        "you want to use the default user_inputs.yaml location")
                exit()

    plot_predictions = inp_data_dict['plot_predictions']

    param_id_method = inp_data_dict['param_id_method']
    file_prefix = inp_data_dict['file_prefix']
    DEBUG = inp_data_dict['DEBUG']

    
    resources_dir = os.path.join(root_dir, 'resources')
    param_id_output_dir = os.path.join(root_dir, 'param_id_output')
    generated_models_dir = os.path.join(root_dir, 'generated_models')
    
    # overwrite dir paths if set in user_inputs.yaml
    if "resources_dir" in inp_data_dict.keys():
        resources_dir = inp_data_dict['resources_dir']
    if "param_id_output_dir" in inp_data_dict.keys():
        param_id_output_dir = inp_data_dict['param_id_output_dir']
    if "generated_models_dir" in inp_data_dict.keys():
        generated_models_dir = inp_data_dict['generated_models_dir']
    

    param_id_obs_path = inp_data_dict['param_id_obs_path']
    if not os.path.exists(param_id_obs_path):
        print(f'param_id_obs_path={param_id_obs_path} does not exist')
        exit()

    data_str_addon = re.sub('.json', '', os.path.split(param_id_obs_path)[1])
    # here we get the subdir of the generated model that has the fitted params in it.
    generated_models_subdir= os.path.join(generated_models_dir, file_prefix + '_' + data_str_addon)
    # generated_models_subdir = os.path.join(generated_models_dir, file_prefix)

    # run the generation script with new param values
    generate_with_new_architecture(True, inp_data_dict=inp_data_dict)

    # generated_models_subdir = os.path.join(generated_models_dir, file_prefix)

    model_path = os.path.join(generated_models_subdir, f'{file_prefix}.cellml')
    model_type = inp_data_dict['model_type']

    if 'params_for_id_file' in inp_data_dict.keys():
        params_for_id_path = os.path.join(resources_dir, inp_data_dict['params_for_id_file'])
    else:
        params_for_id_path = os.path.join(resources_dir, f'{file_prefix}_params_for_id.csv')

    if not os.path.exists(params_for_id_path):
        print(f'params_for_id_path of {params_for_id_path} doesn\'t exist, user must create this file')
        exit()

    do_sensitivity = inp_data_dict['do_sensitivity']
    do_mcmc = inp_data_dict['do_mcmc']

    if 'pre_time' in inp_data_dict.keys():
        pre_time = inp_data_dict['pre_time']
    else:
        pre_time = None
    if 'sim_time' in inp_data_dict.keys():
        sim_time = inp_data_dict['sim_time']
    else:
        sim_time = None
        
    if inp_data_dict['solver_info'] is None:
        print('solver_info must be defined in user_inputs.yaml',
              'maximum_step is now an entry of solver_info in the user_inputs.yaml file')
        exit()
    solver_info = inp_data_dict['solver_info']
    dt = inp_data_dict['dt']
    if DEBUG:
        ga_options = inp_data_dict['debug_ga_options']
        mcmc_options = inp_data_dict['debug_mcmc_options']
    else:
        ga_options = inp_data_dict['ga_options']
        mcmc_options = inp_data_dict['mcmc_options']

    param_id = CVS0DParamID(model_path, model_type, param_id_method, False, file_prefix,
                            params_for_id_path=params_for_id_path,
                            param_id_obs_path=param_id_obs_path,
                            sim_time=sim_time, pre_time=pre_time,
                            solver_info=solver_info, ga_options=ga_options, dt=dt,
                            param_id_output_dir=param_id_output_dir, resources_dir=resources_dir)

    if os.path.exists(os.path.join(param_id.output_dir, 'param_names_to_remove.csv')):
        with open(os.path.join(param_id.output_dir, 'param_names_to_remove.csv'), 'r') as r:
            param_names_to_remove = []
            for row in r:
                name_list = row.split(',')
                name_list = [name.strip() for name in name_list]
                param_names_to_remove.append(name_list)
        param_id.remove_params_by_name(param_names_to_remove)

    # simulate with best values first to check cost
    param_id.simulate_with_best_param_vals()
    param_id.plot_outputs()
    if do_mcmc:
        if os.path.exists(os.path.join(param_id.output_dir, 'mcmc_chain.npy')):
            if not plot_predictions:
                param_id.plot_mcmc()
        else:
            print('no mcmc chain has been created at ' + 
                    os.path.join(param_id.output_dir, 'mcmc_chain.npy'))
    param_id.save_prediction_data()
    if do_sensitivity:
        param_id.run_sensitivity(None)
    param_id.close_simulation()

    if plot_predictions:
        seq_param_id = SequentialParamID(model_path, model_type, param_id_method, file_prefix,
                                            params_for_id_path=params_for_id_path,
                                            param_id_obs_path=param_id_obs_path,
                                            num_calls_to_function=1,
                                            sim_time=sim_time, pre_time=pre_time,
                                            solver_info=solver_info, dt=dt, ga_options=ga_options, 
                                            mcmc_options=mcmc_options, DEBUG=DEBUG,
                                            param_id_output_dir=param_id_output_dir, resources_dir=resources_dir)

        if do_mcmc:
            seq_param_id.plot_mcmc_and_predictions()

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    try:
        plot_param_id()
    except:
        print(traceback.format_exc())
        comm.Abort()
        exit()