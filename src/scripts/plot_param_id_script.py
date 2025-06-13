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
from parsers.PrimitiveParsers import YamlFileParser


def plot_param_id(inp_data_dict=None):


    yaml_parser = YamlFileParser()
    inp_data_dict = yaml_parser.parse_user_inputs_file(inp_data_dict, obs_path_needed=True, do_generation_with_fit_parameters=True)

    DEBUG = inp_data_dict['DEBUG']
    model_path = inp_data_dict['model_path']
    model_type = inp_data_dict['model_type']
    param_id_method = inp_data_dict['param_id_method']
    file_prefix = inp_data_dict['file_prefix']
    params_for_id_path = inp_data_dict['params_for_id_path']
    param_id_obs_path = inp_data_dict['param_id_obs_path']
    sim_time = inp_data_dict['sim_time']
    pre_time = inp_data_dict['pre_time']
    solver_info = inp_data_dict['solver_info']
    dt = inp_data_dict['dt']
    ga_options = inp_data_dict['ga_options']
    mcmc_options = inp_data_dict['mcmc_options']
    resources_dir = inp_data_dict['resources_dir']
    param_id_output_dir = inp_data_dict['param_id_output_dir']
    plot_predictions = inp_data_dict['plot_predictions']
    do_sensitivity = inp_data_dict['do_sensitivity']
    do_mcmc = inp_data_dict['do_mcmc']
    
    # run the generation script with new param values
    generate_with_new_architecture(True, inp_data_dict=inp_data_dict)

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