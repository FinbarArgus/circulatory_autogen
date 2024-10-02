'''
Created on 29/10/2021

@author: Finbar J. Argus
'''

import sys
import os
from mpi4py import MPI
from distutils import util, dir_util


root_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir, 'src'))

user_inputs_dir = os.path.join(root_dir, 'user_run_files')

from param_id.paramID import CVS0DParamID
import traceback
import yaml

def run_param_id(inp_data_dict=None):

    if inp_data_dict is None:
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

    param_id_method = inp_data_dict['param_id_method']
    file_prefix = inp_data_dict['file_prefix']

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

    generated_models_subdir = os.path.join(generated_models_dir, file_prefix)
    model_path = os.path.join(generated_models_subdir, f'{file_prefix}.cellml')
    model_type = inp_data_dict['model_type']
    if 'params_for_id_file' in inp_data_dict.keys():
        params_for_id_path = os.path.join(resources_dir, inp_data_dict['params_for_id_file'])
    else:
        params_for_id_path = os.path.join(resources_dir, f'{file_prefix}_params_for_id.csv')

    if not os.path.exists(params_for_id_path):
        print(f'params_for_id path of {params_for_id_path} doesn\'t exist, user must create this file')
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

    if inp_data_dict['solver_info'] is None:
        print('solver_info must be defined in user_inputs.yaml',
              'MaximumStep is now an entry of solver_info in the user_inputs.yaml file')
        exit()
    solver_info = inp_data_dict['solver_info']
    dt = inp_data_dict['dt']
    if DEBUG:
        ga_options = inp_data_dict['debug_ga_options']
    else:
        ga_options = inp_data_dict['ga_options']

    param_id = CVS0DParamID(model_path, model_type, param_id_method, False, file_prefix,
                            params_for_id_path=params_for_id_path,
                            param_id_obs_path=param_id_obs_path,
                            sim_time=sim_time, pre_time=pre_time,
                            solver_info=solver_info, dt=dt, ga_options=ga_options, DEBUG=DEBUG,
                            param_id_output_dir=param_id_output_dir, resources_dir=resources_dir)

    if rank == 0:
        if os.path.exists(os.path.join(param_id.output_dir, 'param_names_to_remove.csv')):
            os.remove(os.path.join(param_id.output_dir, 'param_names_to_remove.csv'))


    if param_id_method == 'bayesian':
        acq_func = 'PI'  # 'gp_hedge'
        n_initial_points = 5
        random_seed = 1234
        acq_func_kwargs = {'xi': 0.01, 'kappa': 0.1} # these parameters favour exploitation if they are low
                                                            # and exploration if high, see scikit-optimize docs.
                                                            # xi is used when acq_func is “EI” or “PI”,
                                                            # kappa is used when acq_func is "LCB"
                                                            # gp_hedge, chooses the best from "EI", "PI", and "LCB
                                                            # so it needs both xi and kappa
        # TODO this needs to be defined better if we want to keep bayesian optimiser functionality
        if DEBUG:
            num_calls_to_function = inp_data_dict['debug_ga_options']['num_calls_to_function']
        else:
            num_calls_to_function = inp_data_dict['ga_options']['num_calls_to_function']
        param_id.set_bayesian_parameters(num_calls_to_function, n_initial_points, acq_func,  random_seed,
                                            acq_func_kwargs=acq_func_kwargs)
    param_id.run()

    best_param_vals = param_id.get_best_param_vals()
    param_id.close_simulation()
    do_mcmc = inp_data_dict['do_mcmc']

    if DEBUG:
        mcmc_options = inp_data_dict['debug_mcmc_options']
    else:
        mcmc_options = inp_data_dict['mcmc_options']

    if do_mcmc:
        mcmc = CVS0DParamID(model_path, model_type, param_id_method, True, file_prefix,
                                params_for_id_path=params_for_id_path,
                                param_id_obs_path=param_id_obs_path,
                                sim_time=sim_time, pre_time=pre_time,
                                solver_info=solver_info, dt=dt, mcmc_options=mcmc_options, DEBUG=DEBUG,
                                param_id_output_dir=param_id_output_dir, resources_dir=resources_dir)
        mcmc.set_best_param_vals(best_param_vals)
        # mcmc.set_mcmc_parameters() TODO
        mcmc.run_mcmc()
    
    if rank == 0:
        print('param id complete')
        

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    try:
        run_param_id()
        MPI.Finalize()
    except:
        print(traceback.format_exc())
        comm.Abort()
        exit()
