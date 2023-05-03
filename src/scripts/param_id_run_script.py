'''
Created on 29/10/2021

@author: Finbar J. Argus
'''

import sys
import os
from mpi4py import MPI
from distutils import util, dir_util


root_dir_path = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir_path, 'src'))

resources_dir_path = os.path.join(root_dir_path, 'resources')
user_inputs_path = os.path.join(root_dir_path, 'user_run_files')
param_id_dir_path = os.path.join(root_dir_path, 'src/param_id')
generated_models_dir_path = os.path.join(root_dir_path, 'generated_models')

from param_id.paramID import CVS0DParamID
import traceback
import yaml

def run_param_id(inp_data_dict=None):

    if inp_data_dict is None:
        with open(os.path.join(user_inputs_path, 'user_inputs.yaml'), 'r') as file:
            inp_data_dict = yaml.load(file, Loader=yaml.FullLoader)

    DEBUG = inp_data_dict['DEBUG']

    if DEBUG:
        print('WARNING: DEBUG IS ON, TURN THIS OFF IF YOU WANT TO DO ANYTHING QUICKLY')
    mpi_debug = inp_data_dict['MPI_DEBUG']

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()
    print(f'starting script for rank = {rank}')

    # FOR MPI DEBUG WITH PYCHARM
    # set mpi_debug to True
    # You have to change the configurations to "python debug server/mpi" and
    # click the debug button as many times as processes you want. You
    # must but the ports for each process in port_mapping.
    # Then simply run through mpiexec
    if mpi_debug:
        import pydevd_pycharm

        port_mapping = [37979, 34075]
        pydevd_pycharm.settrace('localhost', port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)

    param_id_method = inp_data_dict['param_id_method']
    file_prefix = inp_data_dict['file_prefix']
    generated_models_subdir_path = os.path.join(generated_models_dir_path, file_prefix)
    model_path = os.path.join(generated_models_subdir_path, f'{file_prefix}.cellml')
    param_id_model_type = inp_data_dict['param_id_model_type']

    input_params_path = os.path.join(resources_dir_path, f'{file_prefix}_params_for_id.csv')
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
    # set the simulation number of periods where the cost is calculated (sim_heart_periods) and the amount of
    # periods it takes to get to an oscillating steady state before that (pre_heart_periods)
    # if these exist they overwrite the pre_time and sim_time
    if 'pre_heart_periods' in inp_data_dict.keys():
        pre_heart_periods = inp_data_dict['pre_heart_periods']
    else:
        pre_heart_periods = None
    if 'sim_heart_periods' in inp_data_dict.keys():
        sim_heart_periods = inp_data_dict['sim_heart_periods']
    else:
        sim_heart_periods = None

    if pre_time == None and pre_heart_periods == None:
        print('pre_time and pre_heart_periods are undefined, one of these must be set in user_inputs.yaml')
    if sim_time == None and sim_heart_periods == None:
        print('sim_time and sim_heart_periods are undefined, one of these must be set in user_inputs.yaml')

    maximum_step = inp_data_dict['maximum_step']
    dt = inp_data_dict['dt']
    ga_options = inp_data_dict['ga_options']

    param_id = CVS0DParamID(model_path, param_id_model_type, param_id_method, False, file_prefix,
                            input_params_path=input_params_path,
                            param_id_obs_path=param_id_obs_path,
                            sim_time=sim_time, pre_time=pre_time,
                            sim_heart_periods=sim_heart_periods, pre_heart_periods=pre_heart_periods,
                            maximum_step=maximum_step, dt=dt, ga_options=ga_options, DEBUG=DEBUG)

    if rank == 0:
        if os.path.exists(os.path.join(param_id.output_dir, 'param_names_to_remove.csv')):
            os.remove(os.path.join(param_id.output_dir, 'param_names_to_remove.csv'))

    if DEBUG:
        num_calls_to_function = inp_data_dict['debug_ga_options']['num_calls_to_function']
    else:
        num_calls_to_function = inp_data_dict['ga_options']['num_calls_to_function']

    if param_id_method == 'genetic_algorithm':
        param_id.set_genetic_algorithm_parameters(num_calls_to_function)
    elif param_id_method == 'bayesian':
        acq_func = 'PI'  # 'gp_hedge'
        n_initial_points = 5
        random_seed = 1234
        acq_func_kwargs = {'xi': 0.01, 'kappa': 0.1} # these parameters favour exploitation if they are low
                                                            # and exploration if high, see scikit-optimize docs.
                                                            # xi is used when acq_func is “EI” or “PI”,
                                                            # kappa is used when acq_func is "LCB"
                                                            # gp_hedge, chooses the best from "EI", "PI", and "LCB
                                                            # so it needs both xi and kappa
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
        mcmc = CVS0DParamID(model_path, param_id_model_type, param_id_method, True, file_prefix,
                                input_params_path=input_params_path,
                                param_id_obs_path=param_id_obs_path,
                                sim_time=sim_time, pre_time=pre_time,
                                pre_heart_periods=pre_heart_periods, sim_heart_periods=sim_heart_periods,
                                maximum_step=maximum_step, dt=dt, mcmc_options=mcmc_options, DEBUG=DEBUG)
        mcmc.set_best_param_vals(best_param_vals)
        # mcmc.set_mcmc_parameters() TODO
        mcmc.run_mcmc()
        

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    try:
        run_param_id()
    except:
        print(traceback.format_exc())
        comm.Abort()
        exit
