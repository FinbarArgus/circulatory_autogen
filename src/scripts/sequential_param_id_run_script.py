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

root_dir_path = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir_path, 'src'))

resources_dir_path = os.path.join(root_dir_path, 'resources')
param_id_dir_path = os.path.join(root_dir_path, 'src/param_id')
generated_models_dir_path = os.path.join(root_dir_path, 'generated_models')

from param_id.sequential_paramID import SequentialParamID
import traceback

if __name__ == '__main__':

    try:
        DEBUG = True
        mpi_debug = False

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        num_procs = comm.Get_size()
        print(f'starting script for rank = {rank}')

        start_time = time.time()

        # FOR MPI DEBUG WITH PYCHARM
        # set mpi_debug to True
        # You have to change the configurations to "python debug server/mpi" and
        # click the debug button as many times as processes you want. You
        # must but the ports for each process in port_mapping.
        # Then simply run through mpiexec
        if mpi_debug:
            import pydevd_pycharm

            port_mapping = [39917, 36067]
            pydevd_pycharm.settrace('localhost', port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)

        if len(sys.argv) != 5:
            print(f'incorrect number of inputs to sequential_param_id_run_script.py')
            exit()

        param_id_method = sys.argv[1]
        file_name_prefix = sys.argv[2]
        model_path = os.path.join(generated_models_dir_path, f'{file_name_prefix}.cellml')
        param_id_model_type = 'CVS0D'  # TODO make this an input variable eventually

        input_params_path = os.path.join(resources_dir_path, f'{file_name_prefix}_params_for_id.csv')
        if not os.path.exists(input_params_path):
            print(f'input_params_path of {input_params_path} doesn\'t exist, user must create this file')
            exit()

        param_id_obs_path = sys.argv[4]
        if not os.path.exists(param_id_obs_path):
            print(f'param_id_obs_path={param_id_obs_path} does not exist')
            exit()

        # set the simulation time where the cost is calculated (sim_time) and the amount of
        # simulation time it takes to get to an oscilating steady state before that (pre_time)
        if file_name_prefix == '3compartment' or 'FTU_wCVS':
            pre_time = 20.0
        else:
            pre_time = 20.0
        sim_time = 1.0

        num_calls_to_function = int(sys.argv[3])
        seq_param_id = SequentialParamID(model_path, param_id_model_type, param_id_method, file_name_prefix,
                                input_params_path=input_params_path,
                                param_id_obs_path=param_id_obs_path,
                                num_calls_to_function=num_calls_to_function,
                                sim_time=sim_time, pre_time=pre_time, maximumStep=0.001, DEBUG=DEBUG)

        seq_param_id.run()

        best_param_vals = seq_param_id.param_id.get_best_param_vals()
        best_param_names = seq_param_id.get_best_param_names()

        if rank == 0:
            wall_time = time.time() - start_time
            print(f'wall time = {wall_time}')
            np.save(os.path.join(seq_param_id.param_id.output_dir, 'wall_time.npy'), wall_time)

    except:
        print(traceback.format_exc())
        print("Usage: parameter_id_method file_name_prefix num_calls_to_function "
              "path_to_obs_file.json")
        print("e.g. genetic_algorithm simple_physiological 10 "
              "path/to/circulatory_autogen/resources/simple_physiological_obs_data.json")
        comm.Abort()
        exit
