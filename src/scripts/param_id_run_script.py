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
param_id_dir_path = os.path.join(root_dir_path, 'src/param_id')
generated_models_dir_path = os.path.join(root_dir_path, 'generated_models')

from param_id.paramID import CVS0DParamID
import traceback

if __name__ == '__main__':

    try:
        mpi_debug = False

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

        if len(sys.argv) != 7:
            print(f'incorrect number of inputs to param_id_run_script.py')
            exit()

        param_id_method = sys.argv[1]
        file_name_prefix = sys.argv[2]
        model_path = os.path.join(generated_models_dir_path, f'{file_name_prefix}.cellml')
        param_id_model_type = 'CVS0D' # TODO make this an input variable eventually

        input_params_path = os.path.join(resources_dir_path, f'{file_name_prefix}_params_for_id.csv')
        if not os.path.exists(input_params_path):
            print(f'input_params_path of {input_params_path} doesn\'t exist, user must create this file')
            exit()
        sensitivity_params_path = os.path.join(resources_dir_path, f'{file_name_prefix}_params_for_sensitivity.csv')
        if not os.path.exists(sensitivity_params_path):
            sensitivity_params_path = input_params_path

        param_id_obs_path = sys.argv[4]
        if not os.path.exists(param_id_obs_path):
            print(f'param_id_obs_path={param_id_obs_path} does not exist')
            exit()

        # set the simulation time where the cost is calculated (sim_time) and the amount of 
        # simulation time it takes to get to an oscilating steady state before that (pre_time)
        if file_name_prefix == '3compartment' or 'FTU_wCVS':
          pre_time = 16.0
        else: 
          pre_time = 20.0
        sim_time = 2.0

        param_id = CVS0DParamID(model_path, param_id_model_type, param_id_method, False, file_name_prefix,
                                input_params_path=input_params_path,
                                sensitivity_params_path=sensitivity_params_path,
                                param_id_obs_path=param_id_obs_path,
                                sim_time=sim_time, pre_time=pre_time, maximumStep=0.00003, DEBUG=True)

        num_calls_to_function = int(sys.argv[3])
        if param_id_method == 'genetic_algorithm':
            param_id.set_genetic_algorithm_parameters(num_calls_to_function)
        elif param_id_method == 'bayesian':
            acq_func = 'PI'  # 'gp_hedge'
            n_initial_points = 5
            random_seed = 1234
            acq_func_kwargs = {'xi': 0.01, 'kappa': 0.1} # these parameters favour exploitation if they are low
                                                             # and exploration if high, see scikit-optimize docs.
                                                             # xi is used when acq_func is ???EI??? or ???PI???,
                                                             # kappa is used when acq_func is "LCB"
                                                             # gp_hedge, chooses the best from "EI", "PI", and "LCB
                                                             # so it needs both xi and kappa
            param_id.set_bayesian_parameters(num_calls_to_function, n_initial_points, acq_func,  random_seed,
                                             acq_func_kwargs=acq_func_kwargs)
        num_param_id_runs = int(sys.argv[5])
        if num_param_id_runs > 1:
            # get output_dir
            if rank == 0:
                output_dir_base = param_id.output_dir
                output_dirs = []
                best_cost_all_runs = 999999
                best_idx = 0
            for run_idx in range(num_param_id_runs):
                if rank == 0:
                    output_dir_new = output_dir_base + f'{run_idx}'
                    output_dirs.append(output_dir_new)
                    file_with_output_paths = os.path.join(output_dir_base,
                                                          f'{file_name_prefix}_param_id_output_paths.csv')
                    with open(file_with_output_paths, 'w') as wf:
                        output_dir_lines = [output_dir + '\n' for output_dir in output_dirs]
                        output_dir_lines.insert(0, 'path \n')
                        wf.writelines(output_dir_lines)

                    param_id.set_output_dir(output_dir_new)
                param_id.run()
                if rank == 0:
                    if param_id.param_id.best_cost < best_cost_all_runs:
                        best_idx = run_idx
                        best_cost_all_runs = param_id.param_id.best_cost

            if rank == 0:
                # copy directory of best cost to directory with no number idx.
                dir_util.copy_tree(output_dirs[best_idx], output_dir_base)

        else:
            if rank == 0:
                output_dirs = [param_id.output_dir]
                file_with_output_paths = os.path.join(os.path.split(output_dirs[0])[0], f'{file_name_prefix}_param_id_output_paths.csv')
                with open(file_with_output_paths, 'w') as wf:
                    output_dir_lines = [output_dir+'\n' for output_dir in output_dirs]
                    output_dir_lines.insert(0, 'path \n')
                    wf.writelines(output_dir_lines)
            param_id.run()

        best_param_vals = param_id.get_best_param_vals()
        param_id.close_simulation()
        do_mcmc = sys.argv[6] in ['True', 'true']
        if do_mcmc:
            mcmc = CVS0DParamID(model_path, param_id_model_type, param_id_method, True, file_name_prefix,
                                    input_params_path=input_params_path,
                                    sensitivity_params_path=sensitivity_params_path,
                                    param_id_obs_path=param_id_obs_path,
                                    sim_time=sim_time, pre_time=pre_time, maximumStep=0.00003, DEBUG=True)
            mcmc.set_best_param_vals(best_param_vals)
            # mcmc.set_mcmc_parameters() TODO
            mcmc.run_mcmc()
            

    except:
        print(traceback.format_exc())
        print("Usage: parameter_id_method file_name_prefix num_calls_to_function "
              "path_to_obs_file.json do_mcmc")
        print("e.g. genetic_algorithm simple_physiological 10 "
              "path/to/circulatory_autogen/resources/simple_physiological_obs_data.json True")
        comm.Abort()
        exit
