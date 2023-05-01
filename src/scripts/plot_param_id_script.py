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
user_inputs_path = os.path.join(root_dir_path, 'user_run_files')
param_id_dir_path = os.path.join(root_dir_path, 'src/param_id')
generated_models_dir_path = os.path.join(root_dir_path, 'generated_models')

from param_id.paramID import CVS0DParamID
from param_id.sequential_paramID import SequentialParamID
from utilities import obj_to_string
import traceback
from distutils import util
import yaml

if __name__ == '__main__':

    try:

        with open(os.path.join(user_inputs_path, 'user_inputs.yaml'), 'r') as file:
            inp_data_dict = yaml.load(file, Loader=yaml.FullLoader)

        plot_predictions = inp_data_dict['plot_predictions']

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
        # set the simulation number of periods where the cost is calculated (sim_heart_periods) and the amount of
        # periods it takes to get to an oscilating steady state before that (pre_heart_periods)
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
                                maximum_step=maximum_step, ga_options=ga_options, dt=dt)

        if os.path.exists(os.path.join(param_id.output_dir, 'param_names_to_remove.csv')):
            with open(os.path.join(param_id.output_dir, 'param_names_to_remove.csv'), 'r') as r:
                param_names_to_remove = []
                for row in r:
                    name_list = row.split(',')
                    name_list = [name.strip() for name in name_list]
                    param_names_to_remove.append(name_list)
            param_id.remove_params_by_name(param_names_to_remove)


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
            seq_param_id = SequentialParamID(model_path, param_id_model_type, param_id_method, file_prefix,
                                             input_params_path=input_params_path,
                                             param_id_obs_path=param_id_obs_path,
                                             num_calls_to_function=1,
                                             sim_time=sim_time, pre_time=pre_time,
                                             sim_heart_periods=sim_heart_periods, pre_heart_periods=pre_heart_periods,
                                             maximum_step=maximum_step, dt=dt, ga_options=ga_options, DEBUG=False)

            if do_mcmc:
                seq_param_id.plot_mcmc_and_predictions()

    except:
        print(traceback.format_exc())
        exit()
