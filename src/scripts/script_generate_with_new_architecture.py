'''
Created on 29/10/2021

@author: Gonzalo D. Maso Talou, Finbar J. Argus
'''

import sys
import os
import re
import traceback
import yaml
root_dir_path = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir_path, 'src'))

resources_dir_path = os.path.join(root_dir_path, 'resources')
param_id_dir_path = os.path.join(root_dir_path, 'param_id_output')
output_model_dir_path = os.path.join(root_dir_path, 'generated_models')
user_inputs_path = os.path.join(root_dir_path, 'user_run_files')

from parsers.ModelParsers import CSV0DModelParser
from generators.CVSCellMLGenerator import CVS0DCellMLGenerator


def generate_with_new_architecture(do_generation_with_fit_parameters,
                                   inp_data_dict=None):
    if inp_data_dict is None:
        with open(os.path.join(user_inputs_path, 'user_inputs.yaml'), 'r') as file:
            inp_data_dict = yaml.load(file, Loader=yaml.FullLoader)

    file_prefix = inp_data_dict['file_prefix']
    input_param_file = inp_data_dict['input_param_file']
    vessels_csv_abs_path = os.path.join(resources_dir_path, file_prefix + '_vessel_array.csv')
    parameters_csv_abs_path = os.path.join(resources_dir_path, input_param_file)

    if do_generation_with_fit_parameters:
        param_id_obs_path = inp_data_dict['param_id_obs_path']
        param_id_method = inp_data_dict['param_id_method']
        data_str_addon = re.sub('\.json', '', os.path.split(param_id_obs_path)[1])
        param_id_dir_abs_path = os.path.join(param_id_dir_path, param_id_method + f'_{file_prefix}_{data_str_addon}')
        parser = CSV0DModelParser(vessels_csv_abs_path, parameters_csv_abs_path, param_id_dir_abs_path)
        output_model_subdir_path = os.path.join(output_model_dir_path, file_prefix + '_' + data_str_addon)
    else:
        parser = CSV0DModelParser(vessels_csv_abs_path, parameters_csv_abs_path)
        output_model_subdir_path = os.path.join(output_model_dir_path, file_prefix)

    model = parser.load_model()

    code_generator = CVS0DCellMLGenerator(model, output_model_subdir_path, file_prefix)
    code_generator.generate_files()


if __name__ == '__main__':
    try:
        do_generation_with_fit_parameters = sys.argv[1] in ['true', 'True']
        generate_with_new_architecture(do_generation_with_fit_parameters)

    except:
        print(traceback.format_exc())
        print("Usage with id params: do_generation_with_fit_parameters")
        print("e.g. script_generate_with_new_architecture.py True")
        exit()