'''
Created on 29/10/2021

@author: Gonzalo D. Maso Talou, Finbar J. Argus
'''

import sys
import os
import re
import traceback
import yaml
root_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir, 'src'))

user_inputs_dir= os.path.join(root_dir, 'user_run_files')

from parsers.ModelParsers import CSV0DModelParser
from generators.CVSCellMLGenerator import CVS0DCellMLGenerator
# TODO Cpp generator is commented out for now in the no_libcellml version
# from generators.CVSCppGenerator import CVS0DCppGenerator


def generate_with_new_architecture(do_generation_with_fit_parameters,
                                   inp_data_dict=None):
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

    file_prefix = inp_data_dict['file_prefix']
    input_param_file = inp_data_dict['input_param_file']

    resources_dir = os.path.join(root_dir, 'resources')
    param_id_output_dir = os.path.join(root_dir, 'param_id_output')
    generated_models_dir = os.path.join(root_dir, 'generated_models')
    
    # overwrite dir paths if set in user_inputs.yaml
    if "resources_dir" in inp_data_dict.keys():
        resources_dir = inp_data_dict['resources_dir']
    if "generated_models_dir" in inp_data_dict.keys():
        generated_models_dir = inp_data_dict['generated_models_dir']
    if "param_id_output_dir" in inp_data_dict.keys():
        param_id_output_dir = inp_data_dict['param_id_output_dir']

    vessels_csv_abs_path = os.path.join(resources_dir, file_prefix + '_vessel_array.csv')
    parameters_csv_abs_path = os.path.join(resources_dir, input_param_file)

    if do_generation_with_fit_parameters:
        param_id_obs_path = inp_data_dict['param_id_obs_path']
        param_id_method = inp_data_dict['param_id_method']
        data_str_addon = re.sub('\.json', '', os.path.split(param_id_obs_path)[1])
        param_id_output_dir_abs_path = os.path.join(param_id_output_dir, param_id_method + f'_{file_prefix}_{data_str_addon}')
        parser = CSV0DModelParser(vessels_csv_abs_path, parameters_csv_abs_path, 
                                  param_id_output_dir_abs_path)
        output_model_subdir = os.path.join(generated_models_dir, file_prefix + '_' + data_str_addon)
    else:
        parser = CSV0DModelParser(vessels_csv_abs_path, parameters_csv_abs_path)
        output_model_subdir = os.path.join(generated_models_dir, file_prefix)

    model = parser.load_model()

    if not os.path.exists(generated_models_dir):
        os.mkdir(generated_models_dir)
    if not os.path.exists(output_model_subdir):
        os.mkdir(output_model_subdir)

    if inp_data_dict['model_type'] == 'cellml_only':
        code_generator = CVS0DCellMLGenerator(model, output_model_subdir, file_prefix,
                                          resources_dir=resources_dir)
        code_generator.generate_files()
    elif inp_data_dict['model_type'] == 'cpp':
        if inp_data_dict['couple_to_1d']:
            code_generator = CVS0DCppGenerator(model, output_model_subdir, file_prefix,
                                            resources_dir=resources_dir, solver=inp_data_dict['solver'], 
                                            couple_to_1d=inp_data_dict['couple_to_1d'],
                                            cpp_generated_models_dir=inp_data_dict['cpp_generated_models_dir'],
                                            cpp_1d_model_config_path=inp_data_dict['cpp_1d_model_config_path'])
        else:
            code_generator = CVS0DCppGenerator(model, output_model_subdir, file_prefix,
                                            resources_dir=resources_dir, solver=inp_data_dict['solver'])

        code_generator.generate_cellml()
        code_generator.annotate_cellml()
        code_generator.generate_cpp()

    else: 
        print('model_type must be either cellml_only or cpp, not ' + inp_data_dict['model_type'])
        exit()


if __name__ == '__main__':
    try:
        do_generation_with_fit_parameters = sys.argv[1] in ['true', 'True']
        generate_with_new_architecture(do_generation_with_fit_parameters)

    except:
        print(traceback.format_exc())
        print("Usage with id params: do_generation_with_fit_parameters")
        print("e.g. script_generate_with_new_architecture.py True")
        exit()
