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
from generators.CVSCppGenerator import CVS0DCppGenerator
from parsers.PrimitiveParsers import YamlFileParser


def generate_with_new_architecture(do_generation_with_fit_parameters,
                                   inp_data_dict=None):

    yaml_parser = YamlFileParser()
    inp_data_dict = yaml_parser.parse_user_inputs_file(inp_data_dict, obs_path_needed=False, 
                                                       do_generation_with_fit_parameters=do_generation_with_fit_parameters)

    DEBUG = inp_data_dict['DEBUG']
    file_prefix = inp_data_dict['file_prefix']
    resources_dir = inp_data_dict['resources_dir']
    resources_dir = inp_data_dict['resources_dir']
    generated_models_dir = inp_data_dict['generated_models_dir']
    generated_models_subdir = inp_data_dict['generated_models_subdir']
    vessels_csv_abs_path = inp_data_dict['vessels_csv_abs_path']
    parameters_csv_abs_path = inp_data_dict['parameters_csv_abs_path']


    if do_generation_with_fit_parameters:
        param_id_output_dir_abs_path = inp_data_dict['param_id_output_dir_abs_path']
        parser = CSV0DModelParser(vessels_csv_abs_path, parameters_csv_abs_path, 
                                  param_id_output_dir_abs_path)
    else:
        parser = CSV0DModelParser(vessels_csv_abs_path, parameters_csv_abs_path)

    model = parser.load_model()


    if inp_data_dict['model_type'] == 'cellml_only':
        code_generator = CVS0DCellMLGenerator(model, generated_models_subdir, file_prefix,
                                          resources_dir=resources_dir)
        code_generator.generate_files()
    elif inp_data_dict['model_type'] == 'cpp':
        if inp_data_dict['couple_to_1d']:
            code_generator = CVS0DCppGenerator(model, generated_models_subdir, file_prefix,
                                            resources_dir=resources_dir, solver=inp_data_dict['solver'], 
                                            couple_to_1d=inp_data_dict['couple_to_1d'],
                                            cpp_generated_models_dir=inp_data_dict['cpp_generated_models_dir'],
                                            cpp_1d_model_config_path=inp_data_dict['cpp_1d_model_config_path'])
        else:
            code_generator = CVS0DCppGenerator(model, generated_models_subdir, file_prefix,
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
