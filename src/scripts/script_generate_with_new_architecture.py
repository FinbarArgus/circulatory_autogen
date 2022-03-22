'''
Created on 29/10/2021

@author: Gonzalo D. Maso Talou, Finbar J. Argus
'''

import sys
import os
import re
root_dir_path = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir_path, 'src'))

resources_dir_path = os.path.join(root_dir_path, 'resources')
param_id_dir_path = os.path.join(root_dir_path, 'param_id_output')
output_model_dir_path = os.path.join(root_dir_path, 'generated_models')

from parsers.ModelParsers import CSV0DModelParser
from generators.CVSCellMLGenerator import CVS0DCellMLGenerator
import traceback

if __name__ == '__main__':
    
    try:
        vessels_csv_abs_path = os.path.join(resources_dir_path, sys.argv[1])
        parameters_csv_abs_path = os.path.join(resources_dir_path, sys.argv[2])
        output_files_prefix = sys.argv[3]

        if len(sys.argv) == 4:
            parser = CSV0DModelParser(vessels_csv_abs_path, parameters_csv_abs_path)
        elif len(sys.argv) == 6:
            param_id_obs_path = sys.argv[5]
            data_str_addon = re.sub('\.json', '', os.path.split(param_id_obs_path)[1])
            param_id_dir_abs_path = os.path.join(param_id_dir_path, sys.argv[4]+f'_{sys.argv[3]}_{data_str_addon}')
            parser = CSV0DModelParser(vessels_csv_abs_path, parameters_csv_abs_path, param_id_dir_abs_path)
        else:
            exit()
        model = parser.load_model()
        
        code_generator = CVS0DCellMLGenerator(model, output_model_dir_path, output_files_prefix)
        code_generator.generate_files()
        
    except:
        print(traceback.format_exc())
        print("Usage without id params: script_generate_with_new_architecture.py vessel_file.csv parameter_file.csv "
              "output_files_prefix")
        print("Usage with id params: script_generate_with_new_architecture.py vessel_file.csv parameter_file.csv "
              "output_files_prefix"
              "param_id_method "
              "param_id_obs_data_path")
        print("e.g. script_generate_with_new_architecture.py simple_physiological_vessel_array.csv "
              "parameters_orig.csv "
              "simple_physiological "
              "genetic_algorithm"
              "/path/to/data/json/file.json")