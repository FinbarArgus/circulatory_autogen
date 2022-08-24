'''
Created on 29/10/2021

@author: Finbar J. Argus
'''

import sys
import os
import re
import pandas as pd
root_dir_path = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir_path, 'src'))

resources_dir_path = os.path.join(root_dir_path, 'resources')
param_id_dir_path = os.path.join(root_dir_path, 'param_id_output')
output_model_dir_path = os.path.join(root_dir_path, 'generated_models')

from parsers.PrimitiveParsers import JSONFileParser, CSVFileParser
import traceback

if __name__ == '__main__':
    
    try:
        if len(sys.argv) == 3:
            pass
        else:
            exit()

        parameters_csv_abs_path = os.path.join(resources_dir_path, sys.argv[1])
        parameters_to_add_json_path = os.path.join(resources_dir_path, sys.argv[2])

        cp = CSVFileParser()
        jp = JSONFileParser()

        params_df = cp.get_data_as_dataframe(parameters_csv_abs_path)
        new_params_df = jp.json_to_dataframe(parameters_to_add_json_path)

        # TODO replace params if they already exist.
        # params_df = pd.concat([params_df, new_params_df], ignore_index=True)
        params_df = new_params_df.set_index('variable_name').combine_first(params_df.set_index('variable_name')).reset_index()
        params_df.to_csv(parameters_csv_abs_path, index=None, header=True)
        print(params_df)

        
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
