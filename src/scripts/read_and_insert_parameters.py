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
user_inputs_path = os.path.join(root_dir_path, 'user_run_files')

from parsers.PrimitiveParsers import JSONFileParser, CSVFileParser
import traceback
import yaml

if __name__ == '__main__':
    
    try:

        with open(os.path.join(user_inputs_path, 'user_inputs.yaml'), 'r') as file:
            inp_data_dict = yaml.load(file, Loader=yaml.FullLoader)

        parameters_csv_abs_path = os.path.join(resources_dir_path, inp_data_dict['input_param_file'])
        parameters_to_add_json_path = os.path.join(resources_dir_path, sys.argv[1])

        cp = CSVFileParser()
        jp = JSONFileParser()

        params_df = cp.get_data_as_dataframe(parameters_csv_abs_path)
        new_params_df = jp.json_to_dataframe(parameters_to_add_json_path)

        # params_df = pd.concat([params_df, new_params_df], ignore_index=True)
        params_df = new_params_df.set_index('variable_name').combine_first(params_df.set_index('variable_name')).reset_index()
        params_df.to_csv(parameters_csv_abs_path, index=None, header=True)
        print(params_df)

        
    except:
        print(traceback.format_exc())
        print("expected usage: read_and_insert_parameters.py parameters_to_add.json")
        exit()
