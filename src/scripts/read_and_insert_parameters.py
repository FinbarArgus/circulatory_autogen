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

def insert_parameters(parameters_path, parameters_to_add_path):
    
    cp = CSVFileParser()
    jp = JSONFileParser()

    if parameters_path.endswith('csv'):
        params_df = cp.get_data_as_dataframe(parameters_path)
    elif parameters_path.endswith('json'):
        params_df = jp.json_to_dataframe(parameters_path)

    print('before')
    print(params_df)
    
    if parameters_to_add_path.endswith('csv'):
        new_params_df = cp.get_data_as_dataframe(parameters_to_add_path)
    elif parameters_to_add_path.endswith('json') :
        new_params_df = jp.json_to_dataframe(parameters_to_add_path)

    # params_df = pd.concat([params_df, new_params_df], ignore_index=True)
    params_df = new_params_df.set_index('variable_name').combine_first(params_df.set_index('variable_name')).reset_index()
    params_df.to_csv(parameters_path, index=None, header=True)
    print('after')
    print(params_df)

if __name__ == '__main__':
    
    try:
        if len(sys.argv) == 2:
            with open(os.path.join(user_inputs_path, 'user_inputs.yaml'), 'r') as file:
                inp_data_dict = yaml.load(file, Loader=yaml.FullLoader)

            parameters_csv_abs_path = os.path.join(resources_dir_path, inp_data_dict['input_param_file'])
            insert_parameters(parameters_csv_abs_path, sys.argv[1])
        elif len(sys.argv) == 3:
            insert_parameters(sys.argv[1], sys.argv[2])

        
    except:
        print(traceback.format_exc())
        print("expected usage: read_and_insert_parameters.py parameters_to_add.json")
        exit()
