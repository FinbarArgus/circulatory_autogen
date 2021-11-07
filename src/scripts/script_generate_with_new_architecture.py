'''
Created on 29/10/2021

@author: Gonzalo D. Maso Talou, Finbar J. Argus
'''

import sys
import os
root_dir_path = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir_path, 'src'))

resources_dir_path = os.path.join(root_dir_path, 'resources')
param_id_dir_path = os.path.join(root_dir_path, 'param_id_output')

from parsers.ModelParsers import CSV0DModelParser
from generators.CSVCellMLGenerator import CVS0DCellMLGenerator
import traceback

if __name__ == '__main__':
    
    try:
        vessels_csv_abs_path = os.path.join(resources_dir_path, sys.argv[1])
        parameters_csv_abs_path = os.path.join(resources_dir_path, sys.argv[2])
        output_abs_path = sys.argv[3]
        output_files_prefix = sys.argv[4]

        if len(sys.argv) == 5:
            parser = CSV0DModelParser(vessels_csv_abs_path, parameters_csv_abs_path)
        elif len(sys.argv) == 6:
            param_id_dir_abs_path = os.path.join(param_id_dir_path, sys.argv[5])
            parser = CSV0DModelParser(vessels_csv_abs_path, parameters_csv_abs_path, param_id_dir_abs_path)
        model = parser.load_model()
        
        code_generator = CVS0DCellMLGenerator(model, output_abs_path, output_files_prefix)
        code_generator.generate_files()
        
    except:
        print(traceback.format_exc())
        print("Usage: script_generate_with_new_architecture.py vessel_file.csv parameter_file.csv "
              "output_path output_files_prefix param_id_dir(optional)")
        print("e.g. script_generate_with_new_architecture.py simple_physiological_vessel_array_header.csv "
              "parameters_orig.csv /home/gonzalo/ "
              "test_CVS genetic_algorithm_simple_physiological")