'''
Created on 29/10/2021

@author: Gonzalo D. Maso Talou
'''

import sys

from parsers.ModelParsers import CSV0DModelParser
from generators.CSVCellMLGenerator import CVS0DCellMLGenerator

if __name__ == '__main__':
    
    try:
        if len(sys.argv) == 5:
            parser = CSV0DModelParser(sys.argv[1], sys.argv[2])
        elif len(sys.argv) == 6:
            parser = CSV0DModelParser(sys.argv[1], sys.argv[2], sys.argv[5])
        model = parser.load_model()
        
        code_generator = CVS0DCellMLGenerator(model, sys.argv[3], sys.argv[4])
        code_generator.generate_files()
        
    except Exception as e:
        print(e)
        print("Usage: script_generate_with_new_architecture.py vessel_file.csv parameter_file.csv "
              "output_path output_files_prefix param_id_dir(optional)")
        print("e.g. script_generate_with_new_architecture.py simple_physiological_vessel_array_header.csv "
              "parameters_orig.csv /home/gonzalo/ "
              "test_CVS genetic_algorithm_simple_physiological")