'''
Created on 29/10/2021

@author: Gonzalo D. Maso Talou
'''

import sys

from parsers.ModelParsers import CSV0DModelParser
from generators.CSVCellMLGenerator import CVS0DCellMLGenerator

if __name__ == '__main__':
    
    try:    
        parser = CSV0DModelParser(sys.argv[1], sys.argv[2])
        model = parser.load_model()
        
        code_generator = CVS0DCellMLGenerator(model, sys.argv[3], sys.argv[4])
        code_generator.generate_files()
    except:
        print("Usage: script_generate_with_new_architecture.py vessel_file.csv parameter_file.csv output_path output_files_prefix")
        print("e.g. script_generate_with_new_architecture.py simple_physiological_vessel_array_header.csv parameters_orig.csv /home/gonzalo/ test_CVS")        