'''
Created on 23/05/2023

@author: Finbar Argus
'''

import numpy as np
import re
import pandas as pd
import os
from sys import exit
generators_dir_path = os.path.dirname(__file__)
from generators.CVSCellMLGenerator import CVS0DCellMLGenerator
import cellml
from libcellml import Annotator


class CVS0DCppGenerator(object):
    '''
    Generates Cpp files from CellML files and annotations for a 0D model
    '''


    def __init__(self, model, generated_model_subdir, filename_prefix, resources_dir=None):
        '''
        Constructor
        '''
        self.model = model
        self.generated_model_subdir = generated_model_subdir
        if not os.path.exists(self.generated_model_subdir):
            os.mkdir(self.generated_model_subdir)
        self.filename_prefix = filename_prefix
        self.file_name_with_ids = f'{file_name}-with-ids'
        if resources_dir is None:
            self.resources_dir = os.path.join(generators_dir_path, '../../resources')
        else:
            self.resources_dir = resources_dir 
        self.generated_model_file_path = os.path.join(self.generated_model_subdir, 
                                                      self.filename_prefix, '.cellml')
        self.annotated_model_file_path = os.path.join(self.generated_model_subdir, 
                                                      self.filename_prefix_with_ids, '.cellml')


    def annotate_cellml(self):
        cellml_model = cellml.parse_model(self.generated_model_file_path, False)
        annotator = Annotator()
        annotator.setModel(cellml_model)
        model_string = cellml.print_model(model)
        print(model_string)



    
    def generate_files(self):
        print("generating CellML files, before Cpp generation")
        cellml_generator = CVS0DCellMLGenerator(self.model, self.generated_model_subdir, self.filename_prefix)
        cellml_generator.generate_files()

        print("now generating Cpp files")

class CVS1DCppGenerator(object):
    '''
    Generates Cpp files for 1D model. This is a wrapper around Lucas Muller's code.

    WARNING: THIS CPP 1D CODE IS CURRENTLY NOT OPEN SOURCE, SO THIS CLASS WON'T WORK UNTIL IT IS MADE OPEN SOURCE.
    '''


    def __init__(self, model, output_path, filename_prefix):
        '''
        Constructor
        '''
        self.model = model
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        self.filename_prefix = filename_prefix
        self.user_resources_path = os.path.join(generators_dir_path, '../../resources')

    
    def generate_files(self):
        print("generating 1D Cpp files")

class CVSCoupledCppGenerator(object):
    '''
    Generates Cpp files for coupled 0D and 1D models.
    '''
    
    def __init__(self, model, output_path, filename_prefix):
        '''
        Constructor
        '''
        self.model = model
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        self.filename_prefix = filename_prefix
        self.user_resources_path = os.path.join(generators_dir_path, '../../resources')
    
    def generate_files(self):
        zeroD_generator = CVS0DCppGenerator(self.model, self.output_path, self.filename_prefix)
        zeroD_generator.generate_files()

        # currently we don't generate the 1D model, we assume it is already generated.
        # oneD_generator = CVS0DCppGenerator(self.model, self.output_path, self.filename_prefix)
        # oneD_generator.generate_files()
        print("now do the coupling")

