'''
Created on 29/10/2021

@author: Finbar Argus, Gonzalo D. Maso Talou
'''

from parsers.PrimitiveParsers import CSVFileParser
from models.LumpedModels import CVS0DModel

class CSV0DModelParser(object):
    '''
    Creates a 0D model representation from a vessel and a parameter CSV files.
    '''
    def __init__(self, vessel_filename, parameter_filename):
        self.vessel_filename = vessel_filename
        self.parameter_filename = parameter_filename
        self.csv_parser = CSVFileParser()

    def load_model(self):
        vessels_array = self.csv_parser.get_data_as_nparray(self.vessel_filename,True)
        parameters_array = self.csv_parser.get_data_as_nparray(self.parameter_filename,True)
        
        model_0D = CVS0DModel(vessels_array,parameters_array)
        
        return model_0D


