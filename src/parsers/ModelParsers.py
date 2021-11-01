'''
Created on 29/10/2021

@author: Finbar Argus, Gonzalo D. Maso Talou
'''

from parsers.PrimitiveParsers import CSVFileParser
from models.LumpedModels import CVS0DModel
from checks.LumpedModelChecks import LumpedChecks


class CSV0DModelParser(object):
    '''
    Creates a 0D model representation from a vessel and a parameter CSV files.
    '''
    def __init__(self, vessel_filename, parameter_filename, parameter_id_dir=None):
        self.vessel_filename = vessel_filename
        self.parameter_filename = parameter_filename
        self.parameter_id_dir = parameter_id_dir
        self.csv_parser = CSVFileParser()

    def load_model(self):
        # TODO if file ending is csv. elif file ending is json
        # TODO create a json_parser
        vessels_array = self.csv_parser.get_data_as_nparray(self.vessel_filename,True)
        parameters_array = self.csv_parser.get_data_as_nparray(self.parameter_filename,True)
        if self.parameter_id_dir:
            param_id_states, param_id_consts, param_id_date = self.csv_parser.get_param_id_params_as_lists_of_tuples(
                self.parameter_id_dir)
        else:
            param_id_states = None
            param_id_consts = None
            param_id_date= None

        model_0D = CVS0DModel(vessels_array,parameters_array,
                              param_id_states=param_id_states,
                              param_id_consts=param_id_consts,
                              param_id_date=param_id_date)
        checker = LumpedChecks()
        checker.check_all(model_0D)

        return model_0D






