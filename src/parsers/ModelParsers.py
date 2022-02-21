'''
Created on 29/10/2021

@author: Finbar Argus, Gonzalo D. Maso Talou
'''


from parsers.PrimitiveParsers import CSVFileParser, JSONFileParser
from models.LumpedModels import CVS0DModel
from checks.LumpedModelChecks import LumpedCompositeCheck, LumpedBCVesselCheck, LumpedIDParamsCheck
import numpy as np
import re
import os

generator_resources_dir_path = os.path.join(os.path.dirname(__file__), '../generators/resources')

class CSV0DModelParser(object):
    '''
    Creates a 0D model representation from a vessel and a parameter CSV files.
    '''
    def __init__(self, vessel_filename, parameter_filename, parameter_id_dir=None):
        self.vessel_filename = vessel_filename
        self.parameter_filename = parameter_filename
        self.parameter_id_dir = parameter_id_dir
        self.module_config_path = os.path.join(generator_resources_dir_path, 'module_config.json')
        self.csv_parser = CSVFileParser()
        self.json_parser = JSONFileParser()

    def load_model(self):
        # TODO if file ending is csv. elif file ending is json
        # TODO create a json_parser
        vessels_df = self.csv_parser.get_data_as_dataframe_multistrings(self.vessel_filename,True)

        # TODO remove the below:
        #  Temporarily we add a pulmonary system if there isnt one defined, this should be defined by
        #   the user but we include this to improve backwards compatitibility.
        if len(vessels_df.loc[vessels_df["name"] == 'heart'].out_vessels.values[0]) < 2:
            # if the heart only has one output we assume it doesn't have an output to a pulmonary artery
            # add pulmonary vein and artey to df
            vessels_df.loc[vessels_df.index.max()+1] = ['par', 'vp', 'arterial_simple', ['heart'], ['pvn']]
            vessels_df.loc[vessels_df.index.max()+1] = ['pvn', 'vp', 'arterial_simple', ['par'], ['heart']]
            # add pulmonary artery (par) to output of heart and pvn to input
            vessels_df.loc[vessels_df["name"] == 'heart'].out_vessels.values[0].append('par')
            vessels_df.loc[vessels_df["name"] == 'heart'].inp_vessels.values[0].append('pvn')

        # add module info to each row of vessel array
        self.json_parser.append_module_config_info_to_vessel_df(vessels_df, self.module_config_path)

        parameters_array_orig = self.csv_parser.get_data_as_nparray(self.parameter_filename,True)
        # Reduce parameters_array so that it only includes the required parameters for
        # this vessel_array.
        # This will output True if all the required parameters have been defined and
        # False if they have not.
        # TODO change reduce_parameters_array to be automatic with respect to the modules
        parameters_array, all_parameters_defined = self.__reduce_parameters_array(parameters_array_orig, vessels_df)
        # this vessel_array
        if self.parameter_id_dir:
            param_id_states, param_id_consts, param_id_date = self.csv_parser.get_param_id_params_as_lists_of_tuples(
                self.parameter_id_dir)
        else:
            param_id_states = None
            param_id_consts = None
            param_id_date= None

        model_0D = CVS0DModel(vessels_df,parameters_array,
                              param_id_states=param_id_states,
                              param_id_consts=param_id_consts,
                              param_id_date=param_id_date,
                              all_parameters_defined=all_parameters_defined)
        if self.parameter_id_dir:
            check_list = [LumpedBCVesselCheck(), LumpedIDParamsCheck()]
        else:
            check_list = [LumpedBCVesselCheck()]

        checker = LumpedCompositeCheck(check_list=check_list)
        checker.execute(model_0D)

        return model_0D

    def __reduce_parameters_array(self, parameters_array_orig, vessels_df):
        # TODO get the required params from the BG modules
        #  maybe have a config file for each module that specifies the cellml model and the required constants
        required_params = []
        num_params = 0
        # Add pulmonary parameters # TODO put this into the for loop when pulmonary vessels are modules
        # TODO include units and model_environment in the appended item so they can be included
        for vessel_tup in vessels_df.itertuples():
            if vessel_tup.vessel_type == 'heart':
                str_addon = ''
                module = 'heart'
            elif vessel_tup.vessel_type == 'terminal':
                str_addon = re.sub('_T', '', f'_{vessel_tup.name}')
                module = 'systemic'
            else:
                str_addon = f'_{vessel_tup.name}'
                module = 'systemic'

            required_params += [(vessel_tup.variables_and_units[i][0] + str_addon,
                                 vessel_tup.variables_and_units[i][1], module)  for
                                   i in range(len(vessel_tup.variables_and_units)) if
                                   vessel_tup.variables_and_units[i][3] == 'constant']
            # new_global_params = [(vessel_tup.variables_and_units[i][0],
            #                      vessel_tup.variables_and_units[i][1], module)  for
            #                      i in range(len(vessel_tup.variables_and_units)) if
            #                      vessel_tup.variables_and_units[i][3] == 'global_constant' and
            #                      vessel_tup.variables_and_units[i][0] not in required_params]
            # required_params += new_global_params
        # The below params are no longer in the params files.
        # # append global parameters
        # required_params.append([f'beta_g', 'dimensionless', 'systemic'])
        # required_params.append([f'gain_int', 'dimensionless', 'systemic'])
        # num_params += 2

        required_params = np.array(required_params)

        all_parameters_defined = True
        parameters_array = np.empty([len(required_params), ],
                                    dtype=parameters_array_orig.dtype)

        for idx, param_tuple in enumerate(required_params):
            try:
                parameters_array[idx] = parameters_array_orig[np.where(parameters_array_orig["variable_name"] ==
                                                                       param_tuple[0])]
            except:
                # the other entries apart from name in this row are left empty
                parameters_array[idx][0] = param_tuple[0]
                parameters_array[idx][1] = param_tuple[1]
                parameters_array[idx][2] = param_tuple[2]
                parameters_array[idx][3] = 'EMPTY_MUST_BE_FILLED'
                parameters_array[idx][4] = 'EMPTY_MUST_BE_FILLED'

                all_parameters_defined = False


        return parameters_array, all_parameters_defined





