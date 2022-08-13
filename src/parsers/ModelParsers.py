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
        vessels_df = self.csv_parser.get_data_as_dataframe_multistrings(self.vessel_filename, True)

        # TODO remove the below:
        #  Temporarily we add a pulmonary system if there isnt one defined, this should be defined by
        #   the user but we include this to improve backwards compatitibility.

        if len(vessels_df.loc[vessels_df["name"] == 'heart']) == 1:
            if len(vessels_df.loc[vessels_df["name"] == 'heart'].out_vessels.values[0]) < 2:
                # if the heart only has one output we assume it doesn't have an output to a pulmonary artery
                # add pulmonary vein and artery to df
                vessels_df.loc[vessels_df.index.max()+1] = ['par', 'vp', 'arterial_simple', ['heart'], ['pvn']]
                vessels_df.loc[vessels_df.index.max()+1] = ['pvn', 'vp', 'arterial_simple', ['par'], ['heart']]
                # add pulmonary artery (par) to output of heart and pvn to input
                vessels_df.loc[vessels_df["name"] == 'heart'].out_vessels.values[0].append('par')
                vessels_df.loc[vessels_df["name"] == 'heart'].inp_vessels.values[0].append('pvn')
        elif len(vessels_df.loc[vessels_df["name"] == 'heart']) == 0:
            pass
        else:
            print('cannot have more than 2 hearts, we dont model octopii')
            exit()


        # add module info to each row of vessel array
        self.json_parser.append_module_config_info_to_vessel_df(vessels_df, self.module_config_path)

        # TODO change to using a pandas dataframe
        parameters_array_orig = self.csv_parser.get_data_as_nparray(self.parameter_filename, True)
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

        # get the allowable types from the modules_config.json file
        module_df = self.json_parser.json_to_dataframe(self.module_config_path)
        model_0D.possible_vessel_BC_types = list(set(list(zip(module_df["vessel_type"].to_list(), module_df["BC_type"].to_list()))))

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
            if vessel_tup.vessel_type.startswith('heart'):
                str_addon = ''
                module = 'heart'
            elif vessel_tup.vessel_type == 'terminal' or vessel_tup.vessel_type == 'terminal2':
                str_addon = re.sub('_T$', '', f'_{vessel_tup.name}')
                module = 'systemic'
            else:
                str_addon = f'_{vessel_tup.name}'

            # add str_addon to param name from module_config if constant
            required_params += [(vessel_tup.variables_and_units[i][0] + str_addon,
                                 vessel_tup.variables_and_units[i][1],vessel_tup.variables_and_units[i][3])  for
                                   i in range(len(vessel_tup.variables_and_units)) if
                                   vessel_tup.variables_and_units[i][3] in ['constant']]

            # dont add str_addon if global_constant
            required_params += [(vessel_tup.variables_and_units[i][0],
                                 vessel_tup.variables_and_units[i][1],vessel_tup.variables_and_units[i][3])  for
                                i in range(len(vessel_tup.variables_and_units)) if
                                vessel_tup.variables_and_units[i][3] in ['global_constant']]
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

        required_params_unique = []
        for entry in required_params:
            if entry not in required_params_unique:
                required_params_unique.append(entry)
        required_params = np.array(required_params_unique)

        all_parameters_defined = True

        parameters_list = []

        for idx, param_tuple in enumerate(required_params):
            try:
                new_entry = parameters_array_orig[np.where(parameters_array_orig["variable_name"] ==
                                                                       param_tuple[0])][0]
                new_entry = [item for item in new_entry]
                if new_entry[1] != param_tuple[1]:
                    print('units in parameters.csv file does not match with units in module_config.json file'
                          f'for param {new_entry[0]}, exiting')
                    exit()

                new_entry.insert(2, param_tuple[2])
                parameters_list.append(new_entry)
                # overwrite 2 index with local or global, it doesn't matter where the
            except:
                # the other entries apart from name in this row are left empty
                new_entry = ([param_tuple[0], param_tuple[1], param_tuple[2],
                                     'EMPTY_MUST_BE_FILLED', 'EMPTY_MUST_BE_FILLED'])
                parameters_list.append(new_entry)

                all_parameters_defined = False

        if len(set([len(a) for a in parameters_list])) != 1:
            print('parameters rows are of non consistent length, exiting')
            exit()
        parameters_array = np.empty((len(parameters_list)),
                                    dtype=[('variable_name', 'U80'), ('units', 'U80'),('const_type', 'U80'),
                                           ('value', 'U80'), ('data_reference', 'U80')])
        parameters_array['variable_name'] = np.array(parameters_list)[:,0]
        parameters_array['units'] = np.array(parameters_list)[:,1]
        parameters_array['const_type'] = np.array(parameters_list)[:,2]
        parameters_array['value'] = np.array(parameters_list)[:,3]
        parameters_array['data_reference'] = np.array(parameters_list)[:,4]
        return parameters_array, all_parameters_defined





