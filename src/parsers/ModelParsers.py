'''
Created on 29/10/2021

@author: Finbar Argus, Gonzalo D. Maso Talou
'''

from parsers.PrimitiveParsers import CSVFileParser
from models.LumpedModels import CVS0DModel
from checks.LumpedModelChecks import LumpedCompositeCheck, LumpedBCVesselCheck
import numpy as np
import re


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
        parameters_array_orig = self.csv_parser.get_data_as_nparray(self.parameter_filename,True)
        # Reduce parameters_array so that it only includes the required parameters for
        # this vessel_array.
        # This will output True if all the required parameters have been defined and
        # False if they have not.
        parameters_array, all_parameters_defined = self.__reduce_parameters_array(parameters_array_orig, vessels_array)
        # this vessel_array
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
                              param_id_date=param_id_date,
                              all_parameters_defined=all_parameters_defined)
        checker = LumpedCompositeCheck(check_list=[LumpedBCVesselCheck()])
        checker.execute(model_0D)

        return model_0D

    def __reduce_parameters_array(self, parameters_array_orig, vessels_array):
        # TODO get the required params from the BG modules
        required_params = []
        num_params = 0
        # Add pulmonary parameters # TODO put this into the for loop when pulmonary vessels are modules
        required_params.append('C_par')
        required_params.append('C_pvn')
        required_params.append('R_par')
        required_params.append('R_pvn')
        required_params.append('I_par')
        required_params.append('I_pvn')
        num_params += 6
        for vessel in vessels_array:
            vessel_name = vessel["name"]
            if vessel_name.startswith('heart'):
                required_params.append('T')
                required_params.append('t_ac')
                required_params.append('t_ar')
                required_params.append('T_ac')
                required_params.append('T_ar')
                required_params.append('T_vc')
                required_params.append('T_vr')
                required_params.append('CQ_trv')
                required_params.append('CQ_puv')
                required_params.append('CQ_miv')
                required_params.append('CQ_aov')
                required_params.append('E_ra_A')
                required_params.append('E_ra_B')
                required_params.append('E_rv_A')
                required_params.append('E_rv_B')
                required_params.append('E_la_A')
                required_params.append('E_la_B')
                required_params.append('E_lv_A')
                required_params.append('E_lv_B')
                required_params.append('q_ra_0')
                required_params.append('q_rv_0')
                required_params.append('q_la_0')
                required_params.append('q_lv_0')
                num_params += 23
            elif vessel["vessel_type"] in ['arterial', 'split_junction', 'merge_junction', '2in2out_junction']:
                required_params.append(f'r_{vessel_name}')
                required_params.append(f'l_{vessel_name}')
                required_params.append(f'theta_{vessel_name}')
                required_params.append(f'E_{vessel_name}')
                num_params += 4
            elif vessel["vessel_type"] in ['terminal']:
                vessel_name_minus_T = re.sub('_T$', '', vessel_name)
                required_params.append(f'R_T_{vessel_name_minus_T}')
                required_params.append(f'C_T_{vessel_name_minus_T}')
                required_params.append(f'alpha_{vessel_name_minus_T}')
                required_params.append(f'v_nom_{vessel_name_minus_T}')
                num_params += 4
            elif vessel["vessel_type"] in ['venous']:
                required_params.append(f'C_{vessel_name}')
                required_params.append(f'R_{vessel_name}')
                required_params.append(f'I_{vessel_name}')
                num_params += 3
            else:
                print(f'unknown required parameters for vessel_type {vessel["vessel_type"]}, exiting')
        # append global parameters
        required_params.append(f'beta_g')
        required_params.append(f'gain_int')
        num_params += 2

        required_params = np.array(required_params)

        all_parameters_defined = True
        parameters_array = np.empty([num_params,],
                                    dtype=parameters_array_orig.dtype)

        for idx, param in enumerate(required_params):
            try:
                parameters_array[idx] = parameters_array_orig[np.where(parameters_array_orig["variable_name"] == param)]
            except:
                # the other entries apart from name in this row are left empty
                parameters_array[idx][0] = param
                all_parameters_defined = False


        return parameters_array, all_parameters_defined





