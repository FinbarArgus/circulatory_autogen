'''
Created on 29/10/2021

@author: Finbar Argus, Gonzalo D. Maso Talou
'''

from parsers.PrimitiveParsers import CSVFileParser
from models.LumpedModels import CVS0DModel
from checks.LumpedModelChecks import LumpedCompositeCheck, LumpedBCVesselCheck, LumpedIDParamsCheck
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
        if self.parameter_id_dir:
            check_list = [LumpedBCVesselCheck(), LumpedIDParamsCheck()]
        else:
            check_list = [LumpedBCVesselCheck()]

        checker = LumpedCompositeCheck(check_list=check_list)
        checker.execute(model_0D)

        return model_0D

    def __reduce_parameters_array(self, parameters_array_orig, vessels_array):
        # TODO get the required params from the BG modules
        required_params = []
        num_params = 0
        # Add pulmonary parameters # TODO put this into the for loop when pulmonary vessels are modules
        # TODO include units and model_environment in the appended item so they can be included
        required_params.append(['C_par', 'm6_per_J', 'pulmonary'])
        required_params.append(['C_pvn', 'm6_per_J', 'pulmonary'])
        required_params.append(['R_par', 'Js_per_m6', 'pulmonary'])
        required_params.append(['R_pvn', 'Js_per_m6', 'pulmonary'])
        required_params.append(['I_par', 'Js2_per_m6', 'pulmonary'])
        required_params.append(['I_pvn', 'Js2_per_m6', 'pulmonary'])
        num_params += 6
        for vessel in vessels_array:
            vessel_name = vessel["name"]
            if vessel_name.startswith('heart'):
                required_params.append(['T', 'second', 'heart'])
                required_params.append(['t_ac', 'dimensionless', 'heart'])
                required_params.append(['t_ar', 'dimensionless', 'heart'])
                required_params.append(['T_ac', 'dimensionless', 'heart'])
                required_params.append(['T_ar', 'dimensionless', 'heart'])
                required_params.append(['T_vc', 'dimensionless', 'heart'])
                required_params.append(['T_vr', 'dimensionless', 'heart'])
                required_params.append(['CQ_trv', 'UnitValve', 'heart'])
                required_params.append(['CQ_puv', 'UnitValve', 'heart'])
                required_params.append(['CQ_miv', 'UnitValve', 'heart'])
                required_params.append(['CQ_aov', 'UnitValve', 'heart'])
                required_params.append(['E_ra_A', 'J_per_m6', 'heart'])
                required_params.append(['E_ra_B', 'J_per_m6', 'heart'])
                required_params.append(['E_rv_A', 'J_per_m6', 'heart'])
                required_params.append(['E_rv_B', 'J_per_m6', 'heart'])
                required_params.append(['E_la_A', 'J_per_m6', 'heart'])
                required_params.append(['E_la_B', 'J_per_m6', 'heart'])
                required_params.append(['E_lv_A', 'J_per_m6', 'heart'])
                required_params.append(['E_lv_B', 'J_per_m6', 'heart'])
                required_params.append(['q_ra_0', 'm3', 'heart'])
                required_params.append(['q_rv_0', 'm3', 'heart'])
                required_params.append(['q_la_0', 'm3', 'heart'])
                required_params.append(['q_lv_0', 'm3', 'heart'])
                num_params += 23
            elif vessel["vessel_type"] in ['arterial', 'split_junction', 'merge_junction', '2in2out_junction']:
                required_params.append([f'r_{vessel_name}', 'metre', 'systemic'])
                required_params.append([f'l_{vessel_name}', 'metre', 'systemic'])
                required_params.append([f'theta_{vessel_name}', 'dimensionless', 'systemic'])
                required_params.append([f'E_{vessel_name}', 'J_per_m3', 'systemic'])
                num_params += 4
            elif vessel["vessel_type"] in ['terminal']:
                vessel_name_minus_T = re.sub('_T$', '', vessel_name)
                required_params.append([f'R_T_{vessel_name_minus_T}', 'Js_per_m6', 'systemic'])
                required_params.append([f'C_T_{vessel_name_minus_T}', 'm6_per_J', 'systemic'])
                required_params.append([f'alpha_{vessel_name_minus_T}', 'dimensionless', 'systemic'])
                required_params.append([f'v_nom_{vessel_name_minus_T}', 'm3_per_s', 'systemic'])
                num_params += 4
            elif vessel["vessel_type"] in ['venous', 'arterial_simple']:
                required_params.append([f'R_{vessel_name}', 'Js_per_m6', 'systemic'])
                required_params.append([f'C_{vessel_name}', 'm6_per_J', 'systemic'])
                required_params.append([f'I_{vessel_name}', 'Js2_per_m6', 'systemic'])
                num_params += 3
            else:
                print(f'unknown required parameters for vessel_type {vessel["vessel_type"]}, exiting')
        # append global parameters
        required_params.append([f'beta_g', 'dimensionless', 'systemic'])
        required_params.append([f'gain_int', 'dimensionless', 'systemic'])
        num_params += 2

        required_params = np.array(required_params)

        all_parameters_defined = True
        parameters_array = np.empty([num_params, ],
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





