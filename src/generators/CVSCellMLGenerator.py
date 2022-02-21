'''
Created on 29/10/2021

@author: Finbar Argus, Gonzalo D. Maso Talou
'''

import numpy as np
import re
import pandas as pd
import os
from sys import exit
generators_dir_path = os.path.dirname(__file__)


class CVS0DCellMLGenerator(object):
    '''
    Generates CellML files for the 0D model represented in @
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
        self.base_script = os.path.join(generators_dir_path, 'resources/base_script.cellml')
        self.modules_script = os.path.join(generators_dir_path, 'resources/BG_modules.cellml')
        self.heart_modules_script = os.path.join(generators_dir_path, 'resources/heart_modules.cellml')
        self.units_script = os.path.join(generators_dir_path, 'resources/units.cellml')

    def generate_files(self):
        if(type(self.model).__name__ != "CVS0DModel"):
            print("Error: The model should be a CVS0DModel representation")
            return
        
        print("Generating model files at {}".format(self.output_path))
        
        #    Code to generate model files
        self.__generate_CellML_file()
        if self.model.param_id_consts:
            self.__modify_parameters_array_from_param_id()
        self.__generate_parameters_csv()
        self.__generate_parameters_file()
        self.__generate_modules_file()
        self.__generate_units_file()

        # TODO check that model generation is succesful, possibly by calling to opencor
        print('Model generation complete.')
        print('Testing to see if model opens in OpenCOR')
        opencor_available = True
        try:
            import opencor as oc
        except:
            opencor_available = False
            pass
        if opencor_available:
            sim = oc.open_simulation(os.path.join(self.output_path, f'{self.filename_prefix}.cellml'))
            if sim.valid():
                print('Model generation has been successful.')
            else:
                if self.model.all_parameters_defined:
                    print('The OpenCOR model is not yet working, The reason for this is unknown.\n')
                else:
                    print('The OpenCOR model is not yet working because all parameters have not been given values, \n'
                          f'Enter the values in '
                          f'{os.path.join(self.user_resources_path, f"{self.filename_prefix}_parameters_unfinished.csv")}')

        else:
            print('Model generation is complete but OpenCOR could not be opened to test the model. \n'
                  'If you want this check to happen make sure you use the python that is shipped with OpenCOR')


    def __generate_CellML_file(self):
        print("Generating CellML file {}.cellml".format(self.filename_prefix))
        with open(self.base_script, 'r') as rf:
            with open(os.path.join(self.output_path,f'{self.filename_prefix}.cellml'), 'w') as wf:
                for line in rf:
                    if 'import xlink:href="units.cellml"' in line:
                        line = re.sub('units', f'{self.filename_prefix}_units', line)
                    elif 'import xlink:href="parameters_autogen.cellml"' in line:
                        line = re.sub('parameters_autogen', f'{self.filename_prefix}_parameters', line)

                    # copy the start of the basescript until line that says #STARTGENBELOW
                    wf.write(line)
                    if '#STARTGENBELOW' in line:
                        break

                # import vessels
                print('writing imports')
                self.__write_section_break(wf, 'imports')
                self.__write_imports(wf, self.model.vessels_df)
    
                # define mapping between vessels
                print('writing vessel mappings')
                self.__write_section_break(wf, 'vessel mappings')
                self.__write_vessel_mappings(wf, self.model.vessels_df)

                # create computation environment to sum flows from terminals
                # to have a total flow input into each first venous component.
                print('writing environment to sum venous input flows')
                self.__write_section_break(wf, 'terminal venous connection')
                self.__write_terminal_venous_connection_comp(wf, self.model.vessels_df)

                # define variables so they can be accessed
                print('writing variable access')
                self.__write_section_break(wf, 'access_variables')
                self.__write_access_variables(wf, self.model.vessels_df)

                # map between computational environment and module so they can be accessed
                print('writing mappings between computational environment and modules')
                self.__write_section_break(wf, 'vessel mappings')
                self.__write_comp_to_module_mappings(wf, self.model.vessels_df)
    
                # map constants to different modules
                print('writing mappings between constant params')
                self.__write_section_break(wf, 'parameters mapping to modules')
                self.__write_param_mappings(wf, self.model.vessels_df, params_array=self.model.parameters_array)
    
                # map environment time to module times
                print('writing writing time mappings between environment and modules')
                self.__write_section_break(wf, 'time mapping')
                self.__write_time_mappings(wf, self.model.vessels_df)
    
                # Finalise the file
                wf.write('</model>\n')
            
    
    def __generate_parameters_file(self):
        print("Generating CellML file {}_parameters.cellml".format(self.filename_prefix))
        """
        Takes in a data frame of the params and generates the parameter_cellml file
        """

        with open(os.path.join(self.output_path, f'{self.filename_prefix}_parameters.cellml'), 'w') as wf:

            wf.write('<?xml version=\'1.0\' encoding=\'UTF-8\'?>\n')
            wf.write('<model name="Parameters" xmlns="http://www.cellml.org/cellml/1.1#'
                     '" xmlns:cellml="http://www.cellml.org/cellml/1.1#">\n')
    
            heart_params_array = self.model.parameters_array[np.where(self.model.parameters_array["comp_env"] ==
                                                                      'heart')]
            pulmonary_params_array = self.model.parameters_array[np.where(self.model.parameters_array["comp_env"] ==
                                                                          'pulmonary')]
            systemic_params_array = self.model.parameters_array[np.where(self.model.parameters_array["comp_env"] ==
                                                                         'systemic')]

            wf.write('<component name="parameters_heart">\n')
            self.__write_constant_declarations(wf, heart_params_array["variable_name"],
                                        heart_params_array["units"],
                                        heart_params_array["value"])
            wf.write('</component>\n')
            wf.write('<component name="parameters">\n')
            self.__write_constant_declarations(wf, systemic_params_array["variable_name"],
                                        systemic_params_array["units"],
                                        systemic_params_array["value"])
            self.__write_constant_declarations(wf, pulmonary_params_array["variable_name"],
                                               pulmonary_params_array["units"],
                                               pulmonary_params_array["value"])
            wf.write('</component>\n')
            wf.write('</model>\n')

    def __modify_parameters_array_from_param_id(self):
        # first modify param_const names easily by modifying them in the array
        print('modifying constants to values identified from parameter id')
        for const_name, val in self.model.param_id_consts:
            self.model.parameters_array[np.where(self.model.parameters_array['variable_name'] ==
                                           const_name)[0][0]]['value'] = f'{val:.4e}'
            self.model.parameters_array[np.where(self.model.parameters_array['variable_name'] ==
                                           const_name)[0][0]]['data_reference'] = \
                f'{self.model.param_id_date}_identified'

    def __generate_parameters_csv(self):

        # check if all the required parameters have been defined, if not we make an "unfinished"
        # csv file which makes it easy for the user to include the required parameters
        if self.model.all_parameters_defined:
            file_to_create = os.path.join(self.output_path, f'{self.filename_prefix}_parameters.csv')
        else:
            file_to_create = os.path.join(self.user_resources_path,
                                          f'{self.filename_prefix}_parameters_unfinished.csv')
            print(f'\n WARNING \nRequired parameters are missing. \nCreating a file {file_to_create},\n'
                  f'which has EMPTY_MUST_BE_FILLED tags where parameters\n'
                  f'need to be included. The user should include these parameters then remove \n'
                  f'the "_unfinished" ending of the file name, then rerun the model generation \n'
                  f'with the new parameters file as input.\n')
        df = pd.DataFrame(self.model.parameters_array)
        df.to_csv(file_to_create, index=None, header=True)
    
    def __generate_units_file(self):
        # TODO allow a specific units file to be generated
        #  This function simply copies the units file
        print(f'Generating CellML file {self.filename_prefix}_units.cellml')
        with open(self.units_script, 'r') as rf:
            with open(os.path.join(self.output_path, f'{self.filename_prefix}_units.cellml'), 'w') as wf:
                for line in rf:
                    wf.write(line)

    def __generate_modules_file(self):
        if self.model.param_id_states:
            # create list to check if all states get modified to param_id values
            state_modified = [False]*len(self.model.param_id_states) #  whether this state has been found and modified
        print(f'Generating modules file {self.filename_prefix}_modules.cellml')
        with open(os.path.join(self.output_path, f'{self.filename_prefix}_modules.cellml'), 'w') as wf:
            with open(self.modules_script, 'r') as rf:
                # skip last line so enddef; doesn't get written till the end.
                lines = rf.readlines()[:-1]
                for line in lines:
                    wf.write(line)
            with open(self.heart_modules_script, 'r') as rf:
                # skip first two lines
                lines = rf.readlines()[2:]
                for line in lines:
                    if self.model.param_id_states:
                        for idx, (state_name, val) in enumerate(self.model.param_id_states):
                            if state_name in line and 'initial_value' in line:
                                inp_string = f'initial_value="{val:.4e}"'
                                line = re.sub('initial_value=\"\d*\.?\d*e?-?\d*\"', inp_string, line)
                                state_modified[idx] = True
                    wf.write(line)

                    # check if each state was modified
                if self.model.param_id_states:
                    if any(state_modified) == False:
                        false_states = [self.model.param_id_states[JJ][0] for JJ in range(len(state_modified)) if
                                        state_modified[JJ] == False]
                        print(f'The parameter id states {false_states} \n'
                              f'were not found in the cellml script, check the parameter id state names and the '
                              f'base_script.cellml file')
                        exit()

    def __write_section_break(self, wf, text):
        wf.write('<!--&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;' +
                text + '&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;//-->\n')

    def __write_imports(self, wf, vessel_df):
        for vessel_tup in vessel_df.itertuples():
            self.__write_import(wf, vessel_tup)
        # add a zero mapping to heart ivc or svc flow input if only one input is specified
        if "venous_ivc" not in vessel_df.loc[vessel_df["name"] == 'heart'].inp_vessels.values[0] or \
                "venous_svc" not in vessel_df.loc[vessel_df["name"] == 'heart'].inp_vessels.values[0]:
            wf.writelines([f'<import xlink:href="{self.filename_prefix}_modules.cellml">\n',
                           f'    <component component_ref="zero_flow" name="zero_flow_module"/>\n',
                           '</import>\n'])
        if "venous_ivc" not in vessel_df.loc[vessel_df["name"] == 'heart'].inp_vessels.values[0] and \
                "venous_svc" not in vessel_df.loc[vessel_df["name"] == 'heart'].inp_vessels.values[0]:
            print('either venous_ivc, or venous_svc, or both must be inputs to the heart, exiting')
            exit()

    def __write_vessel_mappings(self, wf, vessel_df):
        vessel_df.apply(self.__write_vessel_mapping_for_row, args=(vessel_df, wf), axis=1)

    def __write_vessel_mapping_for_row(self, vessel_row, vessel_df, wf):
        # input and output vessels
        main_vessel = vessel_row["name"]
        main_vessel_BC_type = vessel_row["BC_type"]
        main_vessel_type = vessel_row["vessel_type"]

        for out_vessel_idx, out_vessel in enumerate(vessel_row["out_vessels"]):
            if not (vessel_df["name"] == out_vessel).any():
                print(f'the output vessel of {out_vessel} is not defined')
                exit()
            out_vessel_row = vessel_df.loc[vessel_df["name"] == out_vessel].squeeze()
            out_vessel_BC_type = out_vessel_row["BC_type"]
            out_vessel_type = out_vessel_row["vessel_type"]

            # check that input and output vessels are defined as connection variables
            # for that vessel and they have corresponding BCs
            self.__check_input_output_vessels(vessel_df, main_vessel, out_vessel,
                                              main_vessel_BC_type, out_vessel_BC_type,
                                              main_vessel_type, out_vessel_type)

            # ___ get variables to map ___ #

            # get the entrance port idx for the output vessel
            inp_port_idx = -1
            for II in range(len(out_vessel_row["inp_vessels"])):
                if out_vessel_row["inp_vessels"][II] == main_vessel:
                    inp_port_idx = II
                    break
            if main_vessel_type == 'terminal':
                if inp_port_idx == -1:
                    # TODO For now we allow the first venous sections to not define all input terminals
                    #  but this should be deprecated
                    pass
                elif inp_port_idx == 0:
                    pass
                else:
                    # TODO create a module called venous_multi_input, to take in multiple terminals
                    inp_port_idx = 0

            # TODO this part is kind of hacky, but it works, is there a better way to do the mapping with the
            #  heart module?
            if out_vessel == 'heart':
                if len(out_vessel_row["inp_vessels"]) == 2:
                    # this is the case if there is only one vc and one pulmonary
                    # We map the ivc to a zero flow mapping
                    if inp_port_idx == 0:
                        self.__write_mapping(wf, 'zero_flow_module', 'heart_module', ['v_zero'], ['v_ivc'])
                    # if only two inputs are defined, then that means there is one vc and one pulmonary vein
                    # we increase the inp_port_idx by 1 to ignore the ivc port of the heart. The vc defined in
                    # the vessel array will be connected to the svc entrance port, which is equivalent to connecting
                    # it to the ivc port. the ivc, is the 0'th inp_port_idx, so adding one avoids it.
                    inp_port_idx += 1

            variables_1 = vessel_row["exit_ports"][out_vessel_idx]['variables']
            variables_2 = out_vessel_row["entrance_ports"][inp_port_idx]['variables']

            main_vessel_module = main_vessel + '_module'
            out_vessel_module = out_vessel + '_module'

            self.__write_mapping(wf, main_vessel_module, out_vessel_module, variables_1, variables_2)

    def __write_terminal_venous_connection_comp(self, wf, vessel_df):
        first_venous_names = [] # stores name of venous compartments that take flow from terminals
        for vessel_tup in vessel_df.itertuples():
            # first map variables between connection and the venous sections
            if vessel_tup.vessel_type == 'terminal':
                vessel_name = vessel_tup.name
                out_vessel_name = vessel_tup.out_vessels[0]
                v_1 = 'v_T'
                v_2 = f'v_{vessel_name}'
    
                self.__write_mapping(wf, vessel_name+'_module','terminal_venous_connection',
                              [v_1], [v_2])

            # check that vessel type is venous but and that it is the first venous thats connected to a terminal
            if vessel_tup.vessel_type == 'venous' and \
                    vessel_df.loc[vessel_df['name'].isin(vessel_tup.inp_vessels)
                    ]['vessel_type'].str.contains('terminal').any():
                vessel_name = vessel_tup.name
                first_venous_names.append(vessel_name)
                vessel_BC_type = vessel_tup.BC_type
                v_1 = f'v_{vessel_name}'
                if vessel_BC_type == 'vp':
                    v_2 = 'v_in'
                else:
                    print(f'first venous vessel BC type of {vessel_BC_type} has not'
                          f'been implemented')
    
                self.__write_mapping(wf, 'terminal_venous_connection', vessel_name+'_module',
                              [v_1], [v_2])
    
        # loop through vessels to get the terminals to add up for each first venous section
        terminal_names_for_first_venous = [[] for i in range(len(first_venous_names))]
        for vessel_tup in vessel_df.itertuples():
            if vessel_tup.vessel_type == 'terminal':
                vessel_name = vessel_tup.name
                for idx, venous_name in enumerate(first_venous_names):
                    if vessel_tup.out_vessels[0] == venous_name:
                        if vessel_name not in vessel_df.loc[vessel_df["name"] == venous_name].squeeze()["inp_vessels"]:
                            print(f'venous input of {venous_name} does not include the terminal vessel '
                                  f'{vessel_name} as an inp_vessel in {self.filename_prefix}_vessel_array. '
                                  f'not including terminal names as input has been deprecated')
                            exit()
                        terminal_names_for_first_venous[idx].append(vessel_name)
    
        # create computation environment for connection and write the
        # variable definition and calculation of flow to each first venous module.
        wf.write(f'<component name="terminal_venous_connection">\n')
        variables = []
        units = []
        in_outs = []
        for idx, venous_name in enumerate(first_venous_names):
            for terminal_name in terminal_names_for_first_venous[idx]:
                variables.append(f'v_{terminal_name}')
                units.append('m3_per_s')
                in_outs.append('in')
    
            variables.append(f'v_{venous_name}')
            units.append('m3_per_s')
            in_outs.append('out')
    
        self.__write_variable_declarations(wf, variables, units, in_outs)
        for idx, venous_name in enumerate(first_venous_names):
            rhs_variables = []
            lhs_variable = f'v_{venous_name}'
            for terminal_name in terminal_names_for_first_venous[idx]:
                rhs_variables.append(f'v_{terminal_name}')
    
            self.__write_variable_sum(wf, lhs_variable, rhs_variables)
        wf.write('</component>\n')

    def __write_access_variables(self, wf, vessel_df):
        vessel_df.apply(self.__write_access_variables_for_row, args=(wf,), axis=1)

    def __write_access_variables_for_row(self, vessel_row, wf):
        wf.write(f'<component name="{vessel_row["name"]}">\n')
        lines_to_write = []
        for variable, unit, access_str, _ in vessel_row["variables_and_units"]:
            if access_str == 'access':
                lines_to_write.append(f'   <variable name="{variable}" public_interface="in" units="{unit}"/>\n')
        wf.writelines(lines_to_write)
        wf.write('</component>\n')

    def __write_comp_to_module_mappings(self, wf, vessel_df):
        vessel_df.apply(self.__write_comp_to_module_mappings_for_row, args=(wf,), axis=1)

    def __write_comp_to_module_mappings_for_row(self, vessel_row, wf):
        vessel_name = vessel_row["name"]
        inp_vars = [vessel_row["variables_and_units"][i][0] for i in
                    range(len(vessel_row["variables_and_units"])) if
                    vessel_row["variables_and_units"][i][2] == 'access']
        out_vars = inp_vars

        self.__write_mapping(wf, vessel_name, vessel_name + '_module', inp_vars, out_vars)

    def __write_param_mappings(self, wf, vessel_df, params_array=None):
        vessel_df.apply(self.__write_param_mappings_for_row, args=(wf,), params_array=params_array, axis=1)

    def __write_param_mappings_for_row(self, vessel_row, wf, params_array=None):
        vessel_name = vessel_row["name"]
        module_addon = '_module'
        if vessel_row["vessel_type"].startswith('heart'):
            global_variable_addon = ''
            param_file_name = 'parameters_heart'
        else:
            global_variable_addon = f'_{vessel_name}'
            if vessel_row["vessel_type"] == 'terminal':
                global_variable_addon = re.sub('_T$', '', global_variable_addon)
            param_file_name = 'parameters'

        vars = [vessel_row["variables_and_units"][i][0] for
                       i in range(len(vessel_row["variables_and_units"])) if
                       vessel_row["variables_and_units"][i][3] == 'constant']

        module_vars = [vars[i] + global_variable_addon for i in range(len(vars))]

        global_vars = [vessel_row["variables_and_units"][i][0] for
                       i in range(len(vessel_row["variables_and_units"])) if
                       vessel_row["variables_and_units"][i][3] == 'global_constant']

        # check that the variables are in the parameter array
        if params_array is not None:
            for variable_name in module_vars:
                if variable_name not in params_array["variable_name"]:
                    print(f'variable {variable_name} is not in the parameter '
                          f'dataframe/csv file')
                    exit()
        self.__write_mapping(wf, param_file_name, vessel_name + module_addon,
                             module_vars, vars)

        self.__write_mapping(wf, 'environment', vessel_name + module_addon,
                             global_vars, global_vars)

    def __write_time_mappings(self, wf, vessel_df):
        vessel_df.apply(self.__write_time_mappings_for_row, args=(wf,), axis=1)

    def __write_time_mappings_for_row(self, vessel_row, wf):
        vessel_name = vessel_row["name"]
        module_addon = '_module'

        self.__write_mapping(wf, 'environment', vessel_name + module_addon,
                             ['time'], ['t'])

    def __write_time_mappings_OLD(self, wf, vessel_df):
        for vessel_tup in vessel_df.itertuples():
            # input and output vessels
            vessel_name = vessel_tup.name
            module_addon = '_module'
            self.__write_mapping(wf, 'environment', vessel_name+module_addon,
                          ['time'], ['t'])

    def __write_import(self, wf, vessel_tup):
        module_type = vessel_tup.module_type

        str_addon = '_module'
    
        wf.writelines([f'<import xlink:href="{self.filename_prefix}_modules.cellml">\n',
        f'    <component component_ref="{module_type}" name="{vessel_tup.name+str_addon}"/>\n',
        '</import>\n'])

    def __check_input_output_vessels(self, vessel_df, main_vessel, out_vessel,
                                   main_vessel_BC_type, out_vessel_BC_type,
                                   main_vessel_type, out_vessel_type):
        if not out_vessel:
            print(f'connection vessels incorrectly defined for {main_vessel}')
            exit()
        if main_vessel_type == 'terminal':
            pass
        elif main_vessel not in vessel_df.loc[vessel_df["name"] == out_vessel].inp_vessels.values[0]:
            print(f'"{main_vessel}" and "{out_vessel}" are incorrectly connected, '
                  f'check that they have eachother as output/input')
            exit()
    
        if main_vessel_BC_type.endswith('v'):
            if not out_vessel_BC_type.startswith('p'):
                print(f'"{main_vessel}" output BC is v, the input BC of "{out_vessel}"',
                      ' should be p')
                exit()
        if main_vessel_BC_type.endswith('p'):
            if not out_vessel_BC_type.startswith('v'):
                print(f'"{main_vessel}" output BC is p, the input BC of "{out_vessel}"',
                      ' should be v')
                exit()

    def __write_mapping(self, wf, inp_name, out_name, inp_vars_list, out_vars_list):
        wf.writelines(['<connection>\n',
        f'   <map_components component_1="{inp_name}" component_2="{out_name}"/>\n'])
        for inp_var, out_var in zip(inp_vars_list, out_vars_list):
            if inp_var and out_var:
                wf.write(f'   <map_variables variable_1="{inp_var}" variable_2="{out_var}"/>\n')
        wf.write('</connection>\n')

    def __write_variable_declarations(self, wf, variables, units, in_outs):
        for variable, unit, in_out in zip(variables, units, in_outs):
            wf.write(f'<variable name="{variable}" public_interface="{in_out}" units="{unit}"/>\n')
    
    def __write_constant_declarations(self, wf, variable_names, units, values):
        for variable, unit, value in zip(variable_names, units, values):
            wf.write(f'<variable initial_value="{value}" name="{variable}" '
                     f'public_interface="out" units="{unit}"/>\n')

    def __write_variable_sum(self, wf, lhs_variable, rhs_variables):
        wf.writelines('<math xmlns="http://www.w3.org/1998/Math/MathML">\n'
                      '   <apply>\n'
                      '       <eq/>\n'
                      f'       <ci>{lhs_variable}</ci>\n')
        if len(rhs_variables) > 1:
            wf.write('       <apply>\n')
            wf.write('           <plus/>\n')
            for variable in rhs_variables:
                wf.write(f'            <ci>{variable}</ci>\n')
            wf.write('       </apply>\n')
        else:
            wf.write(f'            <ci>{rhs_variables[0]}</ci>\n')
    
        wf.write('   </apply>\n')
        wf.write('</math>\n')
