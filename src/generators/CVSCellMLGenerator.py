'''
Created on 29/10/2021

@author: Finbar Argus, Gonzalo D. Maso Talou
'''

import numpy as np
import re
import pandas as pd
import os
from sys import exit
generators_dir = os.path.dirname(__file__)
base_dir = os.path.join(os.path.dirname(__file__), '../..')
LIBCELLML_available = True
try:
    from libcellml import Annotator, Analyser, AnalyserModel, AnalyserExternalVariable, Generator, GeneratorProfile        
    import utilities.libcellml_helper_funcs as cellml
    import utilities.libcellml_utilities as libcellml_utils
except ImportError as e:
    print("Error -> ", e)
    print('continuing without LibCellML, Warning code checks will not be available.'
          'You will need to open generated models in OpenCOR to check for errors.')
    LIBCELLML_available = False



class CVS0DCellMLGenerator(object):
    '''
    Generates CellML files for the 0D model represented in @
    '''


    def __init__(self, model, output_dir, filename_prefix, resources_dir=None):
        '''
        Constructor
        '''
        self.model = model
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.filename_prefix = filename_prefix
        if resources_dir is None:
            self.resources_dir = os.path.join(generators_dir, '../../resources')
        else:
            self.resources_dir = resources_dir 
        self.base_script = os.path.join(generators_dir, 'resources/base_script.cellml')
        self.module_scripts = [os.path.join(generators_dir, 'resources', filename) for filename in
                               os.listdir(os.path.join(generators_dir, 'resources'))
                               if filename.endswith('modules.cellml')]
        self.module_scripts += [os.path.join(base_dir, 'module_config_user', filename) for filename in
                               os.listdir(os.path.join(base_dir, 'module_config_user'))
                               if filename.endswith('modules.cellml')]
        self.units_scripts = [os.path.join(generators_dir, 'resources/units.cellml'),
                              os.path.join(base_dir, 'module_config_user/user_units.cellml')]
        self.all_parameters_defined = False
        self.BC_set = {}
        self.all_units = []
        
        # this is a temporary hack to include zero flow ivc if only one input to heart TODO make more robust
        self.ivc_connection_done = 0

    def generate_files(self):
        if(type(self.model).__name__ != "CVS0DModel"):
            print("Error: The model should be a CVS0DModel representation")
            return
        
        print("Generating model files at {}".format(self.output_dir))
        
        #    Code to generate model files
        self.__generate_units_file()
        self.__generate_CellML_file()
        if self.model.param_id_name_and_vals:
            self.__modify_parameters_array_from_param_id()
        self.__generate_parameters_csv()
        self.__generate_parameters_file()
        self.__generate_modules_file()

        # TODO check that model generation is succesful, possibly by calling to opencor
        print('Model generation complete.')
        print('Checking Status of Model')

        if LIBCELLML_available:
            # parse the model in non-strict mode to allow non CellML 2.0 models
            model = cellml.parse_model(os.path.join(self.output_dir, self.filename_prefix + '.cellml'), False)
            # resolve imports, in non-strict mode
            importer = cellml.resolve_imports(model, self.output_dir, False)
            # need a flattened model for analysing
            flat_model = cellml.flatten_model(model, importer)
            model_string = cellml.print_model(flat_model)
            
            # this if we want to create the flat model, for debugging
            with open(os.path.join(self.output_dir, self.filename_prefix + '_flat.cellml'), 'w') as f:
                f.write(model_string)

            # analyse the model
            a = Analyser()

            a.analyseModel(flat_model)
            analysed_model = a.model()

            libcellml_utils.print_issues(a)
            print(analysed_model.type())
            if analysed_model.type() != AnalyserModel.Type.ODE:
                print("WARNING model is has some issues, see above. "
                    "The model might still run with some of the above issues"
                    "but it is recommended to fix them")
        
        print('Testing to see if model opens in OpenCOR')
        opencor_available = True
        try:
            import opencor as oc
        except:
            opencor_available = False
            pass
        if opencor_available:
            sim = oc.open_simulation(os.path.join(self.output_dir, f'{self.filename_prefix}.cellml'))
            if sim.valid():
                print('Model generation has been successful.')
            else:
                if self.all_parameters_defined:
                    print('The OpenCOR model is not yet working, The reason for this is unknown. \n'
                          'Open the model in OpenCor and check the error in the simulation environment \n'
                          'for further error info. \n')
                else:
                    print('The OpenCOR model is not yet working because all parameters have not been given values, \n'
                          f'Enter the values in '
                          f'{os.path.join(self.resources_dir, f"{self.filename_prefix}_parameters_unfinished.csv")}')

        else:
            print('Model generation is complete but OpenCOR could not be opened to test the model. \n'
                  'If you want this check to happen make sure you use the python that is shipped with OpenCOR')


    def __generate_CellML_file(self):
        print("Generating CellML file {}.cellml".format(self.filename_prefix))
        with open(self.base_script, 'r') as rf:
            with open(os.path.join(self.output_dir,f'{self.filename_prefix}.cellml'), 'w') as wf:
                for line in rf:
                    if 'import xlink:href="units.cellml"' in line:
                        line = re.sub('units', f'{self.filename_prefix}_units', line)
                    elif 'import xlink:href="parameters_autogen.cellml"' in line:
                        line = re.sub('parameters_autogen', f'{self.filename_prefix}_parameters', line)

                    # copy the start of the basescript until line that says #STARTGENBELOW
                    wf.write(line)
                    if '#STARTGENBELOW' in line:
                        break

                # write units mapping
                print('writing units mapping')
                self.__write_section_break(wf, 'units')
                self.__write_units(wf)

                # import vessels
                print('writing imports')
                self.__write_section_break(wf, 'imports')
                self.__write_imports(wf, self.model.vessels_df)
    
                # define mapping between modules
                print('writing vessel mappings')
                self.__write_section_break(wf, 'vessel mappings')
                self.__write_module_mappings(wf, self.model.vessels_df)

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
                self.__write_section_break(wf, 'own vessel mappings')
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

        with open(os.path.join(self.output_dir, f'{self.filename_prefix}_parameters.cellml'), 'w') as wf:

            wf.write('<?xml version=\'1.0\' encoding=\'UTF-8\'?>\n')
            wf.write('<model name="Parameters" xmlns="http://www.cellml.org/cellml/1.1#'
                     '" xmlns:cellml="http://www.cellml.org/cellml/1.1#" xmlns:xlink="http://www.w3.org/1999/xlink">\n')

            self.__write_section_break(wf, 'parameter_units')
            self.__write_units(wf)
    
            global_params_array = self.model.parameters_array[np.where(self.model.parameters_array["const_type"] ==
                                                                      'global_constant')]
            module_params_array = self.model.parameters_array[np.where((self.model.parameters_array["const_type"] ==
                                                                         'constant') | (self.model.parameters_array["const_type"] == 
                                                                         'boundary_condition'))]

            wf.write('<component name="parameters_global">\n')
            self.__write_constant_declarations(wf, global_params_array["variable_name"],
                                        global_params_array["units"],
                                        global_params_array["value"])
            wf.write('</component>\n')
            wf.write('<component name="parameters">\n')
            self.__write_constant_declarations(wf, module_params_array["variable_name"],
                                        module_params_array["units"],
                                        module_params_array["value"])
            wf.write('</component>\n')
            wf.write('</model>\n')

    def __modify_parameters_array_from_param_id(self):
        # first modify param_const names easily by modifying them in the array
        print('modifying constants to values identified from parameter id')
        for const_name, val in self.model.param_id_name_and_vals:
            self.model.parameters_array[np.where(self.model.parameters_array['variable_name'] ==
                                           const_name)[0][0]]['value'] = f'{val:.10e}'
            self.model.parameters_array[np.where(self.model.parameters_array['variable_name'] ==
                                           const_name)[0][0]]['data_reference'] = \
                f'{self.model.param_id_date}_identified'

    def __generate_parameters_csv(self):

        # check if all the required parameters have been defined, if not we make an "unfinished"
        # csv file which makes it easy for the user to include the required parameters
        if self.all_parameters_defined:
            file_to_create = os.path.join(self.output_dir, f'{self.filename_prefix}_parameters.csv')
        else:
            file_to_create = os.path.join(self.resources_dir,
                                          f'{self.filename_prefix}_parameters_unfinished.csv')
            print(f'\n WARNING \nRequired parameters are missing. \nCreating a file {file_to_create},\n'
                  f'which has EMPTY_MUST_BE_FILLED tags where parameters\n'
                  f'need to be included. The user should include these parameters then remove \n'
                  f'the "_unfinished" ending of the file name, then rerun the model generation \n'
                  f'with the new parameters file as input.\n')
        df = pd.DataFrame(self.model.parameters_array)
        df = df.drop(columns=["const_type"])
        df.to_csv(file_to_create, index=None, header=True)
    
    def __generate_units_file(self):
        # TODO allow a specific units file to be generated
        #  This function simply copies the units file
        print(f'Generating CellML file {self.filename_prefix}_units.cellml')
        with open(os.path.join(self.output_dir, f'{self.filename_prefix}_units.cellml'), 'w') as wf:
            for file_idx, units_script in enumerate(self.units_scripts):
                with open(units_script, 'r') as rf:
                    for line_idx, line in enumerate(rf):
                        # if not first file then skip first two lines and last line
                        if file_idx == 0:
                            if line.startswith('</model>'):
                                # don't write the last line
                                continue
                        elif file_idx == len(self.units_scripts) - 1:
                            if line_idx == 0 or line_idx == 1:
                                # don't write the first two lines
                                continue
                        else:
                            if line_idx == 0 or line_idx == 1:
                                # don't write the first two lines
                                continue
                            if line.startswith('</model>'):
                                # don't write the last line
                                continue

                        wf.write(line)
                        if "units name" in line:
                            self.all_units.append(re.search('units name="(.*?)"', line).group(1))

    def __generate_modules_file(self):
        print(f'Generating modules file {self.filename_prefix}_modules.cellml')
        with open(os.path.join(self.output_dir, f'{self.filename_prefix}_modules.cellml'), 'w') as wf:
            # write first two lines
            wf.write("<?xml version='1.0' encoding='UTF-8'?>\n")
            wf.write("<model name=\"modules\" xmlns=\"http://www.cellml.org/cellml/1.1#\" xmlns:cellml=\"http://www.cellml.org/cellml/1.1#\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n")
            # also write units in modules file
            self.__write_section_break(wf, 'module_units')
            self.__write_units(wf)

            for II, module_file_path in enumerate(self.module_scripts):
                with open(module_file_path, 'r') as rf:
                    # skip first lines that are either intro lines written above or comments.
                    lines = rf.readlines()
                    if not lines:
                        continue

                    count = 0
                    while True:
                        if "component" in lines[count]:
                            break
                        else:
                            count += 1
                            if count > 50:
                                print(f'Error in {module_file_path}, no component found')
                                exit()

                    # skip last line so enddef; doesn't get written till the end.
                    # skip first non-component lines
                    lines = lines[count:-1]

                    for line in lines:
                        if "<component name" in line:
                            # check the name of the module we are in
                            module_type = re.search('name="(.*?)"', line).group(1)
                        if module_type in self.model.vessels_df.module_type.values or module_type == 'zero_flow':
                            wf.write(line)
            wf.write('</model>\n')
                

    def __write_section_break(self, wf, text):
        wf.write('<!--&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;' +
                text + '&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;//-->\n')

    def __write_units(self, wf):

        wf.writelines(f'<import xlink:href="{self.filename_prefix}_units.cellml">\n')
        for unit in self.all_units:
            wf.writelines(f'    <units name="{unit}" units_ref="{unit}"/>\n')
        # units_list = []

        # for vessel_tup in vessel_df.itertuples():
        #     for variable in vessel_tup.variables_and_units:
        #         if variable[1] not in units_list:
        #             wf.writelines(f'    <units name="{variable[1]}" units_ref="{variable[1]}"/>\n')
        #             units_list.append(variable[1])
        
        # for param_tup in parameters_array:
        #     if param_tup[1] not in units_list:
        #         wf.writelines(f'    <units name="{param_tup[1]}" units_ref="{param_tup[1]}"/>\n')
        #         units_list.append(param_tup[1])
        
        wf.writelines('</import>\n')
        
        
    def __write_imports(self, wf, vessel_df):
        for vessel_tup in vessel_df.itertuples():
            if vessel_tup.module_format != 'cellml':
                # if not cellml then don't do anything for this vessel/module
                continue
            self.__write_import(wf, vessel_tup)

        # TODO change the below to vessel_type, not "name"
        if len(vessel_df.loc[vessel_df["name"] == 'heart']) == 1:
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
        elif len(vessel_df.loc[vessel_df["name"] == 'heart']) < 1:
            pass
        elif len(vessel_df.loc[vessel_df["name"] == 'heart']) > 1:
            print('you have declared more that one heart module, exiting')
            exit()

    def __write_module_mappings(self, wf, module_df):
        """This function maps between ports of the modules in a module dataframe."""
        # set connected to false for all entrance ports TODO this might be a better way to check connected for exit ports too
        entrance_ports_connected = {}
        for module_row_idx in range(len(module_df)):
            if module_df.iloc[module_row_idx]["module_format"] != 'cellml':
                # if not cellml then don't do anything for this vessel/module
                continue
            entrance_ports_connected[module_df.iloc[module_row_idx]["name"]] = []
            for II in range(len(module_df.iloc[module_row_idx]["entrance_ports"])):
                # module_df.iloc[module_row_idx]["entrance_ports"][II]["connected"] = False
                entrance_ports_connected[module_df.iloc[module_row_idx]["name"]].append(False)

        # set BC_set to false for all boundary_condition variables
        for module_row_idx in range(len(module_df)):
            if module_df.iloc[module_row_idx]["module_format"] != 'cellml':
                # if not cellml then don't do anything for this vessel/module
                continue
            self.BC_set[module_df.iloc[module_row_idx]["name"]] = {}
            for II in range(len(module_df.iloc[module_row_idx]["variables_and_units"])):
                if module_df.iloc[module_row_idx]["variables_and_units"][II][3] == 'boundary_condition':
                    self.BC_set[module_df.iloc[module_row_idx]["name"]] \
                        [module_df.iloc[module_row_idx]["variables_and_units"][II][0]] = False

        # module_df.apply(self.__write_module_mapping_for_row, args=(module_df, wf), axis=1)
        # The above line is much faster but I'm worried about memory access of entrance_ports_connected
        for module_row_idx in range(len(module_df)):
            if module_df.iloc[module_row_idx]["module_format"] != 'cellml':
                # if not cellml then don't do anything for this vessel/module
                continue
            self.__write_module_mapping_for_row(module_df.iloc[module_row_idx], module_df, 
                                                entrance_ports_connected, wf) # entrance_ports_connected is a modified
                                                                              # in this function 
                            
        # check whether the BC variables have been set with a matched module, if so, remove them from 
        # parameters array and if not, set them to constant
        for module_row_idx in range(len(module_df)):
            if module_df.iloc[module_row_idx]["module_format"] != 'cellml':
                # if not cellml then don't do anything for this vessel/module
                continue
            indexes_to_remove = []
            for II in range(len(module_df.iloc[module_row_idx]["variables_and_units"])):
                if module_df.iloc[module_row_idx]["variables_and_units"][II][3] == 'boundary_condition':
                    full_variable_name = module_df.iloc[module_row_idx]["variables_and_units"][II][0] + \
                        '_' + module_df.iloc[module_row_idx]["name"]
                    if self.BC_set[module_df.iloc[module_row_idx]["name"]] \
                        [module_df.iloc[module_row_idx]["variables_and_units"][II][0]] == False:
                        module_df.iloc[module_row_idx]["variables_and_units"][II][3] = 'constant'
                        # self.model.parameters_array[np.where(self.model.parameters_array["variable_name"] == 
                        #                                      full_variable_name)][0][2] = 'constant'
                    else:
                        self.model.parameters_array = np.delete(self.model.parameters_array, np.where(self.model.parameters_array["variable_name"] == 
                                                             full_variable_name))
                        indexes_to_remove.append(II)
                        
            # remove the BC variables from the variables_and_units list
            # these variables will be accesible from the neighbouring
            # modules where they are calculated.
            for index in sorted(indexes_to_remove, reverse=True):
                del module_df.iloc[module_row_idx]["variables_and_units"][index]

        if np.any(self.model.parameters_array["value"] == 'EMPTY_MUST_BE_FILLED'):
            self.all_parameters_defined = False
        else:
            self.all_parameters_defined = True

    def __write_module_mapping_for_row(self, module_row, module_df, entrance_ports_connected, wf):
        """This function maps between ports of the modules in a one row of a module dataframe."""

        # input and output modules
        main_module = module_row["name"]
        main_module_BC_type = module_row["BC_type"]
        main_module_type = module_row["module_type"]
        

        # create a list of dicts that stores info for the exit ports of this main module
        # entrance_port_types is the corresponding connection check for entrance ports and is stored in 
        # this object so it can be obtained for each main module that connects to that out module
        port_types = []
        for out_port_idx in range(len(module_row["exit_ports"])):
            port_type = module_row["exit_ports"][out_port_idx]["port_type"]
            if port_type in [port_types[II]["port_type"] for II in range(len(port_types))]:
                port_type_idx = [port_types[II]["port_type"] for II
                                 in range(len(port_types))].index(port_type)
                port_types[port_type_idx]["out_port_idxs"].append(out_port_idx)
                port_types[port_type_idx]["port_type_count"] += 1
                port_types[port_type_idx]["connected"].append(False)

            else:
                port_types.append({"port_type": module_row["exit_ports"][out_port_idx]["port_type"],
                                   "out_port_idxs": [out_port_idx],
                                   "port_type_count": 1,
                                   "connected": [False]}) # whether this exit port has been connected
        # check inp vessels are all connected # TODO this should be done in the model parsing stage
        for inp_module_idx, inp_module in enumerate(module_row["inp_vessels"]):
            if not (module_df["name"] == inp_module).any():
                print(f'the input module of {inp_module} is not defined')
                exit()
            inp_module_row = module_df.loc[module_df["name"] == inp_module].squeeze()
            inp_module_BC_type = inp_module_row["BC_type"]
            inp_module_type = inp_module_row["module_type"]

            self.__check_input_output_modules(module_df, inp_module, main_module,
                                              inp_module_BC_type, main_module_BC_type,
                                              inp_module_type, main_module_type)

        for out_module_idx, out_module in enumerate(module_row["out_vessels"]):
            if not (module_df["name"] == out_module).any():
                print(f'the output module of {out_module} is not defined')
                exit()
            out_module_row = module_df.loc[module_df["name"] == out_module].squeeze()
            out_module_BC_type = out_module_row["BC_type"]
            out_module_type = out_module_row["module_type"]

            # check that input and output modules are defined as connection variables
            # for that module and they have corresponding BCs # TODO this should be done in the model parsing stage
            self.__check_input_output_modules(module_df, main_module, out_module,
                                              main_module_BC_type, out_module_BC_type,
                                              main_module_type, out_module_type)
            self.__check_input_output_ports(module_row["exit_ports"], out_module_row["entrance_ports"],
                                            out_module, main_module)

            # create a list of dicts that stores info for the entrance ports of this output module
            entrance_port_types = []
            for entrance_port_idx in range(len(out_module_row["entrance_ports"])):
                port_type = out_module_row["entrance_ports"][entrance_port_idx]["port_type"]
                if port_type in [entrance_port_types[II]["port_type"] for II in range(len(entrance_port_types))]:
                    port_type_idx = [entrance_port_types[II]["port_type"] for II
                                     in range(len(entrance_port_types))].index(port_type)
                    entrance_port_types[port_type_idx]["entrance_port_idxs"].append(entrance_port_idx)
                    entrance_port_types[port_type_idx]["port_type_count"] += 1
                else:
                    entrance_port_types.append({"port_type": out_module_row["entrance_ports"][entrance_port_idx]["port_type"],
                                       "entrance_port_idxs": [entrance_port_idx],
                                       "port_type_count": 1})

            for port_type_idx in range(len(port_types)):
                for II in range(len(entrance_port_types)):
                    entrance_port_type_idx = -1
                    if entrance_port_types[II]["port_type"] == port_types[port_type_idx]["port_type"]:
                        entrance_port_type_idx = II
                        entrance_port_idx = -1
                        for JJ in entrance_port_types[II]["entrance_port_idxs"]:
                            if entrance_ports_connected[out_module][JJ] == False:
                                entrance_port_idx = JJ
                                break    
                        break
                if entrance_port_type_idx == -1:
                    # this port type isnt used, continue to next one
                    continue
                if entrance_port_idx == -1:
                    print(f"module {out_module} has a single port {port_types[port_type_idx]['port_type']}, connected",
                            f"to multiple ports. Should it have a multiport:True entry in the",
                            f"module_config.json file??")
                    exit()

                if port_types[port_type_idx]["port_type"] == "vessel_port":
                    for this_type_port_idx in range(port_types[port_type_idx]["port_type_count"]):
                        if not port_types[port_type_idx]["connected"][this_type_port_idx]:
                            out_port_idx = port_types[port_type_idx]["out_port_idxs"][this_type_port_idx]
                            break

                    # I dont think this block is needed
                    # # get the entrance port idx for the output module
                    # if entrance_port_types[entrance_port_type_idx]["port_type_count"] == 1:
                    #     entrance_port_idx = 0
                    # else:
                    #     entrance_port_idx = -1
                    #     for II in range(len(out_module_row["inp_vessels"])):
                    #         # check that the entrance port corresponds to the main_vessel
                    #         if out_module_row["inp_vessels"][II] == main_module:
                    #             entrance_port_idx = entrance_port_types[entrance_port_type_idx]["entrance_port_idxs"][II]
                    #             break

                    # TODO this part is kind of hacky, but it works, there is definitely a better way to do the mapping with the
                    #  heart module!
                    if out_module_type.startswith('heart'):
                        if len(out_module_row["inp_vessels"]) == 2 and self.ivc_connection_done == 0:
                            # this is the case if there is only one vc and one pulmonary
                            # We map the ivc to a zero flow mapping
                            self.__write_mapping(wf, 'zero_flow_module', 'heart_module', ['v_zero'], ['v_ivc'])
                            self.ivc_connection_done = 1
                            self.BC_set[out_module]['v_ivc'] = True
                            # TODO the above isnt robust
                        
                        for heart_inp_idx in range(3):
                            # there are three vessel_port entrances to the heart, ivc, svc, and pulmonary
                            # in the vessel_array file, they must be ordered ivc, svc, pulmonary
                            if main_module == out_module_row["inp_vessels"][heart_inp_idx]:
                                # if the ivc connection was done artificially, then we need to skip it
                                entrance_port_idx = heart_inp_idx + self.ivc_connection_done
                                break
                        
                    variables_1 = module_row["exit_ports"][out_port_idx]['variables']
                    variables_2 = out_module_row["entrance_ports"][entrance_port_idx]['variables']
                    
                    if module_row["vessel_type"].endswith('terminal'):
                        # the terminal connections are done through the terminal_venous_connection
                        # TODO after this if loop I set that this connection is done. It is actually done later on
                        # when doing the venous_terminal_connection. Fine for now.
                        pass
                    else:
                        main_module_module = main_module + '_module'
                        out_module_module = out_module + '_module'

                        if module_row['module_format'] == 'cellml' and out_module_row['module_format'] == 'cellml':
                            self.__write_mapping(wf, main_module_module, out_module_module, variables_1, variables_2)

                    # only assign connected if the port doesnt have a multi_ports flag
                    if 'multi_port' in module_row["exit_ports"][out_port_idx].keys():
                        if module_row["exit_ports"][out_port_idx]['multi_port'] in ['True', True]:
                            pass
                        else:
                            port_types[port_type_idx]["connected"][this_type_port_idx] = True
                    else:
                        port_types[port_type_idx]["connected"][this_type_port_idx] = True

                    if 'multi_port' in out_module_row["entrance_ports"][entrance_port_idx].keys():
                        if out_module_row["entrance_ports"][entrance_port_idx]['multi_port'] in ['True', True]:
                            pass
                        else:
                            entrance_ports_connected[out_module][entrance_port_idx] = True
                    else:
                        entrance_ports_connected[out_module][entrance_port_idx] = True

                    for II in range(len(variables_1)):
                        if variables_1[II] in self.BC_set[main_module].keys():
                            self.BC_set[main_module][variables_1[II]] = True
                        if variables_2[II] in self.BC_set[out_module].keys():
                            self.BC_set[out_module][variables_2[II]] = True
                else:
                    for this_type_port_idx in range(port_types[port_type_idx]["port_type_count"]):
                        if not port_types[port_type_idx]["connected"][this_type_port_idx]:
                            out_port_idx = port_types[port_type_idx]["out_port_idxs"][this_type_port_idx]
                            # get the entrance port idx for the output module

                            # TODO the below if isnt general. Figure out how to generalise this.
                            # if out_module_type == 'flow_sum_2_type':
                            #     for II in range(len(out_module_row["inp_vessels"])):
                            #         # check that the entrance port corresponds to the main_vessel
                            #         if out_module_row["inp_vessels"][II] == main_module:
                            #             entrance_port_idx = \
                            #                 entrance_port_types[entrance_port_type_idx]["entrance_port_idxs"][II]
                            #             break
                            # else:
                            
                            variables_1 = module_row["exit_ports"][out_port_idx]['variables']
                            variables_2 = out_module_row["entrance_ports"][entrance_port_idx]['variables']

                            # TODO make this more general. Have a entry in module_config.json entrance and exit 
                            # types that specify a certain operation, like averaging wrt flow, like this gas port.
                            if main_module_type == 'tissue_GE_simple_type' and out_module_type == "gas_transport_simple_type":
                                # this connection is done through the terminal_venous_connection
                                # TODO after this if loop I set that this connection is done. It is actually done later on
                                # when doing the venous_terminal_connection. Fine for now.
                                pass
                            else:
                                main_module_module = main_module + '_module'
                                out_module_module = out_module + '_module'

                                if module_row['module_format'] == 'cellml' and out_module_row['module_format'] == 'cellml':
                                    self.__write_mapping(wf, main_module_module, out_module_module, variables_1, variables_2)

                            for II in range(len(variables_1)):
                                if variables_1[II] in self.BC_set[main_module].keys():
                                    self.BC_set[main_module][variables_1[II]] = True
                                if variables_2[II] in self.BC_set[out_module].keys():
                                    self.BC_set[out_module][variables_2[II]] = True
                                    
                            # only assign connected if the port doesnt have a multi_ports flag
                            if 'multi_port' in module_row["exit_ports"][out_port_idx].keys():
                                if module_row["exit_ports"][out_port_idx]['multi_port'] in ['True', True]:
                                    pass
                                else:
                                    port_types[port_type_idx]["connected"][this_type_port_idx] = True
                            else:
                                port_types[port_type_idx]["connected"][this_type_port_idx] = True

                            if 'multi_port' in out_module_row["entrance_ports"][entrance_port_idx].keys():
                                if out_module_row["entrance_ports"][entrance_port_idx]['multi_port'] in ['True', True]:
                                    pass
                                else:
                                    entrance_ports_connected[out_module][entrance_port_idx] = True
                            else:
                                entrance_ports_connected[out_module][entrance_port_idx] = True

        return entrance_ports_connected

    def __write_terminal_venous_connection_comp(self, wf, vessel_df):
        first_venous_names = [] # stores name of venous compartments that take flow from terminals
        tissue_GE_names = [] # stores name of venous compartments that take flow from terminals
        for vessel_tup in vessel_df.itertuples():
            if vessel_tup.module_format != 'cellml':
                # if not cellml then don't do anything for this vessel/module
                continue
            if vessel_tup.vessel_type.endswith('terminal'):
                vessel_name = vessel_tup.name
                
                out_vessel_names = []
                for out_vessel_name in vessel_tup.out_vessels:
                    if vessel_df.loc[vessel_df["name"] == out_vessel_name].squeeze()["vessel_type"].startswith('venous'):
                        # This finds the venous module connection
                        out_vessel_names.append(out_vessel_name)
                if len(out_vessel_names) == 0:
                    # there is no venous compartment connected to this terminal but still create a terminal venous section
                    pass
                else:
                    # map pressure between terminal and first venous compartments
                    for out_vessel_idx in range(len(out_vessel_names)):
                        u_1 = 'u_out'
                        u_2 = 'u'
                        self.__write_mapping(wf, vessel_name+'_module', out_vessel_names[out_vessel_idx]+'_module',
                                            [u_1], [u_2])

                # then map variables between connection and the venous sections
                v_1 = 'v_T'
                v_2 = f'v_{vessel_name}'

                self.__write_mapping(wf, vessel_name+'_module','terminal_venous_connection',
                              [v_1], [v_2])

            # This only allowed venous types to be called venous.
            # # check that vessel type is venous but and that it is the first venous thats connected to a terminal
            # if vessel_tup.vessel_type == 'venous' and \
            #         vessel_df.loc[vessel_df['name'].isin(vessel_tup.inp_vessels)
            #         ]['vessel_type'].str.contains('terminal').any():

            # check if the vessel has a terminal as an input and has a flow input
            if vessel_df.loc[vessel_df['name'].isin(vessel_tup.inp_vessels)
                    ]['vessel_type'].str.contains('terminal').any() and vessel_tup.BC_type.startswith('v'):
                vessel_name = vessel_tup.name
                first_venous_names.append(vessel_name)
                v_1 = [f'v_{vessel_name}']
                v_2 = ['v_in']

                self.__write_mapping(wf, 'terminal_venous_connection', vessel_name+'_module',
                              v_1, v_2)

            if vessel_tup.vessel_type == 'gas_transport_simple' and \
                    vessel_df.loc[vessel_df['name'].isin(vessel_tup.inp_vessels)
                                  ]['vessel_type'].str.contains('tissue_GE_simple').any():
                vessel_name = vessel_tup.name

                vars_1 = ['v_venous_total', f'C_O2_p_venous_ave', f'C_CO2_p_venous_ave']
                vars_2 = ['v', 'C_O2_in', 'C_CO2_in']
                self.__write_mapping(wf, 'terminal_venous_connection', vessel_name+'_module',
                                     vars_1, vars_2)


            if vessel_tup.vessel_type == 'tissue_GE_simple':
                vessel_name = vessel_tup.name
                tissue_GE_names.append(vessel_name)

                C_1 = [f'C_O2_p_{vessel_name}', f'C_CO2_p_{vessel_name}']
                C_2 = ['C_O2_p', 'C_CO2_p']

                self.__write_mapping(wf, 'terminal_venous_connection', vessel_name+'_module',
                                     C_1, C_2)



        # loop through vessels to get the terminals to add up for each first venous section
        terminal_names_for_first_venous = [[] for i in range(len(first_venous_names))]
        terminal_names_with_GE = []
        terminal_names = []
        for vessel_tup in vessel_df.itertuples():
            if vessel_tup.vessel_type.endswith('terminal'):
                vessel_name = vessel_tup.name
                for idx, venous_name in enumerate(first_venous_names):
                    for II in range(len(vessel_tup.out_vessels)):
                        if vessel_tup.out_vessels[II] == venous_name:
                            if vessel_name not in vessel_df.loc[vessel_df["name"] == venous_name].squeeze()["inp_vessels"]:
                                print(f'venous input of {venous_name} does not include the terminal vessel '
                                      f'{vessel_name} as an inp_vessel in {self.filename_prefix}_vessel_array. '
                                      f'not including terminal names as input has been deprecated')
                                exit()
                            terminal_names_for_first_venous[idx].append(vessel_name)
                for idx, tissue_GE_name in enumerate(tissue_GE_names):
                    for II in range(len(vessel_tup.out_vessels)):
                        if vessel_tup.out_vessels[II] == tissue_GE_name:
                            terminal_names_with_GE.append(vessel_name)
                terminal_names.append(vessel_name)

        # create computation environment for connection and write the
        # variable definition and calculation of flow to each first venous module.
        wf.write(f'<component name="terminal_venous_connection">\n')
        variables = []
        units = []
        in_outs = []
        
        for terminal_name in terminal_names:
            variables.append(f'v_{terminal_name}')
            units.append('m3_per_s')
            in_outs.append('in')

        for idx, venous_name in enumerate(first_venous_names):
            variables.append(f'v_{venous_name}')
            units.append('m3_per_s')
            # TODO check this for full CVS case
            # in_outs.append('priv_in_pub_out')
            in_outs.append('out')

        for GE_name in tissue_GE_names:
            variables.append(f'C_O2_p_{GE_name}')
            variables.append(f'C_CO2_p_{GE_name}')
            units.append('dimensionless')
            units.append('dimensionless')
            in_outs.append('in')
            in_outs.append('in')

        if len(tissue_GE_names) > 0:
            variables.append(f'C_O2_p_venous_ave')
            variables.append(f'C_CO2_p_venous_ave')
            units.append('dimensionless')
            units.append('dimensionless')
            in_outs.append('out')
            in_outs.append('out')

            variables.append(f'v_venous_total')
            units.append('m3_per_s')
            in_outs.append('out')

        self.__write_variable_declarations(wf, variables, units, in_outs)
        for idx, venous_name in enumerate(first_venous_names):
            rhs_variables = []
            lhs_variable = f'v_{venous_name}'
            for terminal_name in terminal_names_for_first_venous[idx]:
                rhs_variables.append(f'v_{terminal_name}')

            self.__write_variable_sum(wf, lhs_variable, rhs_variables)

        if len(tissue_GE_names) > 0:

            rhs_variables_to_average = []
            rhs_variables_weightings = []
            rhs_variables_to_average_CO2 = []
            # sum all venous components to get a total flow to use for gas transport
            lhs_variable_flow = 'v_venous_total'
            lhs_variable = f'C_O2_p_venous_ave'
            lhs_variable_CO2 = f'C_CO2_p_venous_ave'
            for terminal_name, GE_name in zip(terminal_names_with_GE, tissue_GE_names):
                rhs_variables_to_average.append(f'C_O2_p_{GE_name}')
                rhs_variables_to_average_CO2.append(f'C_CO2_p_{GE_name}')
                rhs_variables_weightings.append(f'v_{terminal_name}')

            self.__write_variable_sum(wf, lhs_variable_flow, rhs_variables_weightings)
            self.__write_variable_average(wf, lhs_variable, rhs_variables_to_average, rhs_variables_weightings)
            self.__write_variable_average(wf, lhs_variable_CO2, rhs_variables_to_average_CO2, rhs_variables_weightings)


        wf.write('</component>\n')

    def __write_access_variables(self, wf, vessel_df):
        vessel_df.apply(self.__write_access_variables_for_row, args=(wf,), axis=1)

    def __write_access_variables_for_row(self, vessel_row, wf):
        if vessel_row["module_format"] != 'cellml':
            # if not cellml then don't do anything for this vessel/module
            return
        wf.write(f'<component name="{vessel_row["name"]}">\n')
        lines_to_write = []
        for variable, unit, access_str, variable_type in vessel_row["variables_and_units"]:
            if access_str == 'access':
                lines_to_write.append(f'   <variable name="{variable}" public_interface="in" units="{unit}"/>\n')
        wf.writelines(lines_to_write)
        wf.write('</component>\n')

    def __write_comp_to_module_mappings(self, wf, vessel_df):
        vessel_df.apply(self.__write_comp_to_module_mappings_for_row, args=(wf,), axis=1)

    def __write_comp_to_module_mappings_for_row(self, vessel_row, wf):
        if vessel_row["module_format"] != 'cellml':
            # if not cellml then don't do anything for this vessel/module
            return
        vessel_name = vessel_row["name"]
        inp_vars = [vessel_row["variables_and_units"][i][0] for i in
                    range(len(vessel_row["variables_and_units"])) if
                    vessel_row["variables_and_units"][i][2] == 'access']
        out_vars = inp_vars

        self.__write_mapping(wf, vessel_name, vessel_name + '_module', inp_vars, out_vars)

    def __write_param_mappings(self, wf, vessel_df, params_array=None):
        vessel_df.apply(self.__write_param_mappings_for_row, args=(wf,), params_array=params_array, axis=1)

    def __write_param_mappings_for_row(self, vessel_row, wf, params_array=None):
        if vessel_row["module_format"] != 'cellml':
            # if not cellml then don't do anything for this vessel/module
            return
        
        vessel_name = vessel_row["name"]
        module_addon = '_module'

        global_variable_addon = f'_{vessel_name}'
        # TODO check this doesnt break anything
        # if vessel_row["vessel_type"] == 'terminal' or vessel_row["vessel_type"] == 'terminal2':
        #     global_variable_addon = re.sub('_T$', '', global_variable_addon)
        params_with_addon_heading = 'parameters'
        params_without_addon_heading = 'parameters_global'

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
                    print(f'________________ERROR_________________')
                    print(f'variable {variable_name} is not in the parameter '
                          f'dataframe/csv file. It needs to be added!!!')
                    print(f'______________________________________')
                    exit()
        self.__write_mapping(wf, params_with_addon_heading, vessel_name + module_addon,
                             module_vars, vars)

        self.__write_mapping(wf, params_without_addon_heading, vessel_name + module_addon,
                             global_vars, global_vars)

    def __write_time_mappings(self, wf, vessel_df):
        vessel_df.apply(self.__write_time_mappings_for_row, args=(wf,), axis=1)

    def __write_time_mappings_for_row(self, vessel_row, wf):
        vessel_name = vessel_row["name"]
        module_addon = '_module'

        if vessel_row["module_format"] != 'cellml':
            # if not cellml then don't do anything for this vessel/module
            return

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

    def __check_input_output_modules(self, vessel_df, main_vessel, out_vessel,
                                   main_vessel_BC_type, out_vessel_BC_type,
                                   main_vessel_type, out_vessel_type):
        if not out_vessel:
            print(f'connection modules incorrectly defined for {main_vessel}')
            exit()
        if main_vessel_type.endswith('terminal'):
            pass
        elif main_vessel not in vessel_df.loc[vessel_df["name"] == out_vessel].inp_vessels.values[0]:
            print(f'"{main_vessel}" and "{out_vessel}" are incorrectly connected, '
                  f'check that they have eachother as output/input')
            exit()
        elif out_vessel not in vessel_df.loc[vessel_df["name"] == main_vessel].out_vessels.values[0]:
            print(f'"{main_vessel}" and "{out_vessel}" are incorrectly connected, '
                  f'check that they have eachother as output/input')
            exit()
        if out_vessel_BC_type.startswith('nn'):
            return
        if main_vessel_BC_type.startswith('nn'):
            return


        if len(main_vessel_BC_type) > 2:
            temp_main_vessel_BC_type = main_vessel_BC_type[:2]
        else:
            temp_main_vessel_BC_type = main_vessel_BC_type

        if temp_main_vessel_BC_type.endswith('v'):
            if not out_vessel_BC_type.startswith('p'):
                print(f'"{main_vessel}" output BC is v, the input BC of "{out_vessel}"',
                      ' should be p')
                exit()
        if temp_main_vessel_BC_type.endswith('p'):
            if not out_vessel_BC_type.startswith('v'):
                print(f'"{main_vessel}" output BC is p, the input BC of "{out_vessel}"',
                      ' should be v')
                exit()

    def __check_input_output_ports(self, exit_ports, entrance_ports, exit_port_name, entrance_port_name):
        # check that input and output modules have a matching port
        shared_port = False
        for exit_port in exit_ports:
            if exit_port["port_type"] in [entrance_port["port_type"] for entrance_port in entrance_ports]:
                shared_port = True
                break
        if shared_port == False:
            print(f'output module {exit_port_name} and input module {entrance_port_name} do not have a matching port,'
                  f'check the module configuration file')
            print(f'input module exit ports: {[exit_port["port_type"] for exit_port in exit_ports]}')
            print(f'output module entrance ports: {[entrance_port["port_type"] for entrance_port in entrance_ports]}')
            exit()
            

    def __write_mapping(self, wf, inp_name, out_name, inp_vars_list, out_vars_list):
        mapping = ['<connection>\n', f'   <map_components component_1="{inp_name}" component_2="{out_name}"/>\n']
        for inp_var, out_var in zip(inp_vars_list, out_vars_list):
            if inp_var and out_var:
                mapping.append(f'   <map_variables variable_1="{inp_var}" variable_2="{out_var}"/>\n')

        mapping.append('</connection>\n')
        if len(mapping) > 3:
            wf.writelines(mapping)

    def __write_variable_declarations(self, wf, variables, units, in_outs):
        for variable, unit, in_out in zip(variables, units, in_outs):
            if in_out == 'priv_in_pub_out':
                wf.write(f'<variable name="{variable}" private_interface="in" public_interface="out" units="{unit}"/>\n')
            else:
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

    def __write_variable_average(self, wf, lhs_variable, rhs_variables_to_average, rhs_variables_weighting):
        """ writes the cellml code for averaging variables with weighting. Designed for getting an equivalent
        concentration

        C = (C_1*v_1 + C_2*v_2)/(v_1 + v_2)
        """
        if len(rhs_variables_to_average) != len(rhs_variables_weighting):
            print('rhs_variables_to_average and rhs_variables_weighting must be the same length, exiting')
            exit()

        wf.writelines('<math xmlns="http://www.w3.org/1998/Math/MathML">\n'
                      '   <apply>\n'
                      '       <eq/>\n'
                      f'       <ci>{lhs_variable}</ci>\n')
        if len(rhs_variables_to_average) > 1:
            wf.write('      <apply>\n')
            wf.write('          <divide/>\n')
            wf.write('          <apply>\n')
            wf.write('              <plus/>\n')
            for variable, weight in zip(rhs_variables_to_average, rhs_variables_weighting):
                wf.write('          <apply>\n')
                wf.write('              <times/>\n')
                wf.write(f'             <ci>{variable}</ci>\n')
                wf.write(f'             <ci>{weight}</ci>\n')
                wf.write('          </apply>\n')
            wf.write('          </apply>\n')
            wf.write('          <apply>\n')
            wf.write('              <plus/>\n')
            wf.write('              <apply>\n')
            wf.write('                  <abs/>\n')
            wf.write('                  <apply>\n')
            wf.write('                      <plus/>\n')
            for weight in rhs_variables_weighting:
                wf.write(f'                     <ci>{weight}</ci>\n')
            wf.write('                  </apply>\n')
            wf.write('              </apply>\n')
            wf.write('              <cn cellml:units="m3_per_s" type="e-notation">1<sep/>-10</cn>\n')
            wf.write('          </apply>\n')
            wf.write('      </apply>\n')

        else:
            wf.write(f'            <ci>{rhs_variables_to_average[0]}</ci>\n')

        wf.write('   </apply>\n')
        wf.write('</math>\n')
