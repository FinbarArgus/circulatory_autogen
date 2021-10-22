import numpy as np
import os
import re
import pandas as pd
from generation_helper import *

def generate_model_cellml_from_array(vessel_array, case_name, parameters_array=None):
    """
    vessel_array: input is a numpy array with information of each circulatory vessel
    make sure the first entry is the output to the left ventricle
    entries are given by the following
    [vessel_name,
    BC type ('vv', 'vp', 'pv', 'pp'),
    vessel type ('arterial', 'venous', 'terminal', 'split_junction', 'merge_junction', 2in2out_junction),
    input vessel name (This doesn't need to be specified for venous type
    input vessel name 2 if merge or 2in2out junction, '' otherwise
    output vessel name
    output vessel name 2 if split or 2in2out junction, '' otherwise]
    TODO create converging junction for more than 2 vessels? atm this is done by summing flows
     in the Systemic component then using the summed flow as the input

    TODO automate choices of BC_type
    """

    base_script_path = 'base_script.cellml'
    new_file = f'autogen_{case_name}.cellml'
    # check BC types and vessel types
    possible_vessel_types = ['heart', 'arterial', 'split_junction', 'merge_junction', '2in2out_junction',
                             'terminal', 'venous']
    possible_BC_types = ['pp', 'vv', 'pv', 'vp']
    check_BC_types_and_vessel_types(vessel_array, possible_BC_types, possible_vessel_types)

    print(f'Starting autogeneration of {new_file}')
    print('copying base gen')
    # Now open base script and create new file
    with open(base_script_path, 'r') as rf:
        with open(new_file, 'w') as wf:
            for line in rf:
                # copy the start of the basescript until line that says #STARTGENBELOW
                wf.write(line)
                if '#STARTGENBELOW' in line:
                    break

            ###### now start generating own code ######

            # Now map between Systemic component and terminal components
            # TODO This doesn't need to be done anymore because we will have
            #  venous sections and alpha params will be in parameters file
            #   make sure I do mapping between terminals and venous section
            #    after arterial mapping

            # import vessels
            print('writing imports')
            write_section_break(wf, 'imports')
            write_imports(wf, vessel_array)

            # define mapping between vessels
            print('writing vessel mappings')
            write_section_break(wf, 'vessel mappings')
            write_vessel_mappings(wf, vessel_array)

            # create computation environment to sum flows from terminals
            # to have a total flow input into each first venous component.
            print('writing environment to sum venous input flows')
            write_section_break(wf, 'terminal venous connection')
            write_terminal_venous_connection_comp(wf, vessel_array)

            # define variables so they can be accessed
            print('writing variable access')
            write_section_break(wf, 'access_variables')
            write_access_variables(wf, vessel_array)

            # map between computational environment and module so they can be accessed
            print('writing mappings between computational environment and modules')
            write_section_break(wf, 'vessel mappings')
            write_comp_to_module_mappings(wf, vessel_array)

            # map constants to different modules
            print('writing mappings between constant parameters')
            write_section_break(wf, 'parameters mapping to modules')
            write_parameter_mappings(wf, vessel_array, parameters_array=parameters_array)

            # map environment time to module times
            print('writing writing time mappings between environment and modules')
            write_section_break(wf, 'time mapping')
            write_time_mappings(wf, vessel_array)

            # Finalise the file
            wf.write('</model>\n')

            print(f'Finished autogeneration of {new_file}')

def generate_parameters_cellml_from_array(parameters_array, case_name=None):
    """
    Takes in a data frame of the parameters and generates the parameter_cellml file
    TODO make this function case_name specific
    """
    print("Starting autogeneration of parameter cellml file")
    param_file = 'parameters_autogen.cellml'

    with open(param_file, 'w') as wf:

        wf.write('<?xml version=\'1.0\' encoding=\'UTF-8\'?>\n')
        wf.write('<model name="Parameters" xmlns="http://www.cellml.org/cellml/1.1#'
                 '" xmlns:cellml="http://www.cellml.org/cellml/1.1#">\n')

        heart_params_array = parameters_array[np.where(parameters_array["comp_env"]=='heart')]
        pulmonary_params_array = parameters_array[np.where(parameters_array["comp_env"]=='pulmonary')]
        systemic_params_array = parameters_array[np.where(parameters_array["comp_env"]=='systemic')]

        wf.write('<component name="parameters_pulmonary">\n')
        write_constant_declarations(wf, pulmonary_params_array["variable_name"],
                                    pulmonary_params_array["units"],
                                    pulmonary_params_array["value"])
        wf.write('</component>\n')
        wf.write('<component name="parameters_heart">\n')
        write_constant_declarations(wf, heart_params_array["variable_name"],
                                    heart_params_array["units"],
                                    heart_params_array["value"])
        wf.write('</component>\n')
        wf.write('<component name="parameters_systemic">\n')
        write_constant_declarations(wf, systemic_params_array["variable_name"],
                                    systemic_params_array["units"],
                                    systemic_params_array["value"])
        wf.write('</component>\n')
        wf.write('</model>\n')

        print(f'Finished autogeneration of {param_file}')

if __name__ == '__main__':
    # case_name = 'physiological' # includes circle of willis, and trunk terminal
    case_name = 'simple_physiological' # same as physiological but cut off at common carotid,
                                       # doesnt include COW or vertebral arteries.
    # case_name = 'test' # non physiological. A simple example for testing
    get_vessel_array_from_csv = True
    vessel_array_csv_path = f'{case_name}_vessel_array.csv'

    parameters_csv_path = 'parameters_autogen.csv'
    parameters_df = get_parameters_df_from_csv(parameters_csv_path)
    parameters_array = get_parameters_array_from_df(parameters_df)

    vessel_array = get_np_array_from_vessel_csv(vessel_array_csv_path)

    # generate the cellml model from the vessel array
    generate_model_cellml_from_array(vessel_array, case_name, parameters_array=parameters_array)
    # generate the parameters cellml file
    generate_parameters_cellml_from_array(parameters_array)


