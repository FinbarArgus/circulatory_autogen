import numpy as np
import os
import re
from sys import exit
import pandas as pd
import csv

def write_mapping(wf, inp_name, out_name, inp_vars_list, out_vars_list):
    wf.writelines(['<connection>\n',
    f'   <map_components component_1="{inp_name}" component_2="{out_name}"/>\n'])
    for inp_var, out_var in zip(inp_vars_list, out_vars_list):
        if inp_var and out_var:
            wf.write(f'   <map_variables variable_1="{inp_var}" variable_2="{out_var}"/>\n')
    wf.write('</connection>\n')

def write_imports(wf, vessel_array):
    for vessel_vec in vessel_array:
        if vessel_vec["vessel_type"] in ['heart', 'pulmonary']:
            continue
        write_import(wf, vessel_vec)

def write_import(wf, vessel_vec):
    if vessel_vec['vessel_type'] == 'terminal':
        module_type = f'{vessel_vec["BC_type"]}_T_type'
    elif vessel_vec['vessel_type'] == 'split_junction':
        module_type = f'{vessel_vec["BC_type"]}_split_type'
    elif vessel_vec['vessel_type'] == 'merge_junction':
        module_type = f'{vessel_vec["BC_type"]}_merge_type'
    elif vessel_vec['vessel_type'] == '2in2out_junction':
        module_type = f'{vessel_vec["BC_type"]}_2in2out_type'
    elif vessel_vec['vessel_type'] == 'venous':
        module_type = f'{vessel_vec["BC_type"]}_simple_type'
    else:
        module_type = f'{vessel_vec["BC_type"]}_type'

    if vessel_vec["name"] == 'heart':
        str_addon = ''
    else:
        str_addon = '_module'

    wf.writelines(['<import xlink:href="BG_Modules.cellml">\n',
    f'    <component component_ref="{module_type}" name="{vessel_vec[0]+str_addon}"/>\n',
    '</import>\n'])

def write_access_variables(wf, vessel_array):
    for vessel_vec in vessel_array:
        if vessel_vec["vessel_type"] in ['heart', 'pulmonary']:
            continue
        wf.writelines([f'<component name="{vessel_vec["name"]}">\n',
        '   <variable name="u" public_interface="in" units="J_per_m3"/>\n',
        '   <variable name="v" public_interface="in" units="m3_per_s"/>\n'])
        if vessel_vec['vessel_type']=='terminal':
            wf.write('   <variable name="R_T" public_interface="in" units="Js_per_m6"/>\n')
            wf.write('   <variable name="C_T" public_interface="in" units="m6_per_J"/>\n')
        if vessel_vec['vessel_type']=='venous':
            wf.write('   <variable name="R" public_interface="in" units="Js_per_m6"/>\n')
            wf.write('   <variable name="C" public_interface="in" units="m6_per_J"/>\n')
        wf.write('</component>\n')

def write_section_break(wf, text):
    wf.write('<!--&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;' +
            text + '&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;//-->\n')

def write_vessel_mappings(wf, vessel_array):
    for vessel_vec in vessel_array:
        # input and output vessels
        main_vessel = vessel_vec["name"]
        main_vessel_BC_type = vessel_vec["BC_type"]
        main_vessel_type = vessel_vec["vessel_type"]
        out_vessel = vessel_vec["out_vessel_1"]
        if out_vessel not in vessel_array["name"]:
            print(f'the output vessel of {out_vessel} is not defined')
            exit()
        out_vessel_vec = vessel_array[np.where(vessel_array["name"] == out_vessel)][0]
        out_vessel_BC_type = out_vessel_vec["BC_type"]
        out_vessel_type = out_vessel_vec["vessel_type"]

        # check that input and output vessels are defined as connection variables
        # for that vessel and they have corresponding BCs
        check_input_output_vessels(vessel_array, main_vessel, out_vessel,
                                   main_vessel_BC_type, out_vessel_BC_type,
                                   main_vessel_type, out_vessel_type)

        # determine BC variables from vessel_type and BC_type
        if main_vessel_type == 'heart':
            v_1 = 'v_lv'
            p_1 = 'u_root'
        elif main_vessel_type == 'last_venous':
            v_1 = 'v'
            p_1 = 'u_out'
        elif main_vessel_type == 'terminal':
            # flow output is to the terminal_venous_connection, not
            # to the venous module
            v_1 = ''
            p_1 = 'u_out'
        elif main_vessel_type == 'split_junction':
            if main_vessel_BC_type.endswith('v'):
                v_1 = 'v_out_1'
                p_1 = 'u'
            elif main_vessel_BC_type == 'vp':
                v_1 = 'v_out_1'
                p_1 = 'u'
            elif main_vessel_BC_type == 'pp':
                print('Currently we have not implemented junctions'
                      'with output pressure boundary conditions, '
                      f'change {main_vessel}')
                exit()
        elif main_vessel_type == '2in2out_junction':
            if main_vessel_BC_type == 'vv':
                v_1 = 'v_out_1'
                p_1 = 'u_d'
            else:
                print('2in2out vessels only have vv type BC, '
                      f'change "{main_vessel}" or create new BC module '
                      f'in BG_Modules.cellml')
                exit()
        elif main_vessel_type == 'merge_junction':
            if main_vessel_BC_type == 'vp':
                v_1 = 'v'
                p_1 = 'u_out'
            else:
                print('Merge boundary condiditons only have vp type BC, '
                      f'change "{main_vessel}" or create new BC module in '
                      f'BG_Modules.cellml')
                exit()
        else:
            if main_vessel_BC_type.endswith('v'):
                v_1 = 'v_out'
                p_1 = 'u'
            elif main_vessel_BC_type == 'vp':
                v_1 = 'v'
                p_1 = 'u_out'
            elif main_vessel_BC_type == 'pp':
                v_1 = 'v_d'
                p_1 = 'u_out'

        if out_vessel_type == 'heart':
            if main_vessel == 'venous_ivc':
                v_2 = 'v_ivc'
            elif main_vessel == 'venous_svc':
                v_2 = 'v_svc'
            else:
                print('venous input to heart can only be venous_ivc or venous_svc')
            p_2 = 'u_ra'
        elif out_vessel_type in ['merge_junction', '2in2out_junction']:
            if out_vessel_vec["inp_vessel_1"] == main_vessel:
                v_2 = 'v_in_1'
            elif out_vessel_vec["inp_vessel_2"] == main_vessel:
                v_2 = 'v_in_2'
            else:
                print('error, exiting')
                exit()
            p_2 = 'u'
        elif main_vessel_type == 'terminal':
            # For terminal output we link to a terminal_venous connection
            # to sum up the output terminal flows
            if out_vessel_BC_type == 'vp':
                v_2 = ''
                p_2 = f'u'
            else:
                print('venous section connected to terminal only works'
                      'for vp BC currently')
                exit()
        else:
            if out_vessel_BC_type == 'vp':
                v_2 = 'v_in'
                p_2 = 'u'
            elif out_vessel_BC_type == 'vv':
                v_2 = 'v_in'
                p_2 = 'u_C'
            elif out_vessel_BC_type.startswith('p'):
                v_2 = 'v'
                p_2 = 'u_in'

        # TODO make heart and pulmonary BG modules so everything can be a module
        if main_vessel in ['heart', 'pulmonary']:
            main_vessel_module = main_vessel
        else:
            main_vessel_module = main_vessel + '_module'
        if out_vessel in ['heart', 'pulmonary']:
            out_vessel_module = out_vessel
        else:
            out_vessel_module = out_vessel + '_module'

        write_mapping(wf, main_vessel_module, out_vessel_module, [v_1, p_1], [v_2, p_2])

        if vessel_vec["vessel_type"].endswith('junction'):
            if vessel_vec["vessel_type"] in ['split_junction', '2in2out_junction']:
                out_vessel = vessel_vec["out_vessel_2"]
                if out_vessel in ['heart', 'pulmonary']:
                    out_vessel_module = out_vessel
                else:
                    out_vessel_module = out_vessel + '_module'
                if vessel_vec["BC_type"].endswith('v'):
                    v_1 = 'v_out_2'
                else:
                    pass
            if out_vessel not in vessel_array["name"]:
                print(f'the output vessel of {out_vessel} is not defined')
                exit()
            out_vessel_vec = vessel_array[np.where(vessel_array["name"] == out_vessel)][0]
            out_vessel_BC_type = out_vessel_vec["BC_type"]
            out_vessel_type = out_vessel_vec["vessel_type"]
            check_input_output_vessels(vessel_array, main_vessel, out_vessel,
                                       main_vessel_BC_type, out_vessel_BC_type,
                                       main_vessel_type, out_vessel_type)
            write_mapping(wf, main_vessel_module, out_vessel_module, [v_1, p_1], [v_2, p_2])

def write_comp_to_module_mappings(wf, vessel_array):
    for vessel_vec in vessel_array:
        # input and output vessels
        vessel_name = vessel_vec["name"]
        if vessel_name in ['heart', 'pulmonary']:
            # TODO make the heart and pulmonary sections modules instead
            # of prewritten comp environments in the base cellml code.
            continue
        if vessel_vec["vessel_type"] == 'terminal':
            inp_vars = ['u', 'v', 'R_T', 'C_T']
            out_vars = ['u', 'v_T', 'R_T', 'C_T']
        elif vessel_vec["vessel_type"] == 'venous':
            inp_vars = ['u', 'v', 'C', 'R']
            out_vars = ['u', 'v', 'C', 'R']
        else:
            inp_vars = ['u', 'v']
            out_vars = ['u', 'v']
        write_mapping(wf, vessel_name, vessel_name+'_module', inp_vars, out_vars)

def write_param_mappings(wf, vessel_array, params_array=None):
    for vessel_vec in vessel_array:
        # input and output vessels
        vessel_name = vessel_vec["name"]
        if vessel_name in ['heart', 'pulmonary']:
            continue

        if vessel_vec["vessel_type"] == 'terminal':
            vessel_name_minus_T = re.sub('_T$', '', vessel_name)
            systemic_vars = [f'R_T_{vessel_name_minus_T}',
                             f'C_T_{vessel_name_minus_T}',
                             f'alpha_{vessel_name_minus_T}',
                             f'v_nom_{vessel_name_minus_T}',
                             'gain_int']
            module_vars = ['R_T',
                           'C_T',
                           'alpha',
                           'v_nominal',
                           'gain_int']

        elif vessel_vec["vessel_type"]=='venous':
            systemic_vars = [f'R_{vessel_name}',
                             f'C_{vessel_name}',
                             f'I_{vessel_name}']
            module_vars = ['R',
                           'C',
                           'I']
        else:
            systemic_vars = [f'l_{vessel_name}',
                             f'E_{vessel_name}',
                             f'r_{vessel_name}',
                             f'theta_{vessel_name}',
                             'beta_g']
            module_vars = ['l',
                           'E',
                           'r',
                           'theta',
                           'beta_g']

        # check that the variables are in the paramter array
        if params_array is not None:
            for variable_name in systemic_vars:
                if variable_name not in params_array["variable_name"]:
                    print(f'variable {variable_name} is not in the parameter '
                          f'dataframe/csv file')
                    exit()
        module_addon = '_module'
        write_mapping(wf, 'parameters_systemic', vessel_name+module_addon,
                      systemic_vars, module_vars)

def write_time_mappings(wf, vessel_array):
    for vessel_vec in vessel_array:
        # input and output vessels
        vessel_name = vessel_vec["name"]
        if vessel_name in ['heart', 'pulmonary']:
            continue

        module_addon = '_module'
        write_mapping(wf, 'environment', vessel_name+module_addon,
                      ['time'], ['t'])

def write_terminal_venous_connection_comp(wf, vessel_array):
    first_venous_names = [] # stores name of venous compartments that take flow from terminals
    for vessel_vec in vessel_array:
        # first map variables between connection and the venous sections
        if vessel_vec["vessel_type"] == 'terminal':
            vessel_name = vessel_vec["name"]
            out_vessel_name = vessel_vec["out_vessel_1"]
            v_1 = 'v_T'
            v_2 = f'v_{vessel_name}'

            write_mapping(wf, vessel_name+'_module','terminal_venous_connection',
                          [v_1], [v_2])

        if vessel_vec["vessel_type"] == 'venous' and not vessel_vec['inp_vessel_1']:
            vessel_name = vessel_vec["name"]
            first_venous_names.append(vessel_name)
            vessel_BC_type = vessel_vec["BC_type"]
            v_1 = f'v_{vessel_name}'
            if vessel_BC_type == 'vp':
                v_2 = 'v_in'
            else:
                print(f'first venous vessel BC type of {vessel_BC_type} has not'
                      f'been implemented')

            write_mapping(wf, 'terminal_venous_connection', vessel_name+'_module',
                          [v_1], [v_2])

    # loop through vessels to get the terminals to add up for each first venous section
    terminal_names_for_first_venous = [[] for i in range(len(first_venous_names))]
    for vessel_vec in vessel_array:
        if vessel_vec['vessel_type'] == 'terminal':
            vessel_name = vessel_vec["name"]
            for idx, venous_name in enumerate(first_venous_names):
                if vessel_vec['out_vessel_1'] == venous_name:
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

    write_variable_declarations(wf, variables, units, in_outs)
    for idx, venous_name in enumerate(first_venous_names):
        rhs_variables = []
        lhs_variable = f'v_{venous_name}'
        for terminal_name in terminal_names_for_first_venous[idx]:
            rhs_variables.append(f'v_{terminal_name}')

        write_variable_sum(wf, lhs_variable, rhs_variables)
    wf.write('</component>')

def write_variable_declarations(wf, variables, units, in_outs):
    for variable, unit, in_out in zip(variables, units, in_outs):
        wf.write(f'<variable name="{variable}" public_interface="{in_out}" units="{unit}"/>')

def write_constant_declarations(wf, variable_names, units, values):
    for variable, unit, value in zip(variable_names, units, values):
        wf.write(f'<variable initial_value="{value}" name="{variable}" '
                 f'public_interface="out" units="{unit}"/>\n')

def write_variable_sum(wf, lhs_variable, rhs_variables):
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
    wf.write('</math>')

def modify_init_states_from_param_id(param_id_dir, cellml_path):

    param_state_names = []

    #param names that were identified in param_id
    with open(os.path.join(os.path.join(param_id_dir, 'param_state_names_for_gen.csv')), 'r') as f:
        rd = csv.reader(f)
        for row in rd:
            param_state_names.append(row)

    param_vals = np.load(os.path.join(param_id_dir, 'best_param_vals.npy'))

    state_param_name_and_val = []
    # this only looks at the first param_vals relating to param_state_names, not to constants
    for name_or_list, val in zip(param_state_names, param_vals):
        if name_or_list in param_state_names:
            if isinstance(name_or_list, list):
                for name in name_or_list:
                    state_param_name_and_val.append((name, val))
            else:
                state_param_name_and_val.append((name, val))

    with open(cellml_path, 'r') as rf:
        lines = rf.readlines()

    for idx, line in enumerate(lines):
        for state_name, val in state_param_name_and_val:
            if state_name in line and 'initial_value' in line:
                inp_string =  f'initial_value="{val:.4e}"'
                new_line = re.sub('initial_value=\"\d*\.?\d*e?-?\d*\"', inp_string, line)
                lines[idx] = new_line

    with open(cellml_path, 'w') as wf:
        wf.writelines(lines)


def modify_consts_from_param_id(param_id_dir, params_array):

    param_state_names = []
    param_const_names = []

    #param names that were identified in param_id
    with open(os.path.join(os.path.join(param_id_dir, 'param_state_names_for_gen.csv')), 'r') as f:
        rd = csv.reader(f)
        for row in rd:
            param_state_names.append(row)
    with open(os.path.join(os.path.join(param_id_dir, 'param_const_names_for_gen.csv')), 'r') as f:
        rd = csv.reader(f)
        for row in rd:
            param_const_names.append(row)

    # get date identifier of the parameter id
    date_id = np.load(os.path.join(os.path.join(param_id_dir, 'date.npy'))).item()

    param_names = param_state_names + param_const_names
    param_vals = np.load(os.path.join(param_id_dir, 'best_param_vals.npy'))

    # first modify param_const names easily by modifying them in the array
    for name_or_list, val in zip(param_names, param_vals):
        if name_or_list in param_state_names:
            pass
        else:
            if isinstance(name_or_list, list):
                for name in name_or_list:
                    params_array[np.where(params_array['variable_name'] == name)[0][0]]['value'] = f'{val:.4e}'
                    params_array[np.where(params_array['variable_name'] == name)[0][0]]['data_reference'] = \
                        f'{date_id}_identified'
            else:
                params_array[np.where(params_array['variable_name'] == name)[0][0]] = val


def save_params_array_as_csv(params_array, params_output_csv_path):
    df = pd.DataFrame(params_array)
    df.to_csv(params_output_csv_path, index=None, header=True)


def check_input_output_vessels(vessel_array, main_vessel, out_vessel,
                               main_vessel_BC_type, out_vessel_BC_type,
                               main_vessel_type, out_vessel_type):
    if not out_vessel:
        print(f'connection vessels incorrectly defined for {main_vessel}')
        exit()
    if main_vessel_type == 'terminal':
        pass
    elif main_vessel not in vessel_array[np.where(vessel_array["name"]==out_vessel
                                                )][["inp_vessel_1", "inp_vessel_2"]][0]:
        print(f'"{main_vessel}" and "{out_vessel}" are incorrectly connected, '
              f'check that they have eachother as output/input')
        exit()

    if main_vessel_BC_type.endswith('v'):
        if not out_vessel_BC_type.startswith('p'):
            print(f'"{main_vessel}" output BC is v, the input BC of "{out_vessel}"',
                  ' should be p')
    if main_vessel_BC_type.endswith('p'):
        if not out_vessel_BC_type.startswith('v'):
            print(f'"{main_vessel}" output BC is p, the input BC of "{out_vessel}"',
                  ' should be v')
def check_BC_types_and_vessel_types(vessel_array, possible_BC_types, possible_vessel_types):
    for vessel_vec in vessel_array:
        if vessel_vec['BC_type'] not in possible_BC_types:
            print(f'BC_type of {vessel_vec["BC_type"]} is not allowed for vessel {vessel_vec["name"]}')
        if vessel_vec['vessel_type'] not in possible_vessel_types:
            print(f'vessel_type of {vessel_vec["BC_type"]} is not allowed for vessel {vessel_vec["name"]}')

def get_params_df_from_csv(params_csv_path):
    """
    Takes in a data frame of the params and
    """
    params_df = pd.read_csv(params_csv_path, dtype=str)
    for column_name in params_df.columns:
        params_df[column_name] = params_df[column_name].str.strip()

    return params_df

def get_np_array_from_vessel_csv(vessel_array_csv_path):
    vessel_df = pd.read_csv(vessel_array_csv_path, header=None)
    for column_name in vessel_df.columns:
        vessel_df[column_name] = vessel_df[column_name].str.strip()

    vessel_array = vessel_df.to_numpy()
    dtype = get_dtype_vessel_array()
    vessel_array = np.array(list(zip(*vessel_array.T)), dtype=dtype)

    return vessel_array

def get_params_array_from_df(params_df):
    param_array = params_df.to_numpy()
    dtype = [(params_df.columns[II],(np.str_, 64)) for II in range(len(params_df.columns))]
    param_array = np.array(list(zip(*param_array.T)), dtype=dtype)

    return param_array

def get_dtype_vessel_array():
    dtype = [('name', (np.str_, 64)), ('BC_type',(np.str_, 64)), ('vessel_type',(np.str_, 64)), ('inp_vessel_1',(np.str_, 64)),
    ('inp_vessel_2',(np.str_, 64)), ('out_vessel_1',(np.str_, 64)), ('out_vessel_2',(np.str_, 64))]
    return dtype
