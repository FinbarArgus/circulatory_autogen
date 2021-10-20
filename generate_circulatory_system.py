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
    case_name = 'test'
    get_vessel_array_from_csv = False
    vessel_array_csv_path = 'simple_vessel_array.csv'
    parameters_csv_path = 'parameters_autogen.csv'
    parameters_df = get_parameters_df_from_csv(parameters_csv_path)
    parameters_array = get_parameters_array_from_df(parameters_df)

    if get_vessel_array_from_csv:
        case_name += '_simple'
        vessel_array = get_np_array_from_vessel_csv(vessel_array_csv_path)
    else:
        vessel_array = np.array([
        ('heart',                   'vp', 'heart',              'venous_ivc',           'venous_svc',
         'ascending_aorta_A',    ''),
        ('ascending_aorta_A',       'vv', 'arterial',           'heart',                '',
         'ascending_aorta_B',    ''),
        ('ascending_aorta_B',       'pv', 'arterial',           'ascending_aorta_A',    '',
         'ascending_aorta_C',    ''),
        ('ascending_aorta_C',       'pv', 'arterial',           'ascending_aorta_B',    '',
         'ascending_aorta_D',    ''),
        ('ascending_aorta_D',       'pv', 'arterial',           'ascending_aorta_C',    '',
         'aortic_arch_C2',       ''),
        ('aortic_arch_C2',          'pv', 'split_junction',     'ascending_aorta_D',    '',
         'aortic_arch_C46',      'brachiocephalic_trunk_C4'),
        ('aortic_arch_C46',         'pv', 'split_junction',     'aortic_arch_C2',       '',
         'aortic_arch_C64',      'common_carotid_L48_A'),
        ('aortic_arch_C64',         'pv', 'split_junction',     'aortic_arch_C46',      '',
         'aortic_arch_C94',      'subclavian_L66'),
        ('brachiocephalic_trunk_C4','pv', 'split_junction',     'aortic_arch_C2',       '',
         'common_carotid_R6_A',  'subclavian_R28'),
        ('subclavian_L66',          'pv', 'split_junction',     'aortic_arch_C64',      '',
         'subclavian_L78',       'vertebral_L2'),
        ('subclavian_L78',          'pp', 'arterial',           'subclavian_L66',       '',
         'arm_L_T',              ''),
        ('common_carotid_L48_A',    'pv', 'arterial',           'aortic_arch_C46',      '',
         'common_carotid_L48_B', ''),
        ('common_carotid_L48_B',    'pv', 'arterial',           'common_carotid_L48_A', '',
         'common_carotid_L48_C', ''),
        ('common_carotid_L48_C',    'pv', 'arterial',           'common_carotid_L48_B', '',
         'common_carotid_L48_D', ''),
        ('common_carotid_L48_D',    'pv', 'split_junction',     'common_carotid_L48_C', '',
         'external_carotid_L62', 'internal_carotid_L50_A'),
        ('common_carotid_R6_A',     'pv', 'arterial',           'brachiocephalic_trunk_C4', '',
         'common_carotid_R6_B',  ''),
        ('common_carotid_R6_B',     'pv', 'arterial',           'common_carotid_R6_A', '',
         'common_carotid_R6_C', ''),
        ('common_carotid_R6_C',     'pv', 'split_junction',     'common_carotid_R6_B', '',
         'external_carotid_R26', 'internal_carotid_R8_A'),
        ('external_carotid_L62',    'pp', 'arterial',           'common_carotid_L48_D', '',
         'external_carotid_L_T', ''),
        ('external_carotid_R26',     'pp', 'arterial',           'common_carotid_R6_C',  '',
         'external_carotid_R_T', ''),
        ('internal_carotid_L50_A',  'pv', 'arterial',           'common_carotid_L48_D', '',
         'internal_carotid_L50_B', ''),
        ('internal_carotid_L50_B',  'pv', 'arterial',           'internal_carotid_L50_A', '',
         'internal_carotid_L50_C', ''),
        ('internal_carotid_L50_C',  'pv', 'split_junction',     'internal_carotid_L50_B', '',
         'internal_carotid_L112', 'posterior_communicating_L8'),
        ('internal_carotid_R8_A',   'pv', 'arterial',           'common_carotid_R6_C', '',
         'internal_carotid_R8_B', ''),
        ('internal_carotid_R8_B',   'pv', 'arterial',           'internal_carotid_R8_A', '',
         'internal_carotid_R8_C', ''),
        ('internal_carotid_R8_C',   'pv', 'split_junction',     'internal_carotid_R8_B', '',
         'internal_carotid_R48', 'posterior_communicating_R206'),
        ('posterior_communicating_L8', 'pp', 'arterial',        'internal_carotid_L50_C', '',
         'posterior_cerebral_post_L12', ''),
        ('posterior_communicating_R206','pp', 'arterial',       'internal_carotid_R8_C', '',
         'posterior_cerebral_post_R208', ''),
        ('internal_carotid_L112',   'pv', 'split_junction',     'internal_carotid_L50_C', '',
         'middle_cerebral_L114', 'anterior_cerebral_L110'),
        ('internal_carotid_R48',    'pv', 'split_junction',     'internal_carotid_R8_C', '',
         'middle_cerebral_R52', 'anterior_cerebral_R46'),
        ('middle_cerebral_L114',    'pp', 'arterial',           'internal_carotid_L112', '',
         'middle_cerebral_L_T',  ''),
        ('middle_cerebral_R52',     'pp', 'arterial',           'internal_carotid_R48', '',
         'middle_cerebral_R_T',  ''),
        ('anterior_cerebral_L110',  'pp', 'arterial',           'internal_carotid_L112', '',
         'anterior_cerebral_L42', ''),
        ('anterior_cerebral_R46',   'pv', 'split_junction',     'internal_carotid_R48', '',
         'anterior_cerebral_R238', 'anterior_communicating_C44'),
        ('anterior_cerebral_R238',  'pp', 'arterial',           'anterior_cerebral_R46', '',
         'anterior_cerebral_R_T', ''),
        ('anterior_cerebral_L42',   'vp', 'merge_junction',     'anterior_cerebral_L110', 'anterior_communicating_C44',
         'anterior_cerebral_L_T', ''),
        ('anterior_communicating_C44','pp', 'arterial',         'anterior_cerebral_R46', '',
         'anterior_cerebral_L42', ''),
        ('vertebral_L2',            'pp', 'arterial',           'subclavian_L66',       '',
         'basilar_C4',           ''),
        ('subclavian_R28',          'pv', 'split_junction',     'brachiocephalic_trunk_C4', '',
         'subclavian_R30',       'vertebral_R272'),
        ('subclavian_R30',          'pp', 'arterial',           'subclavian_R28',       '',
         'arm_R_T',              ''),
        ('vertebral_R272',          'pp', 'arterial',           'subclavian_R28',       '',
         'basilar_C4',           ''),
        ('basilar_C4',              'vv', '2in2out_junction',   'vertebral_R272',       'vertebral_L2',
         'posterior_cerebral_pre_L6', 'posterior_cerebral_pre_R204'),
        ('posterior_cerebral_pre_L6','pp','arterial',           'basilar_C4',           '',
         'posterior_cerebral_post_L12', ''),
        ('posterior_cerebral_post_L12','vp','merge_junction',   'posterior_cerebral_pre_L6', 'posterior_communicating_L8',
         'posterior_cerebral_L_T',    ''),
        ('posterior_cerebral_pre_R204','pp','arterial',         'basilar_C4',           '',
         'posterior_cerebral_post_R208', ''),
        ('posterior_cerebral_post_R208','vp','merge_junction',  'posterior_cerebral_pre_R204', 'posterior_communicating_R206',
         'posterior_cerebral_R_T',    ''),
        ('posterior_cerebral_L_T',  'vp', 'terminal',           'posterior_cerebral_post_L12', '',
         'venous_ub',            ''),
        ('posterior_cerebral_R_T',  'vp', 'terminal',           'posterior_cerebral_post_R208', '',
         'venous_ub',            ''),
        ('external_carotid_L_T',    'vp', 'terminal',           'external_carotid_L62', '',
         'venous_ub',            ''),
        ('external_carotid_R_T',    'vp', 'terminal',           'external_carotid_R26',  '',
         'venous_ub',            ''),
        ('middle_cerebral_L_T',     'vp', 'terminal',           'middle_cerebral_L114', '',
         'venous_ub',            ''),
        ('middle_cerebral_R_T',     'vp', 'terminal',           'middle_cerebral_R52',  '',
         'venous_ub',            ''),
        ('anterior_cerebral_L_T',   'vp', 'terminal',           'anterior_cerebral_L42', '',
         'venous_ub',            ''),
        ('anterior_cerebral_R_T',   'vp', 'terminal',           'anterior_cerebral_R238', '',
         'venous_ub',            ''),
        ('arm_L_T',                 'vp', 'terminal',           'subclavian_L78',       '',
         'venous_ub',            ''),
        ('arm_R_T',                 'vp', 'terminal',           'subclavian_R30',       '',
         'venous_ub',            ''),
        ('aortic_arch_C94',         'pv', 'arterial',           'aortic_arch_C64',      '',
         'thoracic_aorta_C96',   ''),
        ('thoracic_aorta_C96',      'pv', 'arterial',           'aortic_arch_C94',      '',
         'thoracic_aorta_C100',  ''),
        ('thoracic_aorta_C100',     'pv', 'arterial',           'thoracic_aorta_C96',   '',
         'thoracic_aorta_C104',  ''),
        ('thoracic_aorta_C104',     'pv', 'arterial',           'thoracic_aorta_C100',  '',
         'thoracic_aorta_C108',  ''),
        ('thoracic_aorta_C108',     'pv', 'arterial',           'thoracic_aorta_C104',  '',
         'thoracic_aorta_C112',  ''),
        ('thoracic_aorta_C112',     'pv', 'arterial',           'thoracic_aorta_C108',  '',
         'abdominal_aorta_C114', ''),
        ('abdominal_aorta_C114',    'pv', 'split_junction',     'thoracic_aorta_C112',  '',
         'abdominal_aorta_C136', 'celiac_trunk_C116'),
        ('celiac_trunk_C116',       'pp', 'arterial',           'abdominal_aorta_C114', '',
         'splanchnic_C_T',       ''),
        ('splanchnic_C_T',          'vp', 'terminal',           'celiac_trunk_C116',    '',
         'venous_lb',            ''),
        ('abdominal_aorta_C136',    'pv', 'arterial',           'abdominal_aorta_C114', '',
         'abdominal_aorta_C164', ''),
        ('abdominal_aorta_C164',    'pv', 'arterial',           'abdominal_aorta_C136', '',
         'abdominal_aorta_C176', ''),
        ('abdominal_aorta_C176',    'pv', 'arterial',           'abdominal_aorta_C164', '',
         'abdominal_aorta_C188', ''),
        ('abdominal_aorta_C188',    'pv', 'arterial',           'abdominal_aorta_C176',      '',
         'abdominal_aorta_C192', ''),
        ('abdominal_aorta_C192',    'pv', 'split_junction',     'abdominal_aorta_C188', '',
         'common_iliac_L194',    'common_iliac_R216'),
        ('common_iliac_L194',       'pv', 'arterial',           'abdominal_aorta_C192', '',
         'external_iliac_L198',  ''),
        ('external_iliac_L198',     'pv', 'arterial',           'common_iliac_L194',    '',
         'femoral_L200',         ''),
        ('common_iliac_R216',       'pv', 'arterial',           'abdominal_aorta_C192', '',
         'external_iliac_R220',  ''),
        ('external_iliac_R220',     'pv', 'arterial',           'common_iliac_R216',    '',
         'femoral_R222',         ''),
        ('femoral_L200',            'pp', 'arterial',           'external_iliac_L198',  '',
         'leg_L_T',              ''),
        ('femoral_R222',            'pp', 'arterial',           'external_iliac_R220',  '',
         'leg_R_T',              ''),
        ('leg_L_T',                 'vp', 'terminal',           'femoral_L200',         '',
         'venous_lb',            ''),
        ('leg_R_T',                 'vp', 'terminal',           'femoral_R222',         '',
         'venous_lb',            ''),
        ('venous_lb',               'vp', 'venous',             '',                     '',
         'venous_ivc',           ''),
        ('venous_ub',               'vp', 'venous',             '',                     '',
         'venous_svc',           ''),
        ('venous_ivc',              'vp', 'venous',             'venous_lb',            '',
         'heart',                ''),
        ('venous_svc',              'vp', 'venous',             'venous_ub',            '',
         'heart',                '')],
        dtype=[('name', 'U64'), ('BC_type', 'U64'), ('vessel_type', 'U64'), ('inp_vessel_1', 'U64'),
               ('inp_vessel_2', 'U64'), ('out_vessel_1', 'U64'), ('out_vessel_2', 'U64')])

    generate_model_cellml_from_array(vessel_array, case_name, parameters_array=parameters_array)
    generate_parameters_cellml_from_array(parameters_array)


