import numpy as np
import os
import re
from generation_helper import *

def main():
    case = 'test'
    base_script_path = 'base_script.cellml'
    new_file = f'autogen_{case}.cellml'

    # create numpy array with information of each circulatory vessel
    # make sure the first entry is the output to the left ventricle
    # entries are given by the following
    # [vessel_name,
    # BC type ('vv', 'vp', 'pv', 'pp'),
    # vessel type ('arterial', 'venous', 'terminal', 'split_junction', 'merge_junction'),
    # input vessel name (This doesn't need to be specified for venous type
    # input vessel name 2 if merge junction, '' otherwise
    # output vessel name
    # output vessel name 2 if split or merge junction, '' otherwise
    # TODO create converging junction? atm this is done by summing flows
    #  in the Systemic component then using the summed flow as the input

    # TODO I think I will need a "connecting" element for when there is
    #  a connected chain, like in the circle of willis. Do I try to automate
    #   this so the user doesn't need to choose the connecting element?
    possible_vessel_types = ['heart', 'arterial', 'split_junction', 'merge_junction',
                             'terminal', 'venous']

    # TODO The connections atm, are not physiological, they are just chosen to test the code
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
        ('common_carotid_L48_A',    'pp', 'arterial',           'aortic_arch_C46',      '',
                                                                'head_L_T',             ''),
        ('common_carotid_R6_A',     'pp', 'arterial',           'brachiocephalic_trunk_C4', '',
                                                                'head_R_T',             ''),
        ('subclavian_L66',          'pp', 'arterial',           'aortic_arch_C64',      '',
                                                                'arm_L_T',              ''),
        ('subclavian_R28',          'pp', 'arterial',           'brachiocephalic_trunk_C4', '',
                                                                'arm_R_T',              ''),
        ('head_L_T',                'vp', 'terminal',           'common_carotid_L48_A', '',
                                                                'venous_ub',            ''),
        ('head_R_T',                'vp', 'terminal',           'common_carotid_R6_A',  '',
                                                                'venous_ub',            ''),
        ('arm_L_T',                 'vp', 'terminal',           'subclavian_L66',       '',
                                                                'venous_ub',            ''),
        ('arm_R_T',                 'vp', 'terminal',           'subclavian_R28',       '',
                                                                'venous_ub',            ''),
        ('aortic_arch_C94',         'pv', 'arterial',           'aortic_arch_C64',      '',
                                                                'abdominal_aorta_C188', ''),
        ('abdominal_aorta_C188',    'pv', 'arterial',           'aortic_arch_C94',      '',
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
            write_parameter_mappings(wf, vessel_array)

            # map environment time to module times
            print('writing writing time mappings between environment and modules')
            write_section_break(wf, 'time mapping')
            write_time_mappings(wf, vessel_array)

            # Finalise the file
            wf.write('</model>\n')

if __name__ == '__main__':
    main()
