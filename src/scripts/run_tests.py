import os
import sys
import yaml
import traceback

root_dir_path = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir_path, 'src'))

user_inputs_path = os.path.join(root_dir_path, 'user_run_files')
from scripts.script_generate_with_new_architecture import generate_with_new_architecture

if __name__ == '__main__':
    try:
        with open(os.path.join(user_inputs_path, 'user_inputs.yaml'), 'r') as file:
            inp_data_dict = yaml.load(file, Loader=yaml.FullLoader)

        print('_________Running all autogeneration tests_____________')
        print('')

        print('running 3compartment autogeneration test')
        inp_data_dict['file_prefix'] = '3compartment'
        inp_data_dict['input_param_file'] = '3compartment_parameters.csv'
        generate_with_new_architecture(False, inp_data_dict)

        print('')
        print('running simple_physiological autogeneration test')
        inp_data_dict['file_prefix'] = 'simple_physiological'
        inp_data_dict['input_param_file'] = 'simple_physiological_parameters.csv'
        generate_with_new_architecture(False, inp_data_dict)

        print('')
        print('running neonatal autogeneration test')
        inp_data_dict['file_prefix'] = 'neonatal'
        inp_data_dict['input_param_file'] = 'neonatal_parameters.csv'
        generate_with_new_architecture(False, inp_data_dict)

        print('')
        print('running FinalModel autogeneration test')
        inp_data_dict['file_prefix'] = 'FinalModel'
        inp_data_dict['input_param_file'] = 'FinalModel_parameters.csv'
        generate_with_new_architecture(False, inp_data_dict)

        print('')
        print('running cerebral_elic autogeneration test')
        inp_data_dict['file_prefix'] = 'cerebral_elic'
        inp_data_dict['input_param_file'] = 'cerebral_elic_parameters.csv'
        generate_with_new_architecture(False, inp_data_dict)

        print('')
        print('running physiological autogeneration test')
        inp_data_dict['file_prefix'] = 'physiological'
        inp_data_dict['input_param_file'] = 'physiological_parameters.csv'
        # Only the file name is taken here, the path doesnt need to exist
        inp_data_dict['param_id_obs_path'] = '/this/dir/path/isnt/important/observables_biobeat_BB128.json'
        generate_with_new_architecture(True, inp_data_dict)

        print('')
        print('running control_phys autogeneration test')
        inp_data_dict['file_prefix'] = 'control_phys'
        inp_data_dict['input_param_file'] = 'control_phys_parameters.csv'
        # Only the file name is taken here, the path doesnt need to exist
        inp_data_dict['param_id_obs_path'] = '/this/dir/path/isnt/important/observables_biobeat_BB128.json'
        generate_with_new_architecture(False, inp_data_dict)

    except:
        print(traceback.format_exc())
        exit()
