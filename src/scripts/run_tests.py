import os
import sys
import yaml
import traceback

root_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir, 'src'))

user_inputs_dir = os.path.join(root_dir, 'user_run_files')
from scripts.script_generate_with_new_architecture import generate_with_new_architecture

if __name__ == '__main__':
    try:
        with open(os.path.join(user_inputs_dir, 'user_inputs.yaml'), 'r') as file:
            inp_data_dict = yaml.load(file, Loader=yaml.FullLoader)

        # remove user_input entries so they aren't passed to the generation script,
        # this ensures the default dirs are used
        if 'user_inputs_path_override' in inp_data_dict.keys():
            del inp_data_dict['user_inputs_path_override']
        if 'resources_dir' in inp_data_dict.keys():
            del inp_data_dict['resources_dir']
        if 'generated_models_dir' in inp_data_dict.keys():
            del inp_data_dict['generated_models_dir']
        if 'param_id_output_dir' in inp_data_dict.keys():
            del inp_data_dict['param_id_output_dir']

        # test cellml creation
        inp_data_dict['model_type'] = 'cellml_only'
        inp_data_dict['solver'] = 'CVODE'

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
        print('running test_fft autogeneration test')
        inp_data_dict['file_prefix'] = 'test_fft'
        inp_data_dict['input_param_file'] = 'test_fft_parameters.csv'
        generate_with_new_architecture(False, inp_data_dict)

        print('')
        print('running neonatal autogeneration test')
        inp_data_dict['file_prefix'] = 'neonatal'
        inp_data_dict['input_param_file'] = 'neonatal_parameters.csv'
        generate_with_new_architecture(False, inp_data_dict)
        
        print('')
        print('running Sympathetic Neuron autogeneration test')
        inp_data_dict['file_prefix'] = 'SN_to_cAMP'
        inp_data_dict['input_param_file'] = 'SN_to_cAMP_parameters.csv'
        generate_with_new_architecture(False, inp_data_dict)

        # commenting out because it is very slow
        # print('')
        # print('running FinalModel autogeneration test')
        # inp_data_dict['file_prefix'] = 'FinalModel'
        # inp_data_dict['input_param_file'] = 'FinalModel_parameters.csv'
        # generate_with_new_architecture(False, inp_data_dict)

        # temporarily commented out because it is so slow.
        # print('')
        # print('running cerebral_elic autogeneration test')
        # inp_data_dict['file_prefix'] = 'cerebral_elic'
        # inp_data_dict['input_param_file'] = 'cerebral_elic_parameters.csv'
        # generate_with_new_architecture(False, inp_data_dict)

        print('')
        print('running physiological autogeneration test')
        inp_data_dict['file_prefix'] = 'physiological'
        inp_data_dict['input_param_file'] = 'physiological_parameters.csv'
        # TODO currently we don't test the "True" argument below,
        # which runs autogeneration with parameters identified from
        # param_id. This is becuase we don't have identified parameters in
        # The repo... Include this test in the param_id tests after running them.
        generate_with_new_architecture(False, inp_data_dict)

        print('')
        print('running control_phys autogeneration test')
        inp_data_dict['file_prefix'] = 'control_phys'
        inp_data_dict['input_param_file'] = 'control_phys_parameters.csv'
        generate_with_new_architecture(False, inp_data_dict)
        
        # Now test cpp creation. TODO I should also check the cpp models
        # actually run
        inp_data_dict['model_type'] = 'cpp'
        inp_data_dict['solver'] = 'RK4'
        inp_data_dict['couple_to_1d'] = True
        
        print('')
        print('running aortic_bif_1d cpp autogeneration test')
        inp_data_dict['file_prefix'] = 'aortic_bif_1d'
        inp_data_dict['input_param_file'] = 'aortic_bif_1d_parameters.csv'
        inp_data_dict['cpp_generated_models_dir'] = '/tmp'
        inp_data_dict['cpp_1d_model_config_path'] = None # TODO this isn't used yet.
        generate_with_new_architecture(False, inp_data_dict)

        print("autogeneration tests complete. Check above to see",
              "if they were succesful.")

    except:
        print(traceback.format_exc())
        exit()
