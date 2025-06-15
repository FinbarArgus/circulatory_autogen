import os
import sys
import yaml
import traceback

root_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir, 'src'))

user_inputs_dir = os.path.join(root_dir, 'user_run_files')
from scripts.script_generate_with_new_architecture import generate_with_new_architecture
from parsers.PrimitiveParsers import YamlFileParser

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

        gen_success_list = []
    
        print('running port tests autogeneration test')
        inp_data_dict['file_prefix'] = 'ports_test'
        inp_data_dict['input_param_file'] = 'ports_test_parameters.csv'
        success = generate_with_new_architecture(False, inp_data_dict)
        gen_success_list.append(('port_test', success))

        print('')
        print('running 3compartment autogeneration test')
        inp_data_dict['file_prefix'] = '3compartment'
        inp_data_dict['input_param_file'] = '3compartment_parameters.csv'
        success = generate_with_new_architecture(False, inp_data_dict)
        gen_success_list.append(('3compartment', success))

        print('')
        print('running simple_physiological autogeneration test')
        inp_data_dict['file_prefix'] = 'simple_physiological'
        inp_data_dict['input_param_file'] = 'simple_physiological_parameters.csv'
        success = generate_with_new_architecture(False, inp_data_dict)
        gen_success_list.append(('simple_physiological', success))
        
        print('')
        print('running test_fft autogeneration test')
        inp_data_dict['file_prefix'] = 'test_fft'
        inp_data_dict['input_param_file'] = 'test_fft_parameters.csv'
        success = generate_with_new_architecture(False, inp_data_dict)
        gen_success_list.append(('test_fft', success))

        print('')
        print('running neonatal autogeneration test')
        inp_data_dict['file_prefix'] = 'neonatal'
        inp_data_dict['input_param_file'] = 'neonatal_parameters.csv'
        success = generate_with_new_architecture(False, inp_data_dict)
        gen_success_list.append(('neonatal', success))
        
        print('')
        print('running generic junction closed loop autogeneration test')
        inp_data_dict['file_prefix'] = 'generic_junction_test_closed_loop'
        inp_data_dict['input_param_file'] = 'generic_junction_test_closed_loop_parameters.csv'
        success = generate_with_new_architecture(False, inp_data_dict)
        gen_success_list.append(('generic_junction_test_closed_loop', success))
        
        print('')
        print('running generic junction closed loop 2 autogeneration test')
        inp_data_dict['file_prefix'] = 'generic_junction_test2_closed_loop'
        inp_data_dict['input_param_file'] = 'generic_junction_test_closed_loop_parameters.csv'
        success = generate_with_new_architecture(False, inp_data_dict)
        gen_success_list.append(('generic_junction_test2_closed_loop', success))

        print('')
        print('running generic junction open loop autogeneration test')
        inp_data_dict['file_prefix'] = 'generic_junction_test_open_loop'
        inp_data_dict['input_param_file'] = 'generic_junction_test_open_loop_parameters.csv'
        success = generate_with_new_architecture(False, inp_data_dict)
        gen_success_list.append(('generic_junction_test_open_loop', success))

        print('')
        print('running generic junction open loop 2 autogeneration test')
        inp_data_dict['file_prefix'] = 'generic_junction_test2_open_loop'
        inp_data_dict['input_param_file'] = 'generic_junction_test_open_loop_parameters.csv'
        success = generate_with_new_architecture(False, inp_data_dict)  
        gen_success_list.append(('generic_junction_test2_open_loop', success))

        # commenting out because it is very slow
        # print('')
        # print('running FinalModel autogeneration test')
        # inp_data_dict['file_prefix'] = 'FinalModel'
        # inp_data_dict['input_param_file'] = 'FinalModel_parameters.csv'
        # success = generate_with_new_architecture(False, inp_data_dict)
        # gen_success_list.append(('FinalModel', success))

        # temporarily commented out because it is so slow.
        # print('')
        # print('running cerebral_elic autogeneration test')
        # inp_data_dict['file_prefix'] = 'cerebral_elic'
        # inp_data_dict['input_param_file'] = 'cerebral_elic_parameters.csv'
        # success = generate_with_new_architecture(False, inp_data_dict)
        # gen_success_list.append(('cerebral_elic', success))
        
        print('')
        print('running sympathetic neuron (SN_simple) test')
        inp_data_dict['file_prefix'] = 'SN_simple'
        inp_data_dict['input_param_file'] = 'SN_simple_parameters.csv'
        success = generate_with_new_architecture(False, inp_data_dict)
        gen_success_list.append(('SN_simple', success))

        print('')
        print('running physiological autogeneration test')
        inp_data_dict['file_prefix'] = 'physiological'
        inp_data_dict['input_param_file'] = 'physiological_parameters.csv'
        # TODO currently we don't test the "True" argument below,
        # which runs autogeneration with parameters identified from
        # param_id. This is becuase we don't have identified parameters in
        # The repo... Include this test in the param_id tests after running them.
        success = generate_with_new_architecture(False, inp_data_dict)
        gen_success_list.append(('physiological', success))

        print('')
        print('running control_phys autogeneration test')
        inp_data_dict['file_prefix'] = 'control_phys'
        inp_data_dict['input_param_file'] = 'control_phys_parameters.csv'
        success = generate_with_new_architecture(False, inp_data_dict)
        gen_success_list.append(('control_phys', success))
        
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
        success = generate_with_new_architecture(False, inp_data_dict)
        gen_success_list.append(('aortic_bif_1d_cpp', success))

        print("standard autogeneration tests complete. ",
              "Checking to see if CA_users directory exists for extra tests.")
        # Now test the CA_users directory
        CA_user_dir = os.path.join(root_dir, '..', 'CA_user')
        if os.path.exists(CA_user_dir):
            print('CA_users directory exists, running CA_users autogeneration tests')

            print('')
            print('Running Sympathetic neuron test in CA_users directory')
            yaml_file_parser = YamlFileParser()
            # TODO move this to the circulatory autogen directory when this model is published.
            with open(os.path.join(CA_user_dir, 'SN_simple', 'SN_simple_user_inputs.yaml'), 'r') as file:
                inp_data_dict = yaml.load(file, Loader=yaml.FullLoader)
                inp_data_dict['resources_dir'] = os.path.join(CA_user_dir, 'SN_simple', inp_data_dict['resources_dir'])
            success = generate_with_new_architecture(False, inp_data_dict)
            gen_success_list.append(('CA_user_SN_simple', success))
        else:
            print('CA_users directory does not exist, skipping CA_users autogeneration tests')

        print("autogeneration tests complete. Printing results:")
        for model_name, success in gen_success_list:
            if success:
                print(f"{model_name} autogeneration test passed")
            else:
                print(f"{model_name} autogeneration test failed")

        

    except:
        print(traceback.format_exc())
        exit()
