import ruamel.yaml
import sys
import os

def change_patient_num(patient_num, case_type, project_dir, period=1):
    yaml = ruamel.yaml.YAML()

    user_run_files_dir = os.path.dirname(__file__)
    with open(os.path.join(user_run_files_dir, 'user_inputs.yaml'), 'r') as file:
        # data = yaml.safe_load(file)
        inp_data_dict = yaml.load(file)
    
    if "user_inputs_path_override" in inp_data_dict.keys():
        user_run_files_path = inp_data_dict["user_inputs_path_override"]
        with open(inp_data_dict["user_inputs_path_override"], 'r') as file:
            inp_data_dict = yaml.load(file)
    else:
        user_run_files_path = os.path.join(user_run_files_dir, 'user_inputs.yaml')

    inp_data_dict_orig = inp_data_dict.copy()
    # if case_type in ['pre', 'coupled_pre']:
    #     pre_or_post = 'pre'
    # else:
    #     pre_or_post = 'post'

    if case_type in ['pre', 'post']:
        inp_data_dict['pre_time'] = 20*period
        inp_data_dict['sim_time'] = period
    else:
        inp_data_dict['pre_time'] = 40*period
        inp_data_dict['sim_time'] = 2*period

    inp_data_dict['resources_dir'] = os.path.join(project_dir, f"physiology_models/pulmonary_CVS_Alfred/patient_{patient_num}/{case_type}/resources")
    inp_data_dict['generated_models_dir'] = os.path.join(project_dir, f"physiology_models/pulmonary_CVS_Alfred/patient_{patient_num}/{case_type}/generated_models")
    inp_data_dict['param_id_output_dir'] = os.path.join(project_dir, f"physiology_models/pulmonary_CVS_Alfred/patient_{patient_num}/{case_type}/param_id_output")

    if case_type == 'pre':
        inp_data_dict['file_prefix'] = 'lung_ROM'
        inp_data_dict['input_param_file'] = 'lung_ROM_parameters.csv'
        inp_data_dict['param_id_obs_path'] = os.path.join(project_dir, f"data/pulmonary/ground_truth_for_CA/ROM_gt/lung_ROM_lobe_imped_pre_patient_{patient_num}_obs_data.json")
    elif case_type == 'post':
        inp_data_dict['file_prefix'] = 'lung_ROM'
        inp_data_dict['input_param_file'] = 'lung_ROM_parameters.csv'
        inp_data_dict['param_id_obs_path'] = os.path.join(project_dir, f"data/pulmonary/ground_truth_for_CA/ROM_gt/lung_ROM_lobe_imped_post_patient_{patient_num}_obs_data.json")
    elif case_type == 'coupled_pre':
        inp_data_dict['file_prefix'] = 'lung_CVS_coupled'
        inp_data_dict['input_param_file'] = 'lung_CVS_coupled_parameters.csv'
        inp_data_dict['param_id_obs_path'] = os.path.join(project_dir, f"data/pulmonary/ground_truth_for_CA/ground_truth_Alfred_pre_observables_{patient_num}.json")
    elif case_type == 'coupled_post':
        inp_data_dict['file_prefix'] = 'lung_CVS_coupled'
        inp_data_dict['input_param_file'] = 'lung_CVS_coupled_parameters.csv'
        inp_data_dict['param_id_obs_path'] = f"" # this model doesn't get calibrated to anything

    try:
        with open(user_run_files_path, 'w') as wf:
            yaml.dump(inp_data_dict, wf)
    except Exception as e:
        print(f"Error writing to {user_run_files_path}")
        print(e)
        with open(user_run_files_path, 'w') as wf:
            yaml.dump(inp_data_dict_orig, wf)
        print("Restored original user_inputs.yaml")
        exit()

if __name__ == "__main__":
    
    if len(sys.argv) == 4:
        patient_num=sys.argv[1]
        case_type=sys.argv[2]
        project_dir = sys.argv[3]
        if case_type not in ['pre', 'post', 'coupled_pre', 'coupled_post']:
            print(f'case type must be in [pre, post, coupled_pre, coupled_post], not {case_type}')
            exit()
        change_patient_num(patient_num, case_type, project_dir)
    else:
        print("usage:  python change_patient_num.py patient_num case_type project_dir") 
        exit()
