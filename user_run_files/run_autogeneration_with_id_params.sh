# user inputs are defined in user_inputs.sh
source user_inputs.sh
${opencor_pythonshell_path} ../src/scripts/script_generate_with_new_architecture.py ${file_prefix}_vessel_array.csv ${input_param_file} ${file_prefix} ${param_id_method} ${param_id_obs_path}
