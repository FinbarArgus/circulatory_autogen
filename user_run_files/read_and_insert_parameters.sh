# user inputs are defined in user_inputs.sh
source user_inputs.sh
${opencor_pythonshell_path} ../src/scripts/read_and_insert_parameters.py ${input_param_file} $1
