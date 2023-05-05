# user inputs are defined in user_inputs.sh
source opencor_pythonshell_path.sh
./run_autogeneration_with_id_params.sh
${opencor_pythonshell_path} ../src/scripts/plot_param_id_script.py

