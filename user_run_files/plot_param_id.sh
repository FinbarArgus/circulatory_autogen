# user inputs are defined in user_inputs.sh
source user_inputs.sh
./run_autogeneration.sh
${opencor_pythonshell_path} ../src/scripts/plot_param_id_script.py

