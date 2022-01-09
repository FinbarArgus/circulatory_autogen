# user inputs are defined in user_inputs.sh
source user_inputs.sh
${opencor_pythonshell_path} ../src/scripts/plot_sensitivity_script.py ${param_id_method} ${file_prefix} ${param_id_obs_path}

