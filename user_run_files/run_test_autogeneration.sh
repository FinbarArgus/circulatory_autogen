# user inputs are defined in user_inputs.sh
source user_inputs.sh

echo Running Tests for autogeneration
echo _______________________________

echo Running 3compartment autogeneration test
echo _______________________________
file_prefix=3compartment
input_param_file=3compartment_parameters.csv # this must be stored in resources.
${opencor_pythonshell_path} ../src/scripts/script_generate_with_new_architecture.py ${file_prefix}_vessel_array.csv ${input_param_file} ${file_prefix}
echo 
echo _______________________________
echo Running simple_physiological autogeneration test
echo _______________________________
file_prefix=simple_physiological
input_param_file=simple_physiological_parameters.csv # this must be stored in resources.
${opencor_pythonshell_path} ../src/scripts/script_generate_with_new_architecture.py ${file_prefix}_vessel_array.csv ${input_param_file} ${file_prefix}
echo 
echo _______________________________
echo Running neonatal autogeneration test
echo _______________________________
file_prefix=neonatal
input_param_file=neonatal_parameters.csv # this must be stored in resources.
${opencor_pythonshell_path} ../src/scripts/script_generate_with_new_architecture.py ${file_prefix}_vessel_array.csv ${input_param_file} ${file_prefix}
echo 
echo _______________________________
echo Running FinalModel autogeneration test
echo _______________________________
file_prefix=FinalModel
input_param_file=FinalModel_parameters.csv # this must be stored in resources.
${opencor_pythonshell_path} ../src/scripts/script_generate_with_new_architecture.py ${file_prefix}_vessel_array.csv ${input_param_file} ${file_prefix}
echo 
echo _______________________________
echo Running physiological autogeneration with id params test
echo _______________________________
file_prefix=physiological
input_param_file=physiological_parameters.csv # this must be stored in resources.
param_id_obs_path=/home/finbar/Documents/data/heart_projects/Argus_2022/observables_biobeat_BB128.json 
${opencor_pythonshell_path} ../src/scripts/script_generate_with_new_architecture.py ${file_prefix}_vessel_array.csv ${input_param_file} ${file_prefix} ${param_id_method} ${param_id_obs_path}
echo 
echo _______________________________
echo Running control_phys autogeneration with id params test
echo _______________________________
file_prefix=control_phys
input_param_file=control_phys_parameters.csv # this must be stored in resources.
param_id_obs_path=/home/finbar/Documents/data/heart_projects/Argus_2022/observables_biobeat_BB128.json 
${opencor_pythonshell_path} ../src/scripts/script_generate_with_new_architecture.py ${file_prefix}_vessel_array.csv ${input_param_file} ${file_prefix} ${param_id_method} ${param_id_obs_path}
echo ______________________________
echo Running cerebral_elic autogeneration with id params test
echo _______________________________
file_prefix=cerebral_elic
input_param_file=cerebral_elic_parameters.csv # this must be stored in resources.
${opencor_pythonshell_path} ../src/scripts/script_generate_with_new_architecture.py ${file_prefix}_vessel_array.csv ${input_param_file} ${file_prefix}
echo ______________________________
echo Testing complete
echo ______________________________
