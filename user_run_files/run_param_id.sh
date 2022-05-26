# user inputs are defined in user_inputs.sh
source user_inputs.sh
mpiexec -n ${num_procs} ${opencor_pythonshell_path} ../src/scripts/param_id_run_script.py ${param_id_method} ${file_prefix} ${num_calls_to_function} ${param_id_obs_path} ${do_mcmc}

