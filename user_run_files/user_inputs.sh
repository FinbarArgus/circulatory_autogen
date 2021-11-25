## general inputs
file_prefix=simple_physiological
input_param_file=parameters_orig.csv # this must be stored in resources.
                                     # If first creating a model
                                     # set this to parameters_orig.csv and a 
                                     # parameters file will be generated with 
                                     # spaces for the required parameters

## parameter identification inputs
## param_id_method can be any of [genetic_algorithm, bayesian]
param_id_method=genetic_algorithm
num_procs=3
num_calls_to_function=500 
param_id_output_dir=../generated_models/ # path is from the user_run_files dir
param_id_obs_file=simple_physiological_obs_data.json
input_params_to_id=True # Whether an input file for params to identify is given in resources
                        # This should be named {file_prefix}_params_to_id.csv
                        # If false the params to identify will be chosen automatically.


## paths
## the below for my hpc
# opencor_pythonshell_path=/people/farg967/.local/share/OpenCOR-2021-07-04-Linux/pythonshell
## the below for my local
opencor_pythonshell_path=/opt/OpenCOR-2021-10-05-Linux/pythonshell
## Users should modify opencor_pythonshell_path to the path of their own opencor pythonshell

