## general inputs
file_prefix=simple_physiological
input_param_file=parameters_orig.csv # this must be stored in resources

## parameter identification inputs
param_id_method=bayesian
num_procs=10 
num_calls_to_function=100
param_id_output_dir=../generated_models/ # path is from the user_run_files dir

## paths
## the below for my hpc
# opencor_pythonshell_path=/people/farg967/.local/share/OpenCOR-2021-07-04-Linux/pythonshell
## the below for my local
opencor_pythonshell_path=/opt/OpenCOR-2021-10-05-Linux/pythonshell
## Users should modify opencor_pythonshell_path to the path of their own opencor pythonshell

