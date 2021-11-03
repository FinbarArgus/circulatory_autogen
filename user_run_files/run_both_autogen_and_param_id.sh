/opt/OpenCOR-2021-01-13-Linux/pythonshell ../src/scripts/script_generate_with_new_architecture.py simple_physiological_vessel_array.csv parameters_orig.csv ../generated_models/ simple_physiological

# $1 is the number of processors
# $2 is the max number of generations

mpiexec -n $1 /opt/OpenCOR-2021-01-13-Linux/pythonshell ../src/scripts/param_id_run_script.py genetic_algorithm simple_physiological $2

/opt/OpenCOR-2021-01-13-Linux/pythonshell ../src/scripts/script_generate_with_new_architecture.py simple_physiological_vessel_array.csv parameters_orig.csv ../generated_models/ simple_physiological genetic_algorithm_simple_physiological
