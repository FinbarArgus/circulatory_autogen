# $1 is the number of processors
# $2 is the max number of generations

mpiexec -n $1 /opt/OpenCOR-2021-01-13-Linux/pythonshell ../src/scripts/param_id_run_script.py genetic_algorithm physiological $2

