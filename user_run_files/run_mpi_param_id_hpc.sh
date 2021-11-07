# $1 is the number of processors
# $2 is the max number of generations
mpiexec -n $1 /people/farg967/.local/share/OpenCOR-2021-07-04-Linux/pythonshell ../src/scripts/param_id_run_script.py genetic_algorithm physiological $2 > ../param_id_output/log.txt

