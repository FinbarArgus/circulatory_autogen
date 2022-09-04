if [[ $# -eq 0 ]] ; then
    echo 'usage is ./run_param_id.sh num_processors'
    exit 1
fi
source opencor_pythonshell_path.sh
mpiexec -n $1 ${opencor_pythonshell_path} ../src/scripts/run_multiple_param_id.py

