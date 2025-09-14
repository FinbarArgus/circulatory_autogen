source opencor_pythonshell_path.sh
echo "Running sensitivity analysis test with $1 processors"
mpirun -n $1 ${opencor_pythonshell_path} ../src/scripts/run_sensitivity_analysis_test.py 

# ${opencor_pythonshell_path} ../src/scripts/run_sensitivity_analysis_test.py