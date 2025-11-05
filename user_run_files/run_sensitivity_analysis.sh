#!/bin/bash

# Source the path
source opencor_pythonshell_path.sh

echo "Running sensitivity analysis test with $1 processors"

# Run the autogeneration script
./run_autogeneration.sh

# Check the exit status of the previous command
if [ $? -eq 0 ]; then
  echo "Autogeneration completed successfully."
  
  # If successful, proceed with the mpirun command
  mpirun -n "$1" "${opencor_pythonshell_path}" ../src/scripts/sensitivity_analysis_run_script.py
  
else
  echo "Error: Autogeneration failed. Aborting."
  exit 1
fi
