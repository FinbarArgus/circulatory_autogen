#!/bin/bash
# Script to run pytest using OpenCOR's Python shell
# This ensures tests run with the correct Python environment that includes OpenCOR

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Source the OpenCOR Python shell path
source user_run_files/opencor_pythonshell_path.sh

# Check if pytest is installed in the OpenCOR Python environment
if ! ${opencor_pythonshell_path} -m pytest --version > /dev/null 2>&1; then
    echo "pytest is not installed in the OpenCOR Python environment."
    echo "Installing pytest and related packages from pyproject.toml..."
    # Try to install from pyproject.toml dev dependencies first
    if [ -f "pyproject.toml" ]; then
        ${opencor_pythonshell_path} -m pip install -e ".[dev]"
    else
        # Fallback to direct installation
        ${opencor_pythonshell_path} -m pip install pytest pytest-cov pytest-mpi pytest-xdist
    fi
fi

# Run pytest with the OpenCOR Python shell
# Pass all arguments to pytest
${opencor_pythonshell_path} -m pytest "$@"

