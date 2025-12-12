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

# Check if -n flag is used (pytest-xdist parallel execution)
# If so, ensure pytest-xdist is installed and configure it to use OpenCOR Python
PYTEST_ARGS=()
HAS_N_FLAG=false
NUM_WORKERS=""

# Parse arguments and handle -n flag specially
while [[ $# -gt 0 ]]; do
    case $1 in
        -n)
            HAS_N_FLAG=true
            shift
            if [[ $# -gt 0 ]]; then
                NUM_WORKERS="$1"
                shift
            else
                NUM_WORKERS="auto"
            fi
            ;;
        *)
            PYTEST_ARGS+=("$1")
            shift
            ;;
    esac
done

# If -n flag was used, configure pytest-xdist to use OpenCOR Python
if [ "$HAS_N_FLAG" = true ]; then
    if ! ${opencor_pythonshell_path} -m pip show pytest-xdist > /dev/null 2>&1; then
        echo "pytest-xdist is required for parallel execution (-n flag). Installing..."
        ${opencor_pythonshell_path} -m pip install pytest-xdist
    fi
    
    # Determine number of workers
    if [ "$NUM_WORKERS" = "auto" ]; then
        # Use CPU count for auto
        if command -v nproc > /dev/null 2>&1; then
            NUM_WORKERS=$(nproc)
        else
            NUM_WORKERS=4  # Fallback
        fi
    fi
    
    # Replace -n with explicit --dist and --tx options
    # Use loadgroup so xdist honors xdist_group marks (serial groups)
    PYTEST_ARGS+=("--dist=loadgroup")
    for ((i=0; i<NUM_WORKERS; i++)); do
        PYTEST_ARGS+=("--tx" "popen//python=${opencor_pythonshell_path}")
    done
fi

# Run pytest with the OpenCOR Python shell
# Pass all arguments to pytest and tee output to a log file
LOG_FILE="${SCRIPT_DIR}/log_test.log"
touch "$LOG_FILE"
echo "===== pytest run $(date -Iseconds) =====" | tee -a "$LOG_FILE"
"${opencor_pythonshell_path}" -m pytest "${PYTEST_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
exit "${PIPESTATUS[0]}"

