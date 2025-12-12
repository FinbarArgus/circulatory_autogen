"""
Pytest configuration file for circulatory_autogen tests.

This file sets up the test environment to work with OpenCOR's Python shell.
It provides fixtures for test data, configuration, and deterministic randomness.
"""
import os
import sys
import yaml
import tempfile
import shutil
import pytest
import numpy as np
import random

# Store pytest config for hooks that need plugin access (xdist reports lack config)
_PYTEST_CONFIG = None


def pytest_configure(config):
    """
    Configure pytest to work with OpenCOR Python environment.
    This ensures the src directory is in the Python path for imports.
    """
    global _PYTEST_CONFIG
    _PYTEST_CONFIG = config

    # Add src directory to path if not already there
    root_dir = os.path.join(os.path.dirname(__file__), '..')
    src_dir = os.path.join(root_dir, 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def pytest_runtest_logreport(report):
    """
    Ensure captured stdout from test_compare_optimisers is shown even when tests pass.
    Some runners/plugins keep capture enabled; this surfaces the comparison output.
    """
    if report.when != "call":
        return
    if report.passed and "test_compare_optimisers" in report.nodeid:
        # report in xdist may not carry config; use stored config instead
        global _PYTEST_CONFIG
        if _PYTEST_CONFIG is None:
            return
        tr = _PYTEST_CONFIG.pluginmanager.get_plugin("terminalreporter")
        if not tr:
            return

        # Debug marker to confirm hook execution
        tr.write_line("\n=== Comparison hook executed ===", yellow=True)

        # Prefer captured stdout if available
        cap = getattr(report, "capstdout", "")
        if cap:
            tr.write_line("\n=== Comparison output (captured) ===", yellow=True)
            tr.write(cap)

        # Always attempt to read the persisted comparison output file if present
        try:
            # temp_output_dir is unique per test instance; collect from report.user_properties if set
            out_file = None
            for name, value in getattr(report, "user_properties", []):
                if name == "comparison_output_file":
                    out_file = value
                    break
            if out_file and os.path.exists(out_file):
                with open(out_file, "r") as f:
                    contents = f.read()
                if contents:
                    tr.write_line("\n=== Comparison output (file) ===", yellow=True)
                    tr.write(contents)
        except Exception:
            # Do not fail the test if reading the file fails
            pass

@pytest.fixture(scope="session")
def project_root():
    """Fixture that returns the project root directory."""
    return os.path.join(os.path.dirname(__file__), '..')


@pytest.fixture(scope="session")
def opencor_python_path():
    """
    Fixture that attempts to read the OpenCOR Python shell path.
    Returns None if the path cannot be determined.
    """
    root_dir = os.path.join(os.path.dirname(__file__), '..')
    opencor_path_file = os.path.join(root_dir, 'user_run_files', 'opencor_pythonshell_path.sh')
    
    if os.path.exists(opencor_path_file):
        try:
            # Read the shell script and extract the path
            with open(opencor_path_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if line and not line.startswith('#') and 'opencor_pythonshell_path=' in line:
                        # Extract the path value
                        path = line.split('opencor_pythonshell_path=', 1)[1].strip()
                        # Remove quotes if present
                        path = path.strip('"\'')
                        if os.path.exists(path):
                            return path
        except Exception as e:
            pytest.skip(f"Could not read OpenCOR Python path: {e}")
    
    return None


@pytest.fixture(scope="session")
def user_inputs_dir(project_root):
    """Fixture that returns the user inputs directory."""
    return os.path.join(project_root, 'user_run_files')


@pytest.fixture(scope="session")
def resources_dir(project_root):
    """Fixture that returns the resources directory."""
    return os.path.join(project_root, 'resources')


@pytest.fixture(scope="function")
def base_user_inputs(user_inputs_dir):
    """
    Fixture that loads and returns base user inputs configuration.
    Removes path overrides to use default directories.
    """
    user_inputs_path = os.path.join(user_inputs_dir, 'user_inputs.yaml')
    
    if not os.path.exists(user_inputs_path):
        pytest.skip(f"User inputs file not found: {user_inputs_path}")
    
    with open(user_inputs_path, 'r') as file:
        inp_data_dict = yaml.load(file, Loader=yaml.FullLoader)
    
    # Remove user_input entries so they aren't passed to the generation script,
    # this ensures the default dirs are used
    for key in ['user_inputs_path_override', 'resources_dir', 'generated_models_dir', 'param_id_output_dir']:
        if key in inp_data_dict:
            del inp_data_dict[key]
    
    return inp_data_dict


@pytest.fixture(scope="function")
def temp_output_dir():
    """
    Fixture that creates a temporary directory for test outputs.
    Automatically cleans up after the test.
    """
    temp_dir = tempfile.mkdtemp(prefix='circulatory_autogen_test_')
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function", autouse=True)
def set_random_seed():
    """
    Fixture that sets random seeds for deterministic tests.
    Applied automatically to all tests.
    """
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    yield
    # Reset seeds after test (though not strictly necessary)


@pytest.fixture(scope="function")
def test_model_configs():
    """
    Fixture that returns test model configurations for parametrized tests.
    Returns a list of tuples: (file_prefix, input_param_file, model_type, solver)
    """
    return [
        # CellML models
        ('ports_test', 'ports_test_parameters.csv', 'cellml_only', 'CVODE'),
        ('3compartment', '3compartment_parameters.csv', 'cellml_only', 'CVODE'),
        ('simple_physiological', 'simple_physiological_parameters.csv', 'cellml_only', 'CVODE'),
        ('parasympathetic_model', 'parasympathetic_model_parameters.csv', 'cellml_only', 'CVODE'),
        ('test_fft', 'test_fft_parameters.csv', 'cellml_only', 'CVODE'),
        ('neonatal', 'neonatal_parameters.csv', 'cellml_only', 'CVODE'),
        ('generic_junction_test_closed_loop', 'generic_junction_test_closed_loop_parameters.csv', 'cellml_only', 'CVODE'),
        ('generic_junction_test2_closed_loop', 'generic_junction_test_closed_loop_parameters.csv', 'cellml_only', 'CVODE'),
        ('generic_junction_test_open_loop', 'generic_junction_test_open_loop_parameters.csv', 'cellml_only', 'CVODE'),
        ('generic_junction_test2_open_loop', 'generic_junction_test_open_loop_parameters.csv', 'cellml_only', 'CVODE'),
        ('SN_simple', 'SN_simple_parameters.csv', 'cellml_only', 'CVODE'),
        ('physiological', 'physiological_parameters.csv', 'cellml_only', 'CVODE'),
        ('control_phys', 'control_phys_parameters.csv', 'cellml_only', 'CVODE'),
        # CPP models
        ('aortic_bif_1d', 'aortic_bif_1d_parameters.csv', 'cpp', 'RK4'),
    ]


def pytest_collection_modifyitems(items):
    """
    Ensure autogeneration tests run before param_id tests, which in turn run before others.
    This is useful when running the full suite so that generated assets exist before
    parameter ID tests execute.
    """
    import os
    autogen_items = [item for item in items if "test_autogeneration" in item.nodeid]
    os.environ["AUTOGEN_TOTAL"] = str(len(autogen_items))

    def sort_key(item):
        nodeid = item.nodeid
        # Highest priority: autogeneration tests
        if "test_autogeneration" in nodeid:
            return (0, nodeid)
        # Next: param_id tests
        if "test_param_id" in nodeid:
            return (1, nodeid)
        # Next: optimiser comparison tests
        if "compare_optimisers" in nodeid:
            return (1, nodeid)
        # Then everything else
        return (2, nodeid)

    items.sort(key=sort_key)

    # Mark param_id and comparison tests to run serially as one group under xdist
    for item in items:
        nodeid = item.nodeid
        if "test_param_id" in nodeid or "compare_optimisers" in nodeid:
            item.add_marker(pytest.mark.xdist_group("param_id_serial"))
            item.add_marker(pytest.mark.need_autogen)

    # Mark autogeneration tests so they record completion
    for item in autogen_items:
        item.add_marker(pytest.mark.autogen_task)


@pytest.fixture(scope="session")
def autogen_status_file(project_root):
    """Path to the shared autogeneration completion tracker file."""
    return os.path.join(project_root, ".pytest_autogen_status")


@pytest.fixture(scope="function", autouse=True)
def track_autogen_completion(request, autogen_status_file):
    """
    For autogeneration tests, append a line to the status file on completion.
    Param/compare tests can wait on this file to ensure all autogen tests finished.
    """
    yield
    if "autogen_task" in request.node.keywords:
        # Append a line to indicate completion
        with open(autogen_status_file, "a") as f:
            f.write("done\n")


@pytest.fixture(scope="function", autouse=True)
def wait_for_autogen_if_needed(request, autogen_status_file):
    """
    Block param/compare tests until all autogeneration tests have completed.
    """
    if "need_autogen" not in request.node.keywords:
        return
    import time
    import os

    try:
        total = int(os.environ.get("AUTOGEN_TOTAL", "0"))
    except ValueError:
        total = 0

    if total == 0:
        # No autogen tests collected; nothing to wait for
        return

    waited = 0.0
    while True:
        if os.path.exists(autogen_status_file):
            try:
                with open(autogen_status_file, "r") as f:
                    count = sum(1 for _ in f)
                if count >= total:
                    return
            except OSError:
                pass
        time.sleep(0.1)
        waited += 0.1
        # Optional: log every ~10 seconds to show waiting
        if abs(waited - round(waited, 1)) < 1e-6 and int(waited) % 10 == 0 and waited > 0:
            print(f"Waiting for autogeneration to finish... ({waited:.1f}s)")



@pytest.fixture(scope="function")
def minimal_param_id_config(base_user_inputs, resources_dir, temp_output_dir):
    """
    Fixture that returns a minimal parameter identification configuration.
    """
    config = base_user_inputs.copy()
    config.update({
        'model_type': 'cellml_only',
        'solver': 'CVODE',
        'param_id_method': 'genetic_algorithm',
        'pre_time': 1,
        'sim_time': 1,
        'dt': 0.01,
        'DEBUG': True,
        'do_mcmc': False,
        'plot_predictions': False,
        'solver_info': {
            'MaximumStep': 0.001,
            'MaximumNumberOfSteps': 5000,
        },
        'param_id_output_dir': temp_output_dir,
    })
    return config
