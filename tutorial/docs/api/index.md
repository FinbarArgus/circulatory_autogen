# API Reference

This section documents the Python API for driving Circulatory Autogen
programmatically — the same calls used by the
[interactive tutorial](https://github.com/FinbarArgus/circulatory_autogen/blob/master/tutorial/interactive/generation_and_calibration.py)
and by external tools such as the CUFLynx GUI.

The reference pages are generated automatically from the docstrings in `src/`,
so they stay in sync with the code.

## Setup

The package imports are relative to `src/`, so add it to `sys.path` before
importing (this is what the tutorials do). Everything must run under the
OpenCOR Python shell.

```python
import sys
sys.path.append("/path/to/circulatory_autogen/src")

from utilities.utility_funcs import get_default_inp_data_dict
```

## Public API at a glance

The entry points below cover the full generate → simulate → calibrate →
analyse pipeline. Each links to its detailed reference page.

| Task | Entry point | Page |
|---|---|---|
| Build the config dict | [`get_default_inp_data_dict`][utilities.utility_funcs.get_default_inp_data_dict] | [Utilities](utilities.md) |
| Generate a model | [`generate_with_new_architecture`][scripts.script_generate_with_new_architecture.generate_with_new_architecture] | [Model generation](generation.md) |
| Run a simulation | [`get_simulation_helper`][solver_wrappers.get_simulation_helper] / [`get_simulation_helper_from_inp_data_dict`][solver_wrappers.get_simulation_helper_from_inp_data_dict] | [Simulation](solver-wrappers.md) |
| Build observation data | [`ObsDataCreator`][utilities.obs_data_helpers.ObsDataCreator] | [Observation data &amp; utilities](utilities.md) |
| Calibrate parameters | [`CVS0DParamID`][param_id.paramID.CVS0DParamID] | [Parameter identification](param-id.md) |
| Sensitivity analysis | [`SensitivityAnalysis`][sensitivity_analysis.sensitivityAnalysis.SensitivityAnalysis] | [Sensitivity analysis](sensitivity.md) |
| Identifiability analysis | [`IdentifiabilityAnalysis`][identifiabilty_analysis.identifiabilityAnalysis.IdentifiabilityAnalysis] | [Identifiability analysis](identifiability.md) |
| Run protocols standalone | [`ProtocolRunner`][protocol_runners.protocol_runner.ProtocolRunner] | [Protocols](protocols.md) |

## Typical call sequence

```python
from utilities.utility_funcs import get_default_inp_data_dict
from scripts.script_generate_with_new_architecture import generate_with_new_architecture
from solver_wrappers import get_simulation_helper_from_inp_data_dict
from utilities.obs_data_helpers import ObsDataCreator
from param_id.paramID import CVS0DParamID

# 1. Configuration (== user_inputs.yaml defaults, then mutate in code)
inp = get_default_inp_data_dict(file_prefix, input_param_file, resources_dir)
inp["sim_time"], inp["pre_time"] = 2, 20.0

# 2. Generate the model
generate_with_new_architecture(inp_data_dict=inp)

# 3. Simulate
sim = get_simulation_helper_from_inp_data_dict(inp)
sim.run()
t, y = sim.get_time(), sim.get_results(variables_to_plot, flatten=True)

# 4. Build observation data
obs = ObsDataCreator()
obs.add_protocol_info(pre_times, sim_times, params_to_change)
obs.add_data_item(entry)
obs_data_dict = obs.get_obs_data_dict()

# 5. Calibrate
pid = CVS0DParamID.init_from_dict(inp)
pid.set_ground_truth_data(obs_data_dict)
pid.set_params_for_id(params_for_id_dict)
pid.run()
pid.simulate_with_best_param_vals()
pid.plot_outputs()
```

`params_for_id_dict` is a list of
`{vessel_name, param_name, min, max, name_for_plotting}` entries (the in-memory
equivalent of `{prefix}_params_for_id.csv`); `vessel_name` may be a single name
or a list to share one calibrated parameter across many vessels.
