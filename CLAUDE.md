# circulatory_autogen — Project Summary

## Repository Overview

`circulatory_autogen` is a Python framework for **automated generation and parameter identification of computational physiology models** (primarily CellML-based circulatory and electrophysiology models).

### Key capabilities
| Area | Description |
|------|-------------|
| **Model generation** | Generates flat CellML models from CSV parameter files and modular CellML templates |
| **Solver backends** | Wraps Myokit (CVODE), OpenCOR (CVODE), SciPy solve_ivp, and CasADi integrators behind a common `SimulationHelper` API |
| **Protocol runner** | Runs multi-experiment, multi-sub-experiment protocols defined in `obs_data.json` files |
| **Parameter identification** | Bayesian and gradient-based param ID (`src/param_id/paramID.py`) |
| **Sensitivity analysis** | Local and global sensitivity (`src/sensitivity/`) |
| **CasADi AD support** | Forward-mode automatic differentiation for gradient-based ID |

**SN_simple / SN_full:** `cellml_only` generation and **Myokit** (`myokit_helper`) accept the emitted CellML (including state initial values that reference `*_init` parameters). **`PythonGenerator`** uses **libCellML Analyser** and requires a strict **ODE** model; the same SN CellML currently fails `ANALYSER_VARIABLE_NON_CONSTANT_INITIALISATION`, so `model_type: python` codegen is not expected to work for SN until the generator or model satisfies the analyser.

### Directory layout
```
src/
  solver_wrappers/     — SimulationHelper per backend (myokit, opencor, python, casadi)
  param_id/            — Parameter identification (paramID.py)
  generators/          — CellML & Python code generators
  utilities/           — Protocol runner, plotting, signal processing
resources/             — CellML models, obs_data.json files, parameter CSVs
generated_models/      — Runtime-generated flat CellML + Python models
tests/                 — pytest suite (test_solvers.py, test_param_id.py, …)
```

### `obs_data.json` format
The core experiment descriptor used by the protocol runner:
```json
{
  "protocol_info": {
    "pre_times":  [0.0, 0.0],
    "sim_times":  [[5], [5]],
    "params_to_change": {
      "component/param": [[exp0_sub0, …], [exp1_sub0, …]]
    },
    "protocol_traces": {
      "trace_key": {"t": [...], "values": [...]}
    }
  },
  "data_items": [...],
  "prediction_items": [...]
}
```
A `params_to_change` value can be a **float** (constant) or a **string** (trace key from `protocol_traces`).

**Timeline conventions**

- **`pre_times[j]`**, **`sim_times[j]`**: `pre_times[j]` is the unlogged Myokit **pre-pass** duration before the first subexperiment of experiment `j`. Each `sim_times[j][k]` is the duration of subexperiment `(j, k)`.
- **`protocol_traces[...].t`**: Times are **seconds from the start of the Myokit segment** where that trace is applied. Match or exceed the segment length in `sim_times[j][k]`.
- **Myokit (`myokit_helper`)**: Each `update_times` calls `simulation.reset()`. Logging instants are **shifted** so the first requested output time matches `simulation.time()` after `pre(pre_time)`.

---

## Testing

Add or extend tests in `tests/` for **every new feature and every bugfix**.

- Place inputs under `tests/test_inputs/` when needed; reuse existing CellML models and `obs_data.json` patterns in `resources/` where possible.
- Mark integration or solver-dependent tests with `@pytest.mark.integration` and `@pytest.mark.solver` as appropriate (see `tests/test_solvers.py`).
- Bugfixes should include a test that would have failed before the fix; new features should cover the main success path and any important edge cases.
- Run `pytest` (or the relevant subset) before opening a PR.
