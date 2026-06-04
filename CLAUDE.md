# circulatory_autogen — Project Summary and Change Log

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
    "pre_times":  [0.0, 0.0],                     // pre-run equilibration per experiment
    "sim_times":  [[5], [5]],                      // [[sub-exp durations], …] per experiment
    "params_to_change": {
      "component/param": [[exp0_sub0, …], [exp1_sub0, …]]
    },
    "protocol_traces": {
      "trace_key": {"t": [...], "values": [...]}   // time-series driving a parameter
    }
  },
  "data_items": [...],
  "prediction_items": [...]
}
```
A `params_to_change` value can be a **float** (constant) or a **string** (trace key from `protocol_traces`).

**Timeline conventions**

- **`pre_times[j]`**, **`sim_times[j]`**: `pre_times[j]` is the unlogged Myokit **pre-pass** duration before the first subexperiment of experiment `j`. Each `sim_times[j][k]` is the duration of subexperiment `(j, k)`. The helper advances a cumulative **segment index** `start_time` across subs (used to build output sample grids along the logical protocol).
- **`protocol_traces[...].t`**: Times are **seconds from the start of the Myokit segment where that trace is applied** (i.e. when `set_param_vals` installs a trace for sub `(j,k)`, `t` usually starts at `0` at the beginning of that sub’s integration after any `pre_time` for that sub has already run). Match or exceed the segment length in `sim_times[j][k]`.
- **Myokit (`myokit_helper`)**: Each `update_times` calls `simulation.reset()`, so model time returns to **0** before every `run()`. Logging instants passed to Myokit are **shifted** so the first requested output time matches `simulation.time()` after `pre(pre_time)` (this keeps multi-subexperiment protocols consistent with OpenCOR-style cumulative `start_time`).

---

## Important Changes Made

### 1. Multi-trace protocol support in `src/solver_wrappers/myokit_helper.py`

**Problem (see `required_changes.md`):**  
`set_protocol_info` statically bound Myokit's `pace` label to the *first* parameter that had any string trace value.  `set_param_vals` then rejected any trace assignment to a *different* parameter, making it impossible to drive different parameters in different experiments.

**Fix:**
Three new/modified methods in `SimulationHelper`:

| Method | Role |
|--------|------|
| `_find_required_paced_qname(param_names, param_vals)` | Scans the current `set_param_vals` call and returns the Myokit qname of the parameter that needs pacing (or `None`). Raises if two *different* variables both request tracing in the same call. |
| `_rebind_pace_to(qname)` | Saves simulation state + time, unbinds the old `pace` label, binds the new variable, calls `_recreate_simulation()` (Myokit requires a fresh `Simulation` after binding changes), then restores state + time. |
| `set_param_vals(...)` *(modified)* | **Phase 1:** calls `_find_required_paced_qname`; if the result differs from `self.paced_parameter_qname`, calls `_rebind_pace_to`.  **Phase 2:** existing loop setting constants / protocols (now safe because rebind already occurred). |

**Result:** Each call to `set_param_vals` can pace a *different* variable from the previous call without losing simulation state or raising errors.

---

### 2. New forced Lotka-Volterra CellML model — `tests/test_inputs/Lotka_Volterra_forced.cellml`

Flat CellML 2.0 Lotka-Volterra model with **explicit time-varying forcing inputs**:

```
dx/dt = (alpha + u_alpha) * x  -  beta * x * y
dy/dt =  delta * x * y  -  (gamma + u_gamma) * y
```

`u_alpha` and `u_gamma` are constants (initial value 0) placed directly in `Lotka_Volterra_module` so Myokit names them `Lotka_Volterra_module.u_alpha` / `Lotka_Volterra_module.u_gamma` — bindable to `pace` without name-resolution ambiguity.

Default parameters: α=5, β=0.2, δ=0.2, γ=3; initial states x=20, y=10.

---

### 3. Multi-trace obs_data.json — `resources/Lotka_Volterra_forced_multi_trace_obs_data.json`

Four-experiment protocol (see file for full `params_to_change`). Highlights:

| Experiment | Notes |
|------------|--------|
| 0 | **pre_time = 1 s**, **two subs** `[0.25, 5.0]` s — first sub constants, second sub `u_alpha_trace`; regression test for cumulative `start_time` + `reset()` log alignment |
| 1 | `u_gamma_trace`; u_alpha constant |
| 2–3 | Pure constant forcing variants |

Canonical test case for multi-trace support, pace rebinding, and multi-subexperiment Myokit timing.

---

### 4. New test — `tests/test_solvers.py::test_myokit_multi_trace_protocol`

`@pytest.mark.integration @pytest.mark.solver` test that:
1. Loads `tests/test_inputs/Lotka_Volterra_forced.cellml` directly (no code generation needed).
2. Calls `set_protocol_info` with the multi-trace obs_data.
3. Runs both experiments via the standard `update_times → set_param_vals → run` loop.
4. Asserts results are finite, non-negative, and **different** between experiments (verifying that distinct traces produced distinct dynamics).

The test was the failing case that required the `myokit_helper.py` fix.

---

### 5. Fix: `set_constant` values lost across `_rebind_pace_to` — issue #219

**Problem (issue #219):** After genetic-algorithm calibration, recomputing cost with `best_param_vals` on a **regenerated CellML** (`do_generation_with_fit_parameters=True`) diverges from `best_cost.npy` when the protocol uses **two** `protocol_traces` clamps.

Root cause:

- `set_protocol_info` pre-binds the *first* trace parameter to `pace`.
- For the *second* trace experiment, `set_param_vals` detects a different paced variable and calls `_rebind_pace_to`, which calls `_recreate_simulation()` → `myokit.Simulation(self.model)`.
- Myokit docs: *"The model passed to the simulation is cloned and stored internally, so changes to the original model object will not affect the simulation."* — the new simulation is cloned from `self.model` (the template), not from the old simulation, so **all `set_constant` calls made on the old simulation are silently discarded**.
- This means ID parameters applied at step A (`set_param_vals(id_param_names, best_param_vals)`) before the rebind are lost for the second trace experiment.
- In **calibration**, `self.model` still has template defaults → simulation uses template defaults after rebind → `best_cost` computed with wrong constants.
- In **post-regeneration**, `self.model` is loaded from the regenerated CellML (which has best-fit constants embedded) → simulation uses best-fit values after rebind → `cost_check` uses correct constants.
- The asymmetry causes the large mismatch; single-trace protocols never trigger a rebind (the one trace is pre-set by `set_protocol_info`), so they are unaffected.

**Fix (one line in `set_param_vals`):** When `simulation.set_constant(qname, val)` is called, also sync `self.model` via:

```python
self.qname_to_var[qname].set_rhs(myokit.Number(float(val)))
```

Because `self.qname_to_var` holds references to Variable objects *inside* `self.model`, this keeps the template model's constant values current. The next `_recreate_simulation()` clone then starts with the correct constants, and calibration / post-regeneration produce identical costs.

**Acceptance:** `abs(best_cost - cost_check) < 1e-3` for the full two-trace sympathetic-neuron protocol after `do_generation_with_fit_parameters=True` regeneration.
