import importlib.util
import numpy as np
import copy
import sys
try:
    import casadi as ca
except ImportError:
    ca = None
from .name_resolver import VariableNameResolver

class SimulationHelper:
    """
    CasADi-based solver for libCellML-generated Python modules.

    Matches the key interface of the OpenCOR SimulationHelper:
    - run()
    - update_times(dt, start_time, sim_time, pre_time)
    - get_results / get_all_results / get_all_variable_names
    - get_init_param_vals / set_param_vals
    """

    def __init__(self, model_path, dt, sim_time, solver_info=None, pre_time=0.0):
        if ca is None:
            raise RuntimeError("CasADi solver requested but CasADi is not available")
        self.model_path = model_path
        self.dt = dt
        self.pre_time = pre_time
        self.sim_time = sim_time
        self.solver_info = solver_info or {}
        solver_method = self.solver_info.get('method')
        self.solve_ivp_method = solver_method
        self._load_model()
        self.update_times(dt, 0.0, sim_time, pre_time)
        self._init_state()
        self._has_run = False
        self._do_ad = False  # set True by _create_param_subset; cleared by reset_and_clear

    def set_protocol_info(self, protocol_info):
        """Store protocol metadata for a common helper API."""
        self.protocol_info = protocol_info

    # ---- setup helpers ----
    def _load_model(self):
        spec = importlib.util.spec_from_file_location("generated_model", self.model_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module

        self.STATE_COUNT = module.STATE_COUNT
        self.VARIABLE_INFO = module.VARIABLE_INFO
        self.STATE_INFO = module.STATE_INFO

        self._resolver = VariableNameResolver(self.STATE_INFO, self.VARIABLE_INFO)

        # Convenience maps (derived from resolver, kept for compatibility)
        self.state_name_to_idx = {name: idx for name, (kind, idx) in self._resolver._map.items() if kind == "state"}
        self.var_name_to_idx   = {name: idx for name, (kind, idx) in self._resolver._map.items() if kind == "var"}
        self.state_idx_to_name = {idx: name for name, idx in self.state_name_to_idx.items()}
        self.var_idx_to_name   = {idx: name for name, idx in self.var_name_to_idx.items()}

        # identify constants and algebraics
        self.constant_indices = [i for i, info in enumerate(self.VARIABLE_INFO) if info["type"].name in ["CONSTANT", "COMPUTED_CONSTANT"]]
        self.algebraic_indices = [i for i, info in enumerate(self.VARIABLE_INFO) if info["type"].name == "ALGEBRAIC"]

    def _init_state(self):
        # Save numeric initial values before symbolic patching
        _s0 = self.model.create_states_array()
        _r0 = self.model.create_states_array()
        _v0 = self.model.create_variables_array()
        self.model.initialise_variables(_s0, _r0, _v0)
        self.model.compute_computed_constants(_v0)
        self._numeric_x0 = np.array(_s0, dtype=float)
        self._numeric_variables_all = np.array(_v0, dtype=float)

        self.states = self.model.create_states_array()
        self.rates = self.model.create_states_array()
        self.variables = self.model.create_variables_array()

        self._patch_math_functions()

        self.states_symb = self._compute_states_symb()
        self.variables_all_symb = self._compute_all_variables_symb()
        self.variables_all_symb, self.rates_symb = self._compute_rates_symb()

        self.model.initialise_variables(self.states, self.rates, self.variables)
        self.model.compute_computed_constants(self.variables)
        self._store_constants_defaults()
        self._store_variable_defaults()

    def _store_constants_defaults(self):
        self.default_constants = [self.variables[i] for i in self.constant_indices]
        self.default_state_inits = [s for s in self.states]

    def _store_variable_defaults(self):
        self.variables_symb = ca.vertcat(
            *[self.variables_all_symb[i] for i in self.constant_indices]
        )
        # Use pre-saved numeric values — symbolic self.variables cannot be cast to float
        self.variables = np.array(
            [self._numeric_variables_all[i] for i in self.constant_indices],
            dtype=float,
        )

    def _compute_states_symb(self):
        states = self.states.copy()
        for i, info in enumerate(self.model.STATE_INFO):
            states[i] = ca.SX.sym(info["name"])
        self.states = states
        return ca.vertcat(*states)

    def _compute_all_variables_symb(self):
        variables = self.variables.copy()
        for i, info in enumerate(self.model.VARIABLE_INFO):
            variables[i] = ca.SX.sym(info["name"])
        self.variables = variables
        return ca.vertcat(*variables)

    def _compute_rates_symb(self):
        self.model.compute_rates(self.start_time, self.states, self.rates, self.variables)
        return ca.vertcat(*self.variables), ca.vertcat(*self.rates)
    
    # Patch math functions to use CasADi versions for symbolic compatibility. Add more functions as needed.
    def _patch_math_functions(self):
        cellml_math_map = {
            "log": ca.log,
            "exp": ca.exp,
            "sin": ca.sin,
            "cos": ca.cos,
            "tan": ca.tan,
            "sqrt": ca.sqrt,
            "floor": ca.floor,
            "pow": ca.power,
        }
        for name, func in cellml_math_map.items():
            setattr(self.model, name, func)

    # ---- name resolution ----
    def _resolve_name(self, name: str):
        """Delegate to VariableNameResolver. Returns (kind, index) or (None, None)."""
        return self._resolver.resolve(name)

    def _var_idx_to_const_pos(self, var_idx: int) -> int:
        """Convert a VARIABLE_INFO index to its position in self.variables (constants array)."""
        return self.constant_indices.index(var_idx)

    # ---- timing helpers ----
    def update_times(self, dt, start_time, sim_time, pre_time):
        self.dt = dt
        self.pre_time = pre_time
        self.sim_time = sim_time
        self.start_time = start_time
        self.stop_time = start_time + pre_time + sim_time
        self.pre_steps = int(pre_time/dt)
        self.n_steps = int(sim_time/dt)
        self.t_eval = np.arange(start_time, self.stop_time + dt/2, dt)
        # stored portion excludes pre_time
        self.tSim = self.t_eval[self.pre_steps:]

    # ---- parameter helpers ----
    def get_init_param_vals(self, param_names):
        vals = []
        for name_or_list in param_names:
            if not isinstance(name_or_list, list):
                name_or_list = [name_or_list]
            sub = []
            for name in name_or_list:
                kind, idx = self._resolver.resolve(name)
                if kind == "state":
                    sub.append(self.states[idx])
                elif kind == "var":
                    sub.append(self.variables[self._var_idx_to_const_pos(idx)])
                else:
                    raise ValueError(f"parameter name {name} not found in states or variables")
            vals.append(sub if len(sub) > 1 else sub[0])
        return vals

    def set_param_vals(self, param_names, param_vals):
        for idx, name_or_list in enumerate(param_names):
            vals = param_vals[idx]

            def _to_list(x):
                if isinstance(x, (list, tuple)):
                    return list(x)
                try:
                    import numpy as _np
                    if isinstance(x, _np.ndarray):
                        return x.tolist()
                except Exception:
                    pass
                return [x]

            name_or_list = _to_list(name_or_list)
            vals = _to_list(vals)

            for name, val in zip(name_or_list, vals):
                kind, idx_res = self._resolver.resolve(name)
                if kind == "state":
                    self.states[idx_res] = val
                elif kind == "var":
                    self.variables[self._var_idx_to_const_pos(idx_res)] = val
                    # Sync state initial condition if this is an _init parameter
                    var_name = self.var_idx_to_name.get(idx_res, "")
                    var_part = var_name.split("/")[-1] if "/" in var_name else var_name
                    if var_part.endswith("_init"):
                        state_var = var_part[:-5]
                        state_kind, state_idx = self._resolver.resolve(state_var)
                        if state_kind == "state":
                            self.states[state_idx] = val
                            self.default_state_inits[state_idx] = val
                else:
                    raise ValueError(f"parameter name {name} not found in states or variables")
        self.model.compute_computed_constants(self.variables)

    def _post_process(self):
        # --- Symbolic pass (preserved for AD) ---
        # Compute algebraic variable trajectories as SX expressions (function of states_symb, variables_symb)
        var_names = list(self.var_name_to_idx.keys())
        var_cols = []

        state_cols = ca.horzsplit(self.state_traj_symb, 1)

        for ti, state_vec in zip(self.tSim, state_cols):
            states = state_vec
            rates = [0.0]*self.STATE_COUNT
            vars_symb_copy = copy.copy(self.variables_all_symb)
            self.model.compute_rates(ti, states, rates, vars_symb_copy)
            self.model.compute_variables(ti, states, rates, vars_symb_copy)

            var_vec = ca.vertcat(*[
                vars_symb_copy[self.var_name_to_idx[name]]
                for name in var_names
            ])

            var_cols.append(var_vec)

        self.var_traj_symb = ca.horzcat(*var_cols)  # SX: shape (n_vars, n_times)

        # --- Numeric pass (for get_results / get_all_results) ---
        # Evaluate the symbolic trajectories at current numeric param values
        x0 = np.array(self.states, dtype=float)
        p = self.variables
        var_func = ca.Function('var_traj', [self.states_symb, self.variables_symb], [self.var_traj_symb])
        self.var_traj_dm = np.array(var_func(ca.DM(x0), ca.DM(p)))  # (n_vars, n_times)

    # ---- simulation ----
    def run(self):
        ode = {
            "x": self.states_symb,
            "p": self.variables_symb,
            "ode": self.rates_symb,
        }

        # Allow caller to pass extra integrator options (e.g. tolerances) via solver_info["options"]
        integrator_opts = {"reltol": 1e-8, "abstol": 1e-10}
        integrator_opts.update(self.solver_info.get("options", {}))
        self.F = ca.integrator("F", self.solve_ivp_method, ode, 0, self.dt, integrator_opts)
        self.F_map = self.F.mapaccum(self.n_steps)

        # Symbolic trajectory — SX function of (states_symb, variables_symb); required for AD
        res = self.F_map(x0=self.states_symb, p=self.variables_symb)
        self.state_traj_symb = ca.horzcat(self.states_symb, res["xf"])

        # Numeric trajectory — evaluate symbolic graph at current param values for get_results()
        x0 = np.array(self.states, dtype=float)
        p = self.variables
        traj_func = ca.Function('state_traj', [self.states_symb, self.variables_symb], [self.state_traj_symb])
        self.state_traj_dm = np.array(traj_func(ca.DM(x0), ca.DM(p)))  # (n_states, n_steps+1)

        self._has_run = True
        self._post_process()
        
        return True

    # ---- results ----
    def get_all_variable_names(self):
        return list(self.state_name_to_idx.keys()) + list(self.var_name_to_idx.keys())

    def _extract(self, name):
        """Return numeric trajectory slice for get_results / get_all_results.

        When in AD mode (_do_ad=True), returns symbolic SX so the cost computed
        from these values remains differentiable w.r.t. variables_symb.
        """
        if self._do_ad:
            return self._extract_symb(name)
        if name == 'time':
            return self.tSim
        if name in self.state_name_to_idx:
            idx = self.state_name_to_idx[name]
            return self.state_traj_dm[idx, self.pre_steps:]
        if name in self.var_name_to_idx:
            idx = self.var_name_to_idx[name]
            return self.var_traj_dm[idx, self.pre_steps:]
        kind, idx_res = self._resolve_name(name)
        if kind == "state":
            return self.state_traj_dm[idx_res, self.pre_steps:]
        if kind == "var":
            return self.var_traj_dm[idx_res, self.pre_steps:]
        raise ValueError(f"variable {name} not found")

    def _extract_symb(self, name):
        """Return symbolic (SX) trajectory slice for AD cost building."""
        if name in self.state_name_to_idx:
            idx = self.state_name_to_idx[name]
            return self.state_traj_symb[idx, self.pre_steps:]
        if name in self.var_name_to_idx:
            idx = self.var_name_to_idx[name]
            return self.var_traj_symb[idx, self.pre_steps:]
        kind, idx_res = self._resolve_name(name)
        if kind == "state":
            return self.state_traj_symb[idx_res, self.pre_steps:]
        if kind == "var":
            return self.var_traj_symb[idx_res, self.pre_steps:]
        raise ValueError(f"variable {name} not found (symbolic)")

    def get_results(self, variables_list_of_lists, flatten=False):
        if type(variables_list_of_lists[0]) is not list:
            variables_list_of_lists = [[entry] for entry in variables_list_of_lists]
        results = []
        for variables_list in variables_list_of_lists:
            row = [self._extract(name) for name in variables_list]
            results.append(row)
        if flatten:
            results = [item for sublist in results for item in sublist]
        return results

    def get_all_results(self, flatten=False):
        return self.get_results(self.get_all_variable_names(), flatten=flatten)
    
    def get_all_results_dict(self):
        if self._has_run:
            self._last_results_dict = self._collect_all_results_dict()
            return {name: val for name, val in self._last_results_dict.items()}
        if self._last_results_dict is not None:
            return {name: val for name, val in self._last_results_dict.items()}
        raise RuntimeError("Simulation has not been run yet.")

    def _collect_all_results_dict(self):
        variable_names = self.get_all_variable_names()
        # Use symbolic extracts so var_func is a differentiable CasADi function (for AD)
        values = [self._extract_symb(name) for name in variable_names]
        values = [
            ca.reshape(v, -1, 1) if isinstance(v, ca.SX) else v
            for v in values
        ]
        self.var_func = ca.Function('var_func', [self.states_symb, self.variables_symb], values)
        values = self.var_func(self.states, self.variables)
        return {name: val for name, val in zip(variable_names, values)}

    def _create_param_subset(self, param_names, param_vals=None):
        param_names = [x[0] for x in param_names]

        # Resolve each param to its VARIABLE_INFO index, then find its SX symbol
        var_indices = []   # VARIABLE_INFO indices
        const_positions = []  # positions in self.variables (constants-only array)
        symb_list = []

        for name in param_names:
            kind, var_idx = self._resolver.resolve(name)
            if kind != "var":
                raise ValueError(f"Parameter {name!r} not found as a variable (resolved kind={kind!r})")
            const_pos = self._var_idx_to_const_pos(var_idx)
            var_indices.append(var_idx)
            const_positions.append(const_pos)
            symb_list.append(self.variables_all_symb[var_idx])

        self.variables_symb_subset = ca.vertcat(*symb_list)

        if param_vals is not None:
            param_vals = np.asarray(param_vals, dtype=float)
            for const_pos, val in zip(const_positions, param_vals):
                self.variables[const_pos] = val
            self.variables_subset = param_vals
        else:
            self.variables_subset = np.array(
                [self.variables[pos] for pos in const_positions], dtype=float
            )

        self._do_ad = True  # switch get_results to symbolic mode for AD

    # ---- reset helpers ----
    def run_offline_pre_and_set_default_state(self, offline_pre_time):
        """Run unlogged warmup once; use end state as default for reset_states()."""
        offline_pre_time = float(offline_pre_time)
        if offline_pre_time <= 0:
            return
        self._do_ad = False
        self.update_times(self.dt, 0.0, offline_pre_time, 0.0)
        success = self.run()
        if not success:
            raise RuntimeError("Offline pre-time simulation failed")
        self.states = list(self.state_traj_dm[:, -1])
        self.default_state_inits = copy.copy(self.states)
        self._has_run = False
        self.states = copy.copy(self.default_state_inits)
        self.model.compute_computed_constants(self.variables)

    def reset_and_clear(self, only_one_exp=-1):
        self._do_ad = False
        self._init_state()

    def reset_states(self):
        self.states = copy.copy(self.default_state_inits)
        self.model.compute_computed_constants(self.variables)

    def close_simulation(self):
        # no-op for scipy solver
        pass

