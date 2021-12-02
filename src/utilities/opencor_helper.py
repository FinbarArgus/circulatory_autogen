import opencor as oc
import numpy as np
# from func_timeout import func_timeout, FunctionTimedOut

class SimulationHelper():
    def __init__(self, cellml_path, dt,
                 sim_time, point_interval, maximumNumberofSteps=100000,
                 maximumStep=0.001, pre_time=0.0):
        self.cellml_path = cellml_path  # path to cellml file
        self.dt = dt  # time step
        self.stop_time = pre_time + sim_time  # full time of simulation
        self.pre_steps = int(pre_time/dt)  # number of steps to do before storing data (used to reach steady state)
        self.n_steps = int(sim_time/dt)  # number of steps for storing data
        self.simulation = oc.open_simulation(cellml_path)
        if not self.simulation.valid():
            print(f'simulation object opened from {cellml_path} is not valid, exiting')
            exit()
        self.data = self.simulation.data()
        self.data.set_ode_solver_property('MaximumNumberOfSteps', maximumNumberofSteps)
        self.data.set_ode_solver_property('MaximumStep', maximumStep)
        self.data.set_point_interval(point_interval)  # time interval for data storage
        self.data.set_starting_point(0)
        self.data.set_ending_point(self.stop_time)
        self.data.set_point_interval(dt)
        self.tSim = np.linspace(pre_time, self.stop_time, self.n_steps + 1) # time values for stored part of simulation

    def run(self):
        try:
            self.simulation.run()
        # except FunctionTimedOut:
        #     print("openCOR timed out")
        #     print('restarting simulation object')
        #     self.simulation.reset()
        #     self.simulation.clear_results()
        #     return False
        except RuntimeError:
            print("Failed to converge")
            print('restarting simulation object')
            self.simulation.reset()
            self.simulation.clear_results()
            return False

        return True

    def reset_and_clear(self):
        self.simulation.reset()
        self.simulation.clear_results()

    def get_results(self, obs_state_names, obs_alg_names):
        """
        gets results after a simulation
        inputs:
        obs_state_names: list of strings, stores the names of state variables you wish to access
        obs_alg_names: list of strings, stores the names of algebraic variables you wish to access
        outputs:
        results: numpy array of size (nStates + nAlgs, n_steps). This will store state variables in the
        top rows, then algebraic variables
        """
        nObs = len(obs_state_names) + len(obs_alg_names)
        results = np.zeros((nObs, self.n_steps + 1))
        for JJ, obsName in enumerate(obs_state_names):
            results[JJ, :] = self.simulation.results().states()[obsName].values()[-self.n_steps-1:]
        for JJ, obsName in enumerate(obs_alg_names):
            results[len(obs_state_names) + JJ, :] = self.simulation.results().algebraic()[obsName].values()[-self.n_steps-1:]

        return results

    def get_init_param_vals(self, init_state_names, const_names):
        param_init = []
        for JJ, param_name_or_list in enumerate(init_state_names):
            if isinstance(param_name_or_list, list):
                param_init.append([])
                for param_name in param_name_or_list:
                    if self.data.states()[param_name] is not None:
                        param_init[JJ].append(self.data.states()[param_name])
                    else:
                        print(f'parameter name of {param_name} does not exist in the simulation object states. '
                              f'The states are:')
                        print([name for name in self.data.states()])
                        exit()
            else:
                if self.data.states()[param_name]:
                    param_init.append(self.data.states()[param_name])
                else:
                    print(f'parameter name of {param_name} does not exist in the simulation object states. '
                          f'The states are:')
                    print([name for name in self.data.states()])
                    exit()
        for JJ, param_name_or_list in enumerate(const_names):
            if isinstance(param_name_or_list, list):
                param_init.append([])
                for param_name in param_name_or_list:
                    if self.data.constants()[param_name] is not None:
                        param_init[len(init_state_names) + JJ].append(self.data.constants()[param_name])
                    else:
                        print(f'parameter name of {param_name} does not exist in the simulation object constants. '
                              f'The constants are:')
                        print([name for name in self.data.constants()])
                        exit()
            else:
                if self.data.constants()[param_name_or_list] is not None:
                    param_init.append(self.data.constants()[param_name_or_list])
                else:
                    print(f'parameter name of {param_name_or_list} does not exist in the simulation object constants. '
                          f'The constants are:')
                    print([name for name in self.data.constants()])
                    exit()

        return param_init

    def set_param_vals(self, init_state_names, const_names, param_vals):
        # ensure param_vals stores state values first, then constant values
        for JJ, param_name_or_list in enumerate(init_state_names):
            if isinstance(param_name_or_list, list):
                for param_name in param_name_or_list:
                    self.data.states()[param_name] = param_vals[JJ]
            else:
                self.data.states()[param_name] = param_vals[JJ]
        for JJ, param_name_or_list in enumerate(const_names):
            if isinstance(param_name_or_list, list):
                # this entry is a list of params, set all of them to the same value
                for param_name in param_name_or_list:
                    self.data.constants()[param_name] = param_vals[len(init_state_names) + JJ]
            else:
                self.data.constants()[param_name_or_list] = param_vals[len(init_state_names) + JJ]

    def modify_params_and_run_and_get_results(self, param_state_names, param_const_names,
                                             mod_factors, obs_state_names, obs_alg_names, absolute=False):

        if absolute:
            new_param_vals= mod_factors
        else:
            init_param_vals = self.get_init_param_vals(param_state_names, param_const_names)
            new_param_vals = [a*b for a, b in zip(init_param_vals, mod_factors)]

        self.set_param_vals(param_state_names, param_const_names, new_param_vals)

        success = self.run()
        if success:
            pred_obs_new = self.get_results(obs_state_names, obs_alg_names)
            # reset params
            self.reset_and_clear()

        else:
            # simulation set cost to large,
            print('simulation failed ')
            exit()

        return pred_obs_new

    def update_times(self, dt, start_time, sim_time, pre_time):
        self.dt = dt
        self.stop_time= pre_time + sim_time # full time of simulation
        self.pre_steps = int(pre_time/self.dt)  # number of steps to do before storing data (used to reach steady state)
        self.n_steps = int(sim_time/self.dt)  # number of steps for storing data
        self.data.set_starting_point(start_time)
        self.data.set_ending_point(start_time + self.stop_time)
        self.tSim = np.linspace(pre_time, self.stop_time, self.n_steps + 1)  # time values for stored part of simulation

    def close_simulation(self):
        oc.close_simulation(self.simulation)


