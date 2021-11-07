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
        self.preSteps = int(pre_time/dt)  # number of steps to do before storing data (used to reach steady state)
        self.nSteps = int(sim_time/dt)  # number of steps for storing data
        self.simulation = oc.open_simulation(cellml_path)
        self.data = self.simulation.data()
        self.data.set_ode_solver_property('MaximumNumberOfSteps', maximumNumberofSteps)
        self.data.set_ode_solver_property('MaximumStep', maximumStep)
        self.data.set_point_interval(point_interval)  # time interval for data storage
        self.data.set_starting_point(0)
        self.data.set_ending_point(self.stop_time)
        self.data.set_point_interval(dt)
        self.tSim = np.linspace(pre_time, self.stop_time, self.nSteps) # time values for stored part of simulation

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

    def reset(self):
        self.simulation.reset()
        self.simulation.clear_results()

    def get_results(self, obs_state_names, obs_alg_names):
        """
        gets results after a simulation
        inputs:
        obs_state_names: list of strings, stores the names of state variables you wish to access
        obs_alg_names: list of strings, stores the names of algebraic variables you wish to access
        outputs:
        results: numpy array of size (nStates + nAlgs, nSteps). This will store state variables in the
        top rows, then algebraic variables
        """
        nObs = len(obs_state_names) + len(obs_alg_names)
        results = np.zeros((nObs, self.nSteps))
        for JJ, obsName in enumerate(obs_state_names):
            results[JJ, :] = self.simulation.results().states()[obsName].values()[-self.nSteps:]
        for JJ, obsName in enumerate(obs_alg_names):
            results[len(obs_state_names) + JJ, :] = self.simulation.results().algebraic()[obsName].values()[-self.nSteps:]

        return results

    def get_init_param_vals(self, init_state_names, const_names):
        param_init = []
        for JJ, param_name_or_list in enumerate(init_state_names):
            if isinstance(param_name_or_list, list):
                param_init.append([])
                for param_name in param_name_or_list:
                    param_init[JJ].append(self.data.states()[param_name])
            else:
                param_init.append(self.data.states()[param_name])
        for JJ, param_name_or_list in enumerate(const_names):
            if isinstance(param_name_or_list, list):
                param_init.append([])
                for param_name in param_name_or_list:
                    param_init[len(init_state_names) + JJ].append(self.data.constants()[param_name])
            else:
                param_init.append(self.data.constants()[param_name_or_list])

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

    def update_times(self, dt, start_time, sim_time, pre_time):
        self.dt = dt
        self.stop_time= pre_time + sim_time # full time of simulation
        self.preSteps = int(pre_time/self.dt)  # number of steps to do before storing data (used to reach steady state)
        self.nSteps = int(sim_time/self.dt)  # number of steps for storing data
        self.data.set_starting_point(start_time)
        self.data.set_ending_point(start_time + self.stop_time)
        self.tSim = np.linspace(pre_time, self.stop_time, self.nSteps)  # time values for stored part of simulation

    def close_simulation(self):
        oc.close_simulation(self.simulation)


