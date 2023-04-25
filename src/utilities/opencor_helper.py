import opencor as oc
import numpy as np
# from func_timeout import func_timeout, FunctionTimedOut

class SimulationHelper():
    def __init__(self, cellml_path, dt,
                 sim_time, maximumNumberofSteps=100000,
                 maximum_step=0.001, pre_time=0.0):
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
        self.data.set_ode_solver_property('MaximumStep', maximum_step)
        self.data.set_point_interval(dt)  # time interval for data storage
        self.data.set_starting_point(0)
        self.data.set_ending_point(self.stop_time)
        self.tSim = np.linspace(pre_time, self.stop_time, self.n_steps + 1) # time values for stored part of simulation
        self.operation_obs_dict = {}
        self.operation_obs_names = []

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

    def get_results(self, obs_names, output_temp_results=False):
        """
        gets results after a simulation
        inputs:
        obs_names: list of strings, stores the names of state and algebraic variables you wish to access
        outputs:
        results: numpy array of size (nStates + nAlgs, n_steps). This will store
        """
        n_obs = len(obs_names)
        results = np.zeros((n_obs, self.n_steps + 1))
        # temp results stores results that are operated together
        # TODO this temporarily is a fixed size of 10, but should be changed to be dynamic
        temp_results = np.zeros((n_obs, 10, self.n_steps + 1))
        for JJ, obs_name in enumerate(obs_names):
            if obs_name in self.simulation.results().states():
                results[JJ, :] = self.simulation.results().states()[obs_name].values()[-self.n_steps - 1:]
            elif obs_name in self.simulation.results().algebraic():
                results[JJ, :] = self.simulation.results().algebraic()[obs_name].values()[-self.n_steps-1:]
            elif obs_name in self.operation_obs_names:
            # check if the obs name is in the created operation observables.
                # loop through operands
                for II in range(len(self.operation_obs_dict[obs_name]["operands"])):
                    operand_name = self.operation_obs_dict[obs_name]["operands"][II]
                    if operand_name in self.simulation.results().states():
                        temp_results[JJ, II] = self.simulation.results().states()[operand_name].values()[-self.n_steps - 1:]
                    elif operand_name in self.simulation.results().algebraic():
                        temp_results[JJ, II] = self.simulation.results().algebraic()[operand_name].values()[-self.n_steps - 1:]
                    else:
                        print(f'variable {self.operation_obs_dict[obs_name]["operands"][II]} is not a '
                              f'model variable. model variables are')
                        print([name for name in self.simulation.results().states()])
                        print([name for name in self.simulation.results().algebraic()])
                        print('exiting')
                        exit()

                if self.operation_obs_dict[obs_name]["operation"] == "multiplication":
                    if len(self.operation_obs_dict[obs_name]["operands"]) != 2:
                        print('multiplication operation must have exactly 2 operands')
                        exit()
                    results[JJ, :] = temp_results[JJ, 0] * temp_results[JJ, 1]
                elif self.operation_obs_dict[obs_name]["operation"] == "division":
                    if len(self.operation_obs_dict[obs_name]["operands"]) != 2:
                        print('division operation must have exactly 2 operands')
                        exit()
                    results[JJ, :] = temp_results[JJ, 0] / temp_results[JJ, 1] # TODO careful here with divide by zero
                elif self.operation_obs_dict[obs_name]["operation"] == "addition":
                    results[JJ, :] = np.sum(temp_results[JJ, :], axis=0)
                elif self.operation_obs_dict[obs_name]["operation"] == "subtraction":
                    if len(self.operation_obs_dict[obs_name]["operands"]) != 2:
                        print('subtraction operation must have exactly 2 operands')
                        exit()
                    results[JJ, :] = temp_results[JJ, 0] - temp_results[JJ, 1]
                else:
                    print(f'operation {self.operation_obs_dict[obs_name]["operation"]} is not a valid'
                          f'operation, must be multiplication, division, addition, or subtraction')

            else:
                print(f'variable {obs_name} is not a model variable. model variables are')
                print([name for name in self.simulation.results().states()])
                print([name for name in self.simulation.results().algebraic()])
                print('exiting')
                exit()

        if output_temp_results:
            return results, temp_results
        else:
            return results

    def get_init_param_vals(self, param_names):
        param_init = []
        for JJ, param_name_or_list in enumerate(param_names):
            if isinstance(param_name_or_list, list):
                param_init.append([])
                for param_name in param_name_or_list:
                    if param_name in self.data.states():
                        param_init[JJ].append(self.data.states()[param_name])
                    elif param_name in self.data.constants():
                        param_init[JJ].append(self.data.constants()[param_name])
                    else:
                        print(f'parameter name of {param_name} doesn\'t exist in either constants or states'
                              f'The states are:')
                        print([name for name in self.data.states()])
                        print('the constants are:')
                        print([name for name in self.data.constants()])
                        exit()
            else:
                param_name = param_name_or_list
                if param_name in self.data.states():
                    param_init.append(self.data.states()[param_name])
                elif param_name in self.data.constants():
                    param_init.append(self.data.constants()[param_name])
                else:
                    print(f'parameter name of {param_name} doesn\'t exist in either constants or states'
                          f'The states are:')
                    print([name for name in self.data.states()])
                    print('the constants are:')
                    print([name for name in self.data.constants()])
                    exit()

        return param_init

    def set_param_vals(self, param_names, param_vals):
        # ensure param_vals stores state values first, then constant values
        for JJ, param_name_or_list in enumerate(param_names):
            if isinstance(param_name_or_list, list):
                for param_name in param_name_or_list:
                    if param_name in self.data.states():
                        self.data.states()[param_name] = param_vals[JJ]
                    elif param_name in self.data.constants():
                        self.data.constants()[param_name] = param_vals[JJ]
                    else:
                        print(f'parameter name of {param_name} doesn\'t exist in either constants or states'
                              f'The states are:')
                        print([name for name in self.data.states()])
                        print('the constants are:')
                        print([name for name in self.data.constants()])
                        exit()

            else:
                param_name = param_name_or_list
                if param_name in self.data.states():
                    self.data.states()[param_name] = param_vals[JJ]
                elif param_name in self.data.constants():
                    self.data.constants()[param_name] = param_vals[JJ]
                else:
                    print(f'parameter name of {param_name} doesn\'t exist in either constants or states'
                          f'The states are:')
                    print([name for name in self.data.states()])
                    print('the constants are:')
                    print([name for name in self.data.constants()])
                    exit()

    def modify_params_and_run_and_get_results(self, param_names, mod_factors, obs_names, absolute=False):

        if absolute:
            new_param_vals= mod_factors
        else:
            init_param_vals = self.get_init_param_vals(param_names)
            new_param_vals = [a*b for a, b in zip(init_param_vals, mod_factors)]

        print(new_param_vals)
        self.set_param_vals(param_names, new_param_vals)

        success = self.run()
        print('here')
        if success:
            pred_obs_new = self.get_results(obs_names)
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

    def create_operation_variables(self, obs_names, operations, operands):
        for II in range(len(obs_names)):
            if operations[II] is not None:
                self.operation_obs_names.append(obs_names[II])
                self.operation_obs_dict[obs_names[II]] = {"operation": operations[II],
                                                          "operands": operands[II]}




