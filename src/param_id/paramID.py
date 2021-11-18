import numpy as np
import os
import sys
from sys import exit
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utilities'))
resources_dir = os.path.join(os.path.dirname(__file__), '../../resources')
import math as math
import opencor as oc
import time
import matplotlib
import matplotlib.pyplot as plt
import paperPlotSetup
matplotlib.use('Agg')
from utilities import Normalise_class
paperPlotSetup.Setup_Plot(3)
from opencor_helper import SimulationHelper
from mpi4py import MPI
import re
from numpy import genfromtxt
import csv
from datetime import date
from skopt import gp_minimize, Optimizer

class CVS0DParamID():
    """
    Class for doing parameter identification on a 0D cvs model
    """
    def __init__(self, model_path, param_id_model_type, param_id_method, file_name_prefix, sim_time=2.0, pre_time=20.0):
        self.model_path = model_path
        self.param_id_method = param_id_method
        self.param_id_model_type = param_id_model_type
        self.file_name_prefix = file_name_prefix

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.num_procs = self.comm.Get_size()
        if self.num_procs == 1:
            print('WARNING Running in serial, are you sure you want to be a snail?')

        case_type = f'{param_id_method}_{file_name_prefix}'
        if self.rank == 0:
            self.param_id_output_dir = os.path.join(os.path.dirname(__file__), '../../param_id_output')
            if not os.path.exists(self.param_id_output_dir):
                os.mkdir(self.param_id_output_dir)
            self.output_dir = os.path.join(self.param_id_output_dir, f'{case_type}')
            if not os.path.exists(self.output_dir):
                os.mkdir(self.output_dir)
            self.plot_dir = os.path.join(self.output_dir, 'plots_param_id')
            if not os.path.exists(self.plot_dir):
                os.mkdir(self.plot_dir)

        self.comm.Barrier()

        # param names
        self.obs_state_names = None
        self.obs_alg_names = None
        self.obs_nom_names = None
        self.weight_vec = None
        self.param_state_names = None
        self.param_const_names = None
        self.num_obs_states = None
        self.num_obs_algs = None
        self.num_obs = None
        self.num_resistance_params = None
        self.__set_and_save_param_names()

        # ground truth values
        self.ground_truth = self.__get_ground_truth_values()

        # define allowed param ranges # FIXME take these values as inputs from script
        # self.param_mins = np.array([100e-6, 1e-9] + [4e6]*self.num_resistance_params)
        self.param_mins = np.array([700e-6] + [5e5]*self.num_resistance_params)
        # self.param_maxs = np.array([1200e-6, 1e-6] + [4e10]*self.num_resistance_params)
        self.param_maxs = np.array([2600e-6] + [5e10]*self.num_resistance_params)

        if param_id_model_type == 'CVS0D':
            self.param_id = OpencorParamID(self.model_path, self.param_id_method,
                                           self.obs_state_names, self.obs_alg_names, self.weight_vec,
                                           self.param_state_names, self.param_const_names, self.ground_truth,
                                           self.param_mins, self.param_maxs, sim_time=sim_time, pre_time=pre_time)
        if self.rank == 0:
            self.param_id.set_output_dir(self.output_dir)

        self.best_output_calculated = False

    def run(self):
        self.param_id.run()

    def simulate_with_best_param_vals(self):
        self.param_id.simulate_once()
        self.best_output_calculated = True

    def update_param_range(self, params_to_update_list_of_lists, mins, maxs):
        # TODO make the user input a parameters_range.csv file to define the mins and maxs
        for params_to_update_list, min, max in zip(params_to_update_list_of_lists, mins, maxs):
            for JJ, param_name_list in enumerate(self.param_state_names):
                if param_name_list == params_to_update_list:
                    self.param_mins[JJ] = min
                    self.param_maxs[JJ] = max
            for JJ, param_name_list in enumerate(self.param_const_names):
                if param_name_list == params_to_update_list:
                    self.param_mins[len(self.param_state_names) + JJ] = min
                    self.param_maxs[len(self.param_state_names) + JJ] = max

    def plot_outputs(self):
        if not self.best_output_calculated:
            print('simulate_with_best_param_vals must be done first '
                  'before plotting output values')
            print('running simulate_with_best_param_vals ')
            self.simulate_with_best_param_vals()

        print('plotting best observables')
        m3_to_cm3 = 1e6
        Pa_to_kPa = 1e-3

        best_fit_obs = self.param_id.sim_helper.get_results(self.obs_state_names, self.obs_alg_names)
        best_fit_obs_means = np.mean(best_fit_obs, axis=1)

        # _________ Plot best comparison _____________
        subplot_width = 2
        fig, axs = plt.subplots(subplot_width, subplot_width)

        obs_names = self.obs_state_names + self.obs_alg_names
        col_idx = 0
        row_idx = 0
        plot_idx = 0
        tSim = self.param_id.sim_helper.tSim - self.param_id.pre_time
        means_plot_gt = np.tile(self.ground_truth.reshape(-1, 1), (1, self.param_id.sim_helper.nSteps))
        means_plot_bf = np.tile(best_fit_obs_means.reshape(-1, 1), (1, self.param_id.sim_helper.nSteps))
        for II in range(self.num_obs):

            words = obs_names[II].replace('_', ' ').upper().split()
            obs_name_for_plot = "".join([word[0] for word in words])
            if II < self.num_obs_states:
                axs[row_idx, col_idx].plot(tSim, m3_to_cm3*means_plot_gt[II, :], 'k', label='gt mean')
                axs[row_idx, col_idx].plot(tSim, m3_to_cm3*means_plot_bf[II, :], 'b', label='bf mean')
                axs[row_idx, col_idx].plot(tSim, m3_to_cm3*best_fit_obs[II, :], 'r', label='bf')
                axs[row_idx, col_idx].set_ylabel(f'v_{obs_name_for_plot} [$cm^3/2$]', fontsize=14)
            else:
                axs[row_idx, col_idx].plot(tSim, Pa_to_kPa*means_plot_gt[II, :], 'k', label='gt mean')
                axs[row_idx, col_idx].plot(tSim, Pa_to_kPa*means_plot_bf[II, :], 'b', label='bf mean')
                axs[row_idx, col_idx].plot(tSim, Pa_to_kPa*best_fit_obs[II, :], 'r', label='bf')
                axs[row_idx, col_idx].set_ylabel(f'P_{obs_name_for_plot} [$kPa$]', fontsize=14)

            axs[row_idx, col_idx].set_xlabel('Time [$s$]', fontsize=14)
            axs[row_idx, col_idx].set_xlim(0.0, self.param_id.sim_time)
            # axs[col_idx, row_idx].set_ylim(0.0, 20.0)
            # axs[row_idx, col_idx].set_yticks(np.arange(0, 21, 10))

            plot_saved = False
            col_idx = col_idx + 1
            if col_idx%subplot_width == 0:
                col_idx = 0
                row_idx += 1
                if row_idx%subplot_width == 0:
                    for JJ in range(subplot_width):
                        fig.align_ylabels(axs[:, JJ])
                    axs[0, 0].legend(loc='lower right', fontsize=6)
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.plot_dir,
                                             f'reconstruct_{self.param_id_method}_'
                                             f'{self.file_name_prefix}_{plot_idx}.eps'))
                    plt.savefig(os.path.join(self.plot_dir,
                                             f'reconstruct_{self.param_id_method}_'
                                             f'{self.file_name_prefix}_{plot_idx}.pdf'))
                    plot_saved = True
                    col_idx = 0
                    row_idx = 0
                    plot_idx += 1
                    # create new plot
                    if II != self.num_obs - 1:
                        fig, axs = plt.subplots(subplot_width, subplot_width)
                        plot_saved = False

        # save final plot if it is not a full set of subplots
        if not plot_saved:
            for JJ in range(subplot_width):
                fig.align_ylabels(axs[:, JJ])
            axs[0, 0].legend(loc='lower right', fontsize=6)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir,
                                     f'reconstruct_{self.param_id_method}_'
                                     f'{self.file_name_prefix}_{plot_idx}.eps'))
            plt.savefig(os.path.join(self.plot_dir,
                                     f'reconstruct_{self.param_id_method}_'
                                     f'{self.file_name_prefix}_{plot_idx}.pdf'))

    def set_genetic_algorithm_parameters(self, n_calls):
        self.param_id.set_genetic_algorithm_parameters(n_calls)

    def set_bayesian_parameters(self, n_calls, n_initial_points, acq_func, random_state, acq_func_kwargs={}):
        self.param_id.set_bayesian_parameters(n_calls, n_initial_points, acq_func, random_state,
                                              acq_func_kwargs=acq_func_kwargs)

    def close_simulation(self):
        self.param_id.close_simulation()

    def __set_and_save_param_names(self):
        # get the name of the vessel prior to the terminal
        vessel_array = genfromtxt(os.path.join(resources_dir, f'{self.file_name_prefix}_vessel_array.csv'),
                                  delimiter=',', dtype=None, encoding='UTF-8')[1:, :]
        vessel_array = np.array([[vessel_array[II, JJ].strip() for JJ in range(vessel_array.shape[1])]
                                 for II in range(vessel_array.shape[0])])

        terminal_names = vessel_array[np.where(vessel_array[:, 2] == 'terminal'), 0].flatten()
        num_terminals = len(terminal_names)
        terminal_names = [terminal_names[II].replace('_T', '') for II in range(num_terminals)]
        self.obs_nom_names = [f'v_nom_{a}' for a in terminal_names]
        # The below commented out command gets the vessels one segment prior to the terminal
        # self.obs_state_names = [vessel_array[np.where(vessel_array[:, 0] == terminal_names[II] + '_T'), 3][0][0] + '/v'
        #                    for II in range(len(terminal_names))]
        self.obs_state_names = [terminal_names[II] + '_T/v' for II in range(len(terminal_names))]
        # TODO get data for flows through time inorder to identify terminal compliances
        self.obs_alg_names = ['aortic_arch_C46/u']
        venous_names = vessel_array[np.where(vessel_array[:, 2] == 'venous'), 0].flatten()

        self.num_obs_states = len(self.obs_state_names)
        self.num_obs_algs = len(self.obs_alg_names)
        self.num_obs = self.num_obs_states + self.num_obs_algs

        # how much to weight the pressure measurement by
        pressure_weight = len(self.obs_state_names)
        self.weight_vec = np.ones(self.num_obs)
        self.weight_vec[-1] = float(pressure_weight)

        # Each entry in param_const_names is a name or list of names that gets modified by one parameter
        self.param_state_names = [['heart/q_lv']]
        # the param_*_for_gen stores the names of the constants as they are saved in the parameters csv file
        param_state_names_for_gen = [['q_lv']]
        # self.param_const_names = [[name + '/C' for name in venous_names]]
        self.param_const_names = []
        # param_const_names_for_gen = [['C_' + name for name in venous_names]]
        param_const_names_for_gen = []
        param_terminals = []
        param_terminals_for_gen = []
        same_group = False
        # get terminal parameter names
        for terminal_name in terminal_names:
            for idx, terminal_group in enumerate(param_terminals):
                # check if left or right of this terminal is already in param_terminals and add it to the
                # same group so that they have the same parameter identified
                if re.sub('_L?R?$', '', terminal_name) in terminal_group[0]:
                    param_terminals[idx].append(f'{terminal_name}_T/R_T')
                    param_terminals_for_gen[idx].append(f'R_T_{terminal_name}')
                    same_group = True
                    break
                else:
                    same_group = False
            if not same_group:
                param_terminals.append([f'{terminal_name}_T/R_T'])
                param_terminals_for_gen.append([f'R_T_{terminal_name}'])

        self.num_resistance_params = len(param_terminals)
        self.param_const_names += param_terminals

        param_const_names_for_gen += param_terminals_for_gen

        if self.rank == 0:
            with open(os.path.join(self.output_dir, 'param_state_names.csv'), 'w') as f:
                wr = csv.writer(f)
                wr.writerows(self.param_state_names)
            with open(os.path.join(self.output_dir, 'param_state_names_for_gen.csv'), 'w') as f:
                wr = csv.writer(f)
                wr.writerows(param_state_names_for_gen)
            with open(os.path.join(self.output_dir, 'param_const_names.csv'), 'w') as f:
                wr = csv.writer(f)
                wr.writerows(self.param_const_names)
            with open(os.path.join(self.output_dir, 'param_const_names_for_gen.csv'), 'w') as f:
                wr = csv.writer(f)
                wr.writerows(param_const_names_for_gen)

        return

    def __get_ground_truth_values(self):

        if self.rank == 0:
            # _______ First we access data for mean values

            data_array = genfromtxt(os.path.join(resources_dir, f'parameters_orig.csv'),
                                    delimiter=',', dtype=None, encoding='UTF-8')[1:, :]

            ground_truth_mean_flows = np.zeros(self.num_obs_states)
            for idx, name in enumerate(self.obs_nom_names):
                ground_truth_mean_flows[idx] = data_array[:, 3][np.where(data_array[:, 0] == name)][0].astype(float)

            ground_truth_mean_pressures = np.array([12000])

            ground_truth = np.concatenate([ground_truth_mean_flows, ground_truth_mean_pressures])

            np.save(os.path.join(self.output_dir, 'ground_truth'), ground_truth)

        else:
            ground_truth = np.zeros(self.num_obs)
            pass

        if self.num_procs > 1:
            self.comm.Bcast(ground_truth, root=0)

        return ground_truth


class OpencorParamID():
    """
    Class for doing parameter identification on opencor models
    """
    def __init__(self, model_path, param_id_method,
                 obs_state_names, obs_alg_names, weight_vec,
                 param_state_names, param_const_names, ground_truth,
                 param_mins, param_maxs, sim_time=2.0, pre_time=20.0):

        self.model_path = model_path
        self.param_id_method = param_id_method
        self.output_dir = None

        self.obs_state_names = obs_state_names
        self.obs_alg_names = obs_alg_names
        self.weight_vec = weight_vec
        self.param_state_names = param_state_names
        self.param_const_names = param_const_names
        self.num_obs_states = len(self.obs_state_names)
        self.num_obs_algs = len(self.obs_alg_names)
        self.num_obs = self.num_obs_states + self.num_obs_algs
        self.num_params = len(self.param_state_names) + len(self.param_const_names)
        self.ground_truth = ground_truth
        self.param_mins = param_mins
        self.param_maxs = param_maxs


        # set up opencor simulation
        self.dt = 0.01  # TODO this could be optimised
        self.point_interval = self.dt
        self.sim_time = sim_time
        self.pre_time = pre_time
        self.nSteps = int(self.sim_time/self.dt)
        self.sim_helper = self.initialise_sim_helper()

        # initialise
        self.param_init = None
        self.best_param_vals = None
        self.best_cost = 999999

        # genetic algorithm constants TODO add more of the constants to this so they can be modified by the user
        self.n_calls = 100
        self.acq_func = 'EI'  # the acquisition function
        self.n_initial_points = 5
        self.acq_func_kwargs = {}
        self.random_state = 1234 # random seed

    def initialise_sim_helper(self):
        return SimulationHelper(self.model_path, self.dt, self.sim_time,
                                self.point_interval, maximumNumberofSteps=100000000,
                                maximumStep=0.0004, pre_time=self.pre_time)

    def run(self):

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        num_procs = comm.Get_size()

        if rank == 0:
          # save date as identifier for the param_id
          np.save(os.path.join(self.output_dir, 'date'), date.today().strftime("%d_%m_%Y"))

        print('starting param id run for rank = {} process'.format(rank))

        # ________ Do parameter identification ________

        self.param_init = self.sim_helper.get_init_param_vals(self.param_state_names, self.param_const_names)

        # C_T min and max was 1e-9 and 1e-5 before

        param_norm_obj = Normalise_class(self.param_mins, self.param_maxs)

        cost_convergence = 0.0001
        if self.param_id_method == 'bayesian':
            if rank == 0:
                print('Running bayesian optimisation')
            param_ranges = [a for a in zip(self.param_mins, self.param_maxs)]
            updated_version = True # TODO remove this and remove the gp_minimize version
            if not updated_version:
                res = gp_minimize(self.get_cost,  # the function to minimize
                                  param_ranges,  # the bounds on each dimension of x
                                  acq_func=self.acq_func,  # the acquisition function
                                  n_calls=self.n_calls,  # the number of evaluations of f
                                  n_initial_points=self.n_initial_points,  # the number of random initialization points
                                  random_state=self.random_state, # random seed
                                  **self.acq_func_kwargs,
                                  callback=[ProgressBar(self.n_calls)])
                              # noise=0.1**2,  # the noise level (optional)
            else:
                # using Optimizer is more flexible and may be needed to implement a parallel usage
                # gp_minimizer is a higher level call that uses Optimizer
                if rank == 0:
                    opt = Optimizer(param_ranges,  # the bounds on each dimension of x
                                    base_estimator='GP', # gaussian process
                                    acq_func=self.acq_func,  # the acquisition function
                                    n_initial_points=self.n_initial_points,  # the number of random initialization points
                                    random_state=self.random_state, # random seed
                                    acq_func_kwargs=self.acq_func_kwargs,
                                    n_jobs=num_procs)


                progress_bar = ProgressBar(self.n_calls, n_jobs=num_procs)
                call_num = 0
                iter_num = 0
                while call_num < self.n_calls:
                    cost = np.zeros(num_procs)
                    if rank == 0:
                        points = opt.ask(n_points=num_procs)
                        points_np = np.array(points)
                    else:
                        points_np = np.zeros((num_procs, self.num_params))

                    # broadcast points so every processor has all of the points. TODO This could be optimized for memory
                    comm.Bcast(points_np, root=0)
                    cost_proc = self.get_cost(points_np[rank, :])
                    print(f'cost for rank = {rank} is {cost_proc}')

                    recv_buf_cost = np.zeros(num_procs)
                    send_buf_cost = cost_proc
                    # gather results from simulation
                    comm.Gatherv(send_buf_cost, [recv_buf_cost, 1,
                                                  None, MPI.DOUBLE], root=0)
                    cost_np = recv_buf_cost
                    cost = cost_np.tolist()

                    if rank == 0:
                        opt.tell(points, cost)
                        res = opt.get_result()
                        progress_bar.call(res)

                        if res.fun < self.best_cost and iter_num > 20:
                            # save if cost improves and its not right at the start
                            self.best_cost = res.fun
                            self.best_param_vals = res.x
                            print('parameters improved! SAVING COST AND PARAM VALUES')
                            np.save(os.path.join(self.output_dir, 'best_cost'), self.best_cost)
                            np.save(os.path.join(self.output_dir, 'best_param_vals'), self.best_param_vals)

                    call_num += num_procs
                    iter_num += 1

                    # TODO save results here every few iterations


            if rank == 0:
                print(res)
                self.best_cost = res.fun
                self.best_param_vals = res.x
                np.save(os.path.join(self.output_dir, 'best_cost'), self.best_cost)
                np.save(os.path.join(self.output_dir, 'best_param_vals'), self.best_param_vals)

        elif self.param_id_method == 'genetic_algorithm':
            num_elite = 3
            num_survivors = 12
            num_mutations_per_survivor = 12
            num_cross_breed = 30
            num_pop = num_survivors + num_survivors*num_mutations_per_survivor + \
                   num_cross_breed
            if self.n_calls < num_pop:
                print(f'Number of calls (n_calls={self.n_calls}) must be greater than the '
                      f'gen alg population (num_pop={num_pop}), exiting')
                exit()
            self.max_generations = math.floor(self.n_calls/num_pop)
            if rank == 0:
                print(f'Running genetic algorithm with a population size of {num_pop},\n'
                      f'and a maximum number of generations of {self.max_generations}')
            simulated_bools = [False]*num_pop
            mutation_weight = 0.01
            gen_count = 0

            if rank == 0:
                param_vals_norm = np.random.rand(self.num_params, num_pop)
                param_vals = param_norm_obj.unnormalise(param_vals_norm)
            else:
                param_vals = None

            cost = np.zeros(num_pop)
            cost[0] = 9999

            while cost[0] > cost_convergence and gen_count < self.max_generations:
                # TODO make these modifiable to the user
                if gen_count > 20:
                    mutation_weight = 0.01
                elif gen_count > 40:
                    mutation_weight = 0.005
                elif gen_count > 80:
                    mutation_weight = 0.002
                elif gen_count > 120:
                    mutation_weight = 0.001

                gen_count += 1
                if rank == 0:
                    print('generation num: {}'.format(gen_count))
                    # check param_vals are within bounds and if not set them to the bound
                    for II in range(self.num_params):
                        for JJ in range(num_pop):
                            if param_vals[II, JJ] < self.param_mins[II]:
                                param_vals[II, JJ] = self.param_mins[II]
                            elif param_vals[II, JJ] > self.param_maxs[II]:
                                param_vals[II, JJ] = self.param_maxs[II]

                    send_buf = param_vals.T.copy()
                    send_buf_cost = cost
                    send_buf_bools = np.array(simulated_bools)
                    # count number of columns for each proc
                    # count: the size of each sub-task
                    ave, res = divmod(param_vals.shape[1], num_procs)
                    pop_per_proc = np.array([ave + 1 if p < res else ave for p in range(num_procs)])
                else:
                    pop_per_proc = np.empty(num_procs, dtype=int)
                    send_buf = None
                    send_buf_bools = None
                    send_buf_cost = None

                comm.Bcast(pop_per_proc, root=0)
                # initialise receive buffer for each proc
                recv_buf = np.zeros((pop_per_proc[rank], self.num_params))
                recv_buf_bools = np.empty(pop_per_proc[rank], dtype=bool)
                recv_buf_cost = np.zeros(pop_per_proc[rank])
                # scatter arrays to each proc
                comm.Scatterv([send_buf, pop_per_proc*self.num_params, None, MPI.DOUBLE],
                              recv_buf, root=0)
                param_vals_proc = recv_buf.T.copy()
                comm.Scatterv([send_buf_bools, pop_per_proc, None, MPI.BOOL],
                              recv_buf_bools, root=0)
                bools_proc = recv_buf_bools
                comm.Scatterv([send_buf_cost, pop_per_proc, None, MPI.DOUBLE],
                              recv_buf_cost, root=0)
                cost_proc = recv_buf_cost

                if rank == 0 and gen_count == 1:
                    print('population per processor is')
                    print(pop_per_proc)

                # each processor runs until all param_val_proc sets have been simulated succesfully
                success = False
                while not success:
                    for II in range(pop_per_proc[rank]):
                        if bools_proc[II]:
                            # this individual has already been simulated
                            success = True
                            continue
                        # set params for this case
                        self.sim_helper.set_param_vals(self.param_state_names, self.param_const_names,
                                                       param_vals_proc[:, II])

                        success = self.sim_helper.run()
                        if success:
                            pred_obs = self.sim_helper.get_results(self.obs_state_names, self.obs_alg_names)
                            pred_obs_mean = np.mean(pred_obs, axis=1)
                            # calculate error between the observables of this set of parameters
                            # and the ground truth
                            cost_proc[II] = np.sum(
                                np.power(self.weight_vec*(pred_obs_mean -
                                                          self.ground_truth)/self.ground_truth, 2))/(self.num_obs)
                            # reset params
                            self.sim_helper.reset()

                        else:
                            # simulation failed, choose a new random point
                            print('simulation failed with params...')
                            print(param_vals_proc[:, II])
                            print('... choosing a new random point')
                            param_vals_proc[:, II:II + 1] = param_norm_obj.unnormalise(np.random.rand(self.num_params, 1))
                            cost_proc[II] = 9999
                            break

                        simulated_bools[II] = True
                        if num_procs == 1:
                            if II%5 == 0 and II > num_survivors:
                                print(' this generation is {:.0f}% done'.format(100.0*(II + 1)/pop_per_proc[0]))
                        else:
                            if rank == num_procs - 1:
                                # if II%4 == 0 and II != 0:
                                print(' this generation is {:.0f}% done'.format(100.0*(II + 1)/pop_per_proc[0]))

                recv_buf = np.zeros((num_pop, self.num_params))
                recv_buf_cost = np.zeros(num_pop)
                send_buf = param_vals_proc.T.copy()
                send_buf_cost = cost_proc
                # gather results from simulation
                comm.Gatherv(send_buf, [recv_buf, pop_per_proc*self.num_params,
                                         None, MPI.DOUBLE], root=0)
                comm.Gatherv(send_buf_cost, [recv_buf_cost, pop_per_proc,
                                              None, MPI.DOUBLE], root=0)

                if rank == 0:
                    param_vals = recv_buf.T.copy()
                    cost = recv_buf_cost

                    # order the vertices in order of cost
                    order_indices = np.argsort(cost)
                    cost = cost[order_indices]
                    param_vals = param_vals[:, order_indices]
                    print('Cost of first 10 of population : {}'.format(cost[:10]))
                    param_vals_norm = param_norm_obj.normalise(param_vals)
                    print('worst survivor params normed : {}'.format(param_vals_norm[:, num_survivors - 1]))
                    print('best params normed : {}'.format(param_vals_norm[:, 0]))
                    np.save(os.path.join(self.output_dir, 'best_cost'), cost[0])
                    np.save(os.path.join(self.output_dir, 'best_param_vals'), param_vals[:, 0])

                    # At this stage all of the population has been simulated
                    simulated_bools = [True]*num_pop
                    # keep the num_survivors best param_vals, replace these with mutations
                    param_idx = num_elite

                    for idx in range(num_elite, num_survivors):
                        # TODO make the below depend on probability (normalised cost function)
                        rand_survivor_idx = np.random.randint(num_elite, num_pop)
                        param_vals_norm[:, param_idx] = param_vals_norm[:, rand_survivor_idx]

                        param_idx += 1

                    for survivor_idx in range(num_survivors):
                        for JJ in range(num_mutations_per_survivor):
                            simulated_bools[param_idx] = False
                            param_vals_norm[:, param_idx] = param_vals_norm[:, survivor_idx] + \
                                                            mutation_weight*np.random.randn(self.num_params)
                            param_idx += 1

                    # now do cross breeding
                    cross_breed_indices = np.random.randint(0, num_survivors, (num_cross_breed, 2))
                    for couple in cross_breed_indices:
                        if couple[0] == couple[1]:
                            couple[1] += 1  # this allows crossbreeding out of the survivors but that's ok
                        simulated_bools[param_idx] = False
                        param_vals_norm[:, param_idx] = (param_vals_norm[:, couple[0]] +
                                                         param_vals_norm[:, couple[1]])/2 + \
                                                        mutation_weight*np.random.randn(self.num_params)
                        param_idx += 1

                    param_vals = param_norm_obj.unnormalise(param_vals_norm)

                else:
                    # non zero ranks don't do any of the ordering or mutations
                    pass

            if rank == 0:
                self.best_cost = cost[0]
                self.best_param_vals = param_vals[:, 0]

        else:
            print(f'param_id_method {self.param_id_method} hasn\'t been implemented')
            exit()

        if rank == 0:
            print('')
            print('parameter identification is complete')
            # print init params and final params
            print('init params     : {}'.format(self.param_init))
            print('best fit params : {}'.format(self.best_param_vals))
            print('best cost       : {}'.format(self.best_cost))

        return

    def get_cost(self, param_vals):

        # set params for this case
        self.sim_helper.set_param_vals(self.param_state_names, self.param_const_names,
                                       param_vals)

        success = self.sim_helper.run()
        if success:
            pred_obs = self.sim_helper.get_results(self.obs_state_names, self.obs_alg_names)
            pred_obs_mean = np.mean(pred_obs, axis=1)
            # calculate error between the observables of this set of parameters
            # and the ground truth
            cost = np.sum(np.power(self.weight_vec*(pred_obs_mean -
                                          self.ground_truth)/self.ground_truth, 2))/(self.num_obs)
            # reset params
            self.sim_helper.reset()

        else:
            # simulation set cost to large,
            print('simulation failed with params...')
            print(param_vals)
            cost = 9999

        return cost

    def simulate_once(self):

        if MPI.COMM_WORLD.Get_rank() != 0:
            print('simulate once should only be done on one rank')
            exit()
        if not self.best_cost:
            self.best_cost = np.load(os.path.join(self.output_dir, 'best_cost.npy'))
            self.best_param_vals = np.load(os.path.join(self.output_dir, 'best_param_vals.npy'))

        # ___________ Run model with new parameters ________________

        self.sim_helper.update_times(self.dt, 0.0, self.sim_time, self.pre_time)
        # set params to best fit from param id
        self.sim_helper.set_param_vals(self.param_state_names, self.param_const_names, self.best_param_vals)
        self.sim_helper.run()

        #check cost
        pred_obs = self.sim_helper.get_results(self.obs_state_names, self.obs_alg_names)
        pred_obs_mean = np.mean(pred_obs, axis=1)
        cost_check = np.sum(np.power(self.weight_vec*(pred_obs_mean -
                                                      self.ground_truth)/self.ground_truth, 2))/(self.num_obs)
        print(f'cost should be {self.best_cost}')
        print('cost check after single simulation is {}'.format(cost_check))

        # TODO remove the below print
        print(f'final pressure mean = {pred_obs_mean[-1]}')

    def set_genetic_algorithm_parameters(self, n_calls):
        if not self.param_id_method == 'genetic_algorithm':
            print('param_id is not set up as a genetic algorithm')
            exit()
        self.n_calls= n_calls
        # TODO add more of the gen alg constants here so they can be changed by user.

    def set_bayesian_parameters(self, n_calls, n_initial_points, acq_func, random_state, acq_func_kwargs={}):
        if not self.param_id_method == 'bayesian':
            print('param_id is not set up as a bayesian optimization process')
            exit()
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.acq_func = acq_func  # the acquisition function
        self.random_state = random_state  # random seed
        self.acq_func_kwargs = acq_func_kwargs
        # TODO add more of the gen alg constants here so they can be changed by user.

    def close_simulation(self):
        self.sim_helper.close_simulation()

    def set_output_dir(self, output_dir):
        self.output_dir = output_dir


class ProgressBar(object):
    """
    Alternatively: Could call ProgBarLogger like in keras
    """

    def __init__(self, n_calls, n_jobs=1, file=sys.stderr):
        self.n_calls = n_calls
        self.n_jobs = n_jobs
        self.iter_no = 0
        self.file = file
        self._start_time = time.time()

    def _to_precision(self, x, precision=5):
        return ("{0:.%ie} seconds"%(precision - 1)).format(x)

    def progress(self, iter_no, curr_min):
        bar_len = 60
        filled_len = int(round(bar_len*iter_no/float(self.n_calls)))

        percents = round(100.0*iter_no/float(self.n_calls), 1)
        bar = '='*filled_len + '-'*(bar_len - filled_len)
        print(f'[{bar}] {percents}% | Elapsed Time: {time.time() - self._start_time} | Current Minimum: {curr_min}')

    def __call__(self, res):
        curr_y = res.func_vals[-1]
        curr_min = res.fun
        self.iter_no += self.n_jobs
        self.progress(self.iter_no, curr_min)

    def call(self, res):
        self.__call__(res)

