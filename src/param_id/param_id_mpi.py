import numpy as np
import os
import sys
from sys import exit
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utilities'))
resources_dir = os.path.join(os.path.dirname(__file__), '../../resources')
import math as math
import opencor as oc
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

if __name__ == '__main__':

    case_name = 'simple_physiological'
    # case_name = 'physiological'
    # TODO turn the below into a user defined input
    generated_models_dir = os.path.join(os.path.dirname(__file__), '../../generated_models')
    do_param_id = False
    mpi_debug = False

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()
    if num_procs == 1:
        print('WARNING Running in serial, are you sure you want to be a snail?')

    # FOR MPI DEBUG WITH PYCHARM
    # You have to change the configurations to "python debug server/mpi" and
    # click the debug button as many times as processes you want. You
    # must but the ports for each process in port_mapping.
    if mpi_debug:
        import pydevd_pycharm
        port_mapping = [36939, 44271, 33017, 46467]
        pydevd_pycharm.settrace('localhost', port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)

    print('starting rank = {} process'.format(rank))

    m3_to_cm3 = 1e6
    Pa_to_kPa = 1e-3
    # case_type = 'Nelder_Meade' # not set up for mpi
    case_type = f'genetic_algorithm_{case_name}'
    if rank == 0:
        param_id_output_dir = os.path.join(os.path.dirname(__file__), '../../param_id_output')
        if not os.path.exists(param_id_output_dir):
            os.mkdir(param_id_output_dir)
        output_dir = os.path.join(param_id_output_dir, f'{case_type}')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        plot_dir = os.path.join(output_dir, 'plots_param_id')
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)
    
    comm.Barrier()

    dt = 0.01  # TODO this could be optimised
    point_interval = dt
    sim_time = 2.0
    pre_time = 12.0
    nSteps = int(sim_time/dt)

    # get the name of the vessel prior to the terminal
    vessel_array = genfromtxt(os.path.join(resources_dir, f'{case_name}_vessel_array.csv'),
                            delimiter=',', dtype=None, encoding='UTF-8')[1:, :]
    vessel_array = np.array([[vessel_array[II, JJ].strip() for JJ in range(vessel_array.shape[1])]
                             for II in range(vessel_array.shape[0])])

    terminal_names = vessel_array[np.where(vessel_array[:, 2] == 'terminal'), 0].flatten()
    num_terminals = len(terminal_names)
    terminal_names = [terminal_names[II].replace('_T', '') for II in range(num_terminals)]
    obs_nom_names = [f'v_nom_{a}' for a in terminal_names]
    obs_state_names = [vessel_array[np.where(vessel_array[:, 0] == terminal_names[II]+'_T'), 3][0][0]+'/v'
                       for II in range(len(terminal_names))]
    # TODO get data for flows through time inorder to identify terminal compliances
    obs_alg_names = ['aortic_arch_C46/u']
    venous_names = vessel_array[np.where(vessel_array[:, 2] == 'venous'), 0].flatten()

    num_obs_states = len(obs_state_names)
    num_obs_algs = len(obs_alg_names)
    num_obs = num_obs_states + num_obs_algs

    get_ground_truth = True

    if rank == 0:
        if get_ground_truth:
            # _______ First we access data for mean values

            data_array = genfromtxt(os.path.join(resources_dir, f'parameters_orig.csv'),
                delimiter=',', dtype=None, encoding='UTF-8')[1:, :]

            ground_truth_mean_flows = np.zeros(num_obs_states)
            for idx, name in enumerate(obs_nom_names):

                ground_truth_mean_flows[idx] = data_array[:,3][np.where(data_array[:, 0] == name)][0].astype(float)

            ground_truth_mean_pressures = np.array([13300])

            ground_truth = np.concatenate([ground_truth_mean_flows, ground_truth_mean_pressures])

            np.save(os.path.join(output_dir, 'ground_truth'), ground_truth)

        else:
            ground_truth = np.load(os.path.join(output_dir, 'ground_truth.npy'))
    else:
        ground_truth = np.zeros(num_obs)
        pass
    if num_procs > 1:
        comm.Bcast(ground_truth, root=0)
    
    # how much to weight the pressure measurement by
    pressure_weight = len(obs_state_names)
    weight_vec = np.ones(num_obs)
    weight_vec[-1] = pressure_weight

    # ________ Do parameter identification ________

    # set up ADAN-218 model
    sim_helper = SimulationHelper(os.path.join(generated_models_dir, f'{case_name}.cellml'), dt, sim_time,
                                  point_interval, maximumNumberofSteps=100000000,
                                  maximumStep=0.1, pre_time=pre_time)

    # Each entry in param_const_names is a name or list of names that gets modified by one parameter
    param_state_names = [['heart/q_lv']]
    # the param_*_for_gen stores the names of the constants as they are saved in the parameters csv file
    param_state_names_for_gen = [['q_lv']]
    # param_const_names = [['venous_ub/C', 'venous_lb/C',
    #                       'venous_svc/C', 'venous_ivc/C']]
    param_const_names = [[name + '/C' for name in venous_names]]
    # param_const_names_for_gen = [['C_venous_ub', 'C_venous_lb',
    #                               'C_venous_svc', 'C_venous_ivc']]
    param_const_names_for_gen = [['C_' + name for name in venous_names]]
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

    num_resistance_params = len(param_terminals)
    param_const_names += param_terminals

    param_const_names_for_gen += param_terminals_for_gen

    if rank == 0:
        with open(os.path.join(output_dir, 'param_state_names.csv'), 'w') as f:
            wr = csv.writer(f)
            wr.writerows(param_state_names)
        with open(os.path.join(output_dir, 'param_state_names_for_gen.csv'), 'w') as f:
            wr = csv.writer(f)
            wr.writerows(param_state_names_for_gen)
        with open(os.path.join(output_dir, 'param_const_names.csv'), 'w') as f:
            wr = csv.writer(f)
            wr.writerows(param_const_names)
        with open(os.path.join(output_dir, 'param_const_names_for_gen.csv'), 'w') as f:
            wr = csv.writer(f)
            wr.writerows(param_const_names_for_gen)

        # save date as identifier for the param_id
        np.save(os.path.join(output_dir, 'date'), date.today().strftime("%d_%m_%Y"))

    param_names = param_state_names + param_const_names

    nParams = len(param_names)

    param_init = sim_helper.get_init_param_vals(param_state_names, param_const_names)

    # define allowed param ranges
    param_mins = np.array([400e-6, 1e-9] + [5e6]*num_resistance_params)
    param_maxs = np.array([2400e-6, 1e-6] + [1e10]*num_resistance_params)

    # C_T min and max was 1e-9 and 1e-5 before

    param_norm_obj = Normalise_class(param_mins, param_maxs)

    cost_convergence = 0.0001
    if not do_param_id:
        pass
    elif case_type.startswith('genetic_algorithm'):
        nElite = 3
        nSurvivors = 12
        nMutations_per_survivor = 12
        nCrossBreed = 30
        nPop = nSurvivors + nSurvivors*nMutations_per_survivor + \
               nCrossBreed
        if rank == 0:
            print('Running genetic algorithm with a population size of {}'.format(nPop))
        simulated_bools = [False]*nPop
        mutation_weight = 0.01
        gen_count = 0

        if rank == 0:
            param_vals_norm = np.random.rand(nParams, nPop)
            param_vals = param_norm_obj.unnormalise(param_vals_norm)
        else:
            param_vals = None

        cost = np.zeros(nPop)
        cost[0] = 9999

        while cost[0] > cost_convergence and gen_count < 180:
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
                for II in range(nParams):
                    for JJ in range(nPop):
                        if param_vals[II, JJ] < param_mins[II]:
                            param_vals[II, JJ] = param_mins[II]
                        elif param_vals[II, JJ] > param_maxs[II]:
                            param_vals[II, JJ] = param_maxs[II]

                send_bufr = param_vals.T.copy()
                send_bufr_cost = cost
                send_bufr_bools = np.array(simulated_bools)
                # count number of columns for each proc
                # count: the size of each sub-task
                ave, res = divmod(param_vals.shape[1], num_procs)
                pop_per_proc = np.array([ave + 1 if p < res else ave for p in range(num_procs)])
            else:
                pop_per_proc = np.empty(num_procs, dtype=int)
                send_bufr = None
                send_bufr_bools = None
                send_bufr_cost = None

            comm.Bcast(pop_per_proc, root=0)
            # initialise receive buffer for each proc
            recv_buf = np.zeros((pop_per_proc[rank], nParams))
            recv_buf_bools = np.empty(pop_per_proc[rank], dtype=bool)
            recv_buf_cost = np.zeros(pop_per_proc[rank])
            # scatter arrays to each proc
            comm.Scatterv([send_bufr, pop_per_proc*nParams, None, MPI.DOUBLE],
                          recv_buf, root=0)
            param_vals_proc = recv_buf.T.copy()
            comm.Scatterv([send_bufr_bools, pop_per_proc, None, MPI.BOOL],
                          recv_buf_bools, root=0)
            bools_proc = recv_buf_bools
            comm.Scatterv([send_bufr_cost, pop_per_proc, None, MPI.DOUBLE],
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
                    sim_helper.set_param_vals(param_state_names, param_const_names, param_vals_proc[:, II])

                    success = sim_helper.run()
                    if success:
                        pred_obs = sim_helper.get_results(obs_state_names, obs_alg_names)
                        pred_obs_mean = np.mean(pred_obs, axis=1)
                        # calculate error between the observables of this set of parameters
                        # and the ground truth
                        cost_proc[II] = np.sum(np.power(weight_vec*(pred_obs_mean - ground_truth)/ground_truth, 2))/(num_obs)
                        # reset params
                        sim_helper.reset()

                    else:
                        # simulation failed, choose a new random point
                        print('simulation failed with params...')
                        print(param_vals_proc[:, II])
                        print('... choosing a new random point')
                        param_vals_proc[:, II:II+1] = param_norm_obj.unnormalise(np.random.rand(nParams, 1))
                        cost_proc[II] = 9999
                        break

                    simulated_bools[II] = True
                    if num_procs == 1:
                        if II%5 == 0 and II > nSurvivors:
                            print(' this generation is {:.0f}% done'.format(100.0*(II+1)/pop_per_proc[0]))
                    else:
                      if rank == num_procs-1:
                            #if II%4 == 0 and II != 0:
                            print(' this generation is {:.0f}% done'.format(100.0*(II+1)/pop_per_proc[0]))

            recv_buf = np.zeros((nPop, nParams))
            recv_buf_cost = np.zeros(nPop)
            send_bufr = param_vals_proc.T.copy()
            send_bufr_cost = cost_proc
            # gather results from simulation
            comm.Gatherv(send_bufr, [recv_buf, pop_per_proc*nParams,
                                     None, MPI.DOUBLE], root=0)
            comm.Gatherv(send_bufr_cost, [recv_buf_cost, pop_per_proc,
                                          None, MPI.DOUBLE], root=0)

            if rank==0:
                param_vals = recv_buf.T.copy()
                cost = recv_buf_cost

                # order the vertices in order of cost
                order_indices = np.argsort(cost)
                cost = cost[order_indices]
                param_vals = param_vals[:, order_indices]
                print('Cost of first 10 of population : {}'.format(cost[:10]))
                param_vals_norm = param_norm_obj.normalise(param_vals)
                print('worst survivor params normed : {}'.format(param_vals_norm[:, nSurvivors-1]))
                print('best params normed : {}'.format(param_vals_norm[:, 0]))
                np.save(os.path.join(output_dir, 'best_cost'), cost[0])
                np.save(os.path.join(output_dir, 'best_param_vals'), param_vals[:, 0])

                # At this stage all of the population has been simulated
                simulated_bools = [True]*nPop
                # keep the nSurvivors best param_vals, replace these with mutations
                param_idx = nElite

                for idx in range(nElite, nSurvivors):
                    # TODO make the below depend on probability (normalised cost function)
                    rand_survivor_idx = np.random.randint(nElite, nPop)
                    param_vals_norm[:, param_idx] = param_vals_norm[:, rand_survivor_idx]

                    param_idx += 1

                for survivor_idx in range(nSurvivors):
                    for JJ in range(nMutations_per_survivor):
                        simulated_bools[param_idx] = False
                        param_vals_norm[:, param_idx] = param_vals_norm[:, survivor_idx] + \
                                                        mutation_weight*np.random.randn(nParams)
                        param_idx += 1

                # now do cross breeding
                crossBreed_indices = np.random.randint(0, nSurvivors, (nCrossBreed, 2))
                for couple in crossBreed_indices:
                        if couple[0] == couple[1]:
                            couple[1] += 1 # this allows crossbreeding out of the survivors but that's ok
                        simulated_bools[param_idx] = False
                        param_vals_norm[:, param_idx] = (param_vals_norm[:, couple[0]] +
                                                        param_vals_norm[:, couple[1]])/2 + \
                                                        mutation_weight*np.random.randn(nParams)
                        param_idx += 1

                param_vals = param_norm_obj.unnormalise(param_vals_norm)

            else:
                # non zero ranks don't do any of the ordering or mutations
                pass

    else:
        print('case_type of {} hasn\'t been implemented'.format(case_type))
        exit()

    if rank == 0:
        if do_param_id:
            best_cost = cost[0]
            best_param_vals = param_vals[:, 0]
        else:
            best_cost = np.load(os.path.join(output_dir, 'best_cost.npy'))
            best_param_vals = np.load(os.path.join(output_dir, 'best_param_vals.npy'))

        # print init params and final params
        print('init params     : {}'.format(param_init))
        print('best fit params : {}'.format(best_param_vals))
        # ___________ Run model with new parameters ________________
        pre_time = 12.0

        sim_helper.update_times(dt, 0.0, sim_time, pre_time)
        # set params to best fit from param id
        sim_helper.set_param_vals(param_state_names, param_const_names, best_param_vals)
        sim_helper.run()

        #check cost
        pred_obs = sim_helper.get_results(obs_state_names, obs_alg_names)
        pred_obs_mean = np.mean(pred_obs, axis=1)
        cost_check = np.sum(np.power(weight_vec*(pred_obs_mean - ground_truth)/ground_truth, 2))/(num_obs)
        print('cost check is {}'.format(cost_check))

        obs_state_names_plot = ['aortic_arch_C46/v',
                         'common_carotid_L48_A/v',
                         'femoral_L200/v',
                         'subclavian_R28/v']

        obs_alg_names_plot = ['aortic_arch_C46/u',
                       'common_carotid_L48_A/u']

        num_obs_states_plot = len(obs_state_names_plot)
        num_obs_algs_plot = len(obs_alg_names_plot)

        AA_v_idx = 0
        LCC_v_idx = 1
        FEM_v_idx = 2
        RSC_v_idx = 3
        AA_u_idx = num_obs_states_plot
        LCC_u_idx = num_obs_states_plot + 1

        LCC_v_idx_gt = np.where(np.array(obs_state_names) == obs_state_names_plot[LCC_v_idx])[0][0]
        FEM_v_idx_gt = np.where(np.array(obs_state_names) == obs_state_names_plot[FEM_v_idx])[0][0]
        RSC_v_idx_gt = np.where(np.array(obs_state_names) == obs_state_names_plot[RSC_v_idx])[0][0]

        best_fit_obs = sim_helper.get_results(obs_state_names_plot, obs_alg_names_plot)
        best_fit_obs_means = np.mean(best_fit_obs, axis=1)
        sim_helper.close_simulation()

        # TODO save all results for plotting
        # _________ Plot best comparison _____________
        fig, axs = plt.subplots(2, 2)

        tSim = sim_helper.tSim - pre_time
        means_plot_gt = np.tile(ground_truth.reshape(-1,1), (1,sim_helper.nSteps))
        means_plot_bf = np.tile(best_fit_obs_means.reshape(-1,1), (1,sim_helper.nSteps))
        axs[0, 0].plot(tSim, Pa_to_kPa*means_plot_gt[-1, :], 'k', label='ground truth mean')
        axs[0, 0].plot(tSim, Pa_to_kPa*means_plot_bf[AA_u_idx, :], 'b', label='best fit mean')
        axs[0, 0].plot(tSim, Pa_to_kPa*best_fit_obs[AA_u_idx, :], 'r', label='best fit waveform')
        axs[0, 0].set_xlabel('Time [$s$]', fontsize=14)
        axs[0, 0].set_ylabel('AA Pressure [$kPa$]', fontsize=14)
        axs[0, 0].set_xlim(0.0, sim_time)
        axs[0, 0].set_ylim(0.0, 20.0)
        axs[0, 0].set_yticks(np.arange(0, 21, 10))

        axs[0, 1].plot(tSim, Pa_to_kPa*best_fit_obs[LCC_u_idx, :], 'r')
        axs[0, 1].set_xlabel('Time [$s$]', fontsize=14)
        axs[0, 1].set_ylabel('LCC Pressure [$kPa$]', fontsize=14)
        axs[0, 1].set_xlim(0.0, sim_time)
        axs[0, 1].set_ylim(0.0, 20.0)
        axs[0, 1].set_yticks(np.arange(0, 21, 10))

        axs[1, 0].plot(tSim, m3_to_cm3*best_fit_obs[AA_v_idx, :], 'r')
        axs[1, 0].set_xlabel('Time [$s$]', fontsize=14)
        axs[1, 0].set_ylabel('AA Flow [$cm^3/s$]', fontsize=14)
        axs[1, 0].set_xlim(0.0, sim_time)
        axs[1, 0].set_ylim(-50.0, 400)
        axs[1, 0].set_yticks(np.arange(0, 401, 100))

        axs[1, 1].plot(tSim, m3_to_cm3*best_fit_obs[LCC_v_idx, :], 'r')
        axs[1, 1].set_xlabel('Time [$s$]', fontsize=14)
        axs[1, 1].set_ylabel('LCC Flow [$cm^3/s$]', fontsize=14)
        axs[1, 1].set_xlim(0.0, sim_time)
        # axs[1, 1].set_ylim(0.0, 15)
        # axs[1, 1].set_yticks(np.arange(0, 15.1, 5))

        fig.align_ylabels(axs[:, 0])
        fig.align_ylabels(axs[:, 1])
        axs[0, 0].legend(loc='lower right', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'reconstruct_{}.eps'.format(case_type)))
        plt.savefig(os.path.join(plot_dir, 'reconstruct_{}.pdf'.format(case_type)))

        fig, axs = plt.subplots(2, 2)

        axs[0, 0].plot(tSim, m3_to_cm3*means_plot_gt[LCC_v_idx_gt, :], 'k', label='ground truth mean')
        axs[0, 0].plot(tSim, m3_to_cm3*means_plot_bf[LCC_v_idx, :], 'b', label='best fit mean')
        axs[0, 0].plot(tSim, m3_to_cm3*best_fit_obs[LCC_v_idx, :], 'r', label='best fit waveform')
        axs[0, 0].set_xlabel('Time [$s$]', fontsize=14)
        axs[0, 0].set_ylabel('LCC Flow [$cm^3/s$]', fontsize=14)
        axs[0, 0].set_xlim(0.0, sim_time)
        # axs[0, 0].set_ylim(0.0, 15.1)
        # axs[0, 0].set_yticks(np.arange(0, 15.1, 5))
        #
        axs[1, 0].plot(tSim, m3_to_cm3*means_plot_gt[FEM_v_idx_gt, :], 'k', label='ground truth mean')
        axs[1, 0].plot(tSim, m3_to_cm3*means_plot_bf[FEM_v_idx, :], 'b', label='best fit mean')
        axs[1, 0].plot(tSim, m3_to_cm3*best_fit_obs[FEM_v_idx, :], 'r', label='best fit waveform')
        axs[1, 0].set_xlabel('Time [$s$]', fontsize=14)
        axs[1, 0].set_ylabel('Left Femoral Flow [$cm^3/s$]', fontsize=14)
        axs[1, 0].set_xlim(0.0, sim_time)
        # axs[1, 0].set_ylim(0.0, 15.1)
        # axs[1, 0].set_yticks(np.arange(0, 15.1, 5))
        #
        axs[0, 1].plot(tSim, m3_to_cm3*means_plot_gt[RSC_v_idx_gt, :], 'k', label='ground truth mean')
        axs[0, 1].plot(tSim, m3_to_cm3*means_plot_bf[RSC_v_idx, :], 'b', label='best fit mean')
        axs[0, 1].plot(tSim, m3_to_cm3*best_fit_obs[RSC_v_idx, :], 'r', label='best fit waveform')
        axs[0, 1].set_xlabel('Time [$s$]', fontsize=14)
        axs[0, 1].set_ylabel('RSC Flow [$cm^3/s$]', fontsize=14)
        axs[0, 1].set_xlim(0.0, sim_time)
        # axs[0, 1].set_ylim(0.0, 4.01)
        # axs[0, 1].set_yticks(np.arange(0, 4.1, 1))
        #
        # axs[1, 1].plot(tSim, m3_to_cm3*means_plot_gt[LMCA_v_idx_gt, :], 'k', label='ground truth mean')
        # axs[1, 1].plot(tSim, m3_to_cm3*means_plot_bf[LMCA_v_idx, :], 'b', label='best fit mean')
        # axs[1, 1].plot(tSim, m3_to_cm3*best_fit_obs[LMCA_v_idx, :], 'r', label='best fit waveform')
        # axs[1, 1].set_xlabel('Time [$s$]', fontsize=14)
        # axs[1, 1].set_ylabel('LMCA Flow [$cm^3/s$]', fontsize=14)
        # axs[1, 1].set_xlim(0.0, sim_time)
        # axs[1, 1].set_ylim(0.0, 0.401)
        # axs[1, 1].set_yticks(np.arange(0, 0.41, 0.1))

        fig.align_ylabels(axs[:, 0])
        fig.align_ylabels(axs[:, 1])
        axs[0, 0].legend(loc='upper right', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'reconstruct_{}_2.eps'.format(case_type)))
        plt.savefig(os.path.join(plot_dir, 'reconstruct_{}_2.pdf'.format(case_type)))




