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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import paperPlotSetup
from utilities import Normalise_class
paperPlotSetup.Setup_Plot(3)
from opencor_helper import SimulationHelper
from mpi4py import MPI
import re
from numpy import genfromtxt
# import tqdm # TODO this needs to be installed for corner plot but doesnt need an import here
mcmc_lib = 'emcee' # TODO make this a user variable
if mcmc_lib == 'emcee':
    import emcee
elif mcmc_lib == 'zeus':
    import zeus
else:
    print(f'unknown mcmc lib : {mcmc_lib}')
import corner
import csv
from datetime import date
from skopt import gp_minimize, Optimizer
from parsers.PrimitiveParsers import CSVFileParser
import pandas as pd
import json
import copy
import math
import scipy.linalg as la
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
# set resource limit to inf to stop seg fault problem #TODO remove this, I don't think it does much
import resource
curlimit = resource.getrlimit(resource.RLIMIT_STACK)
resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY,resource.RLIM_INFINITY))

# This mcmc_object will be an instance of the OpencorParamID class
# it needs to be global so that it can be used in calculate_lnlikelihood()
# without having its attributes pickled. opencor simulation objects
# can't be pickled because they are pyqt.
global mcmc_object
mcmc_object = None


class CVS0DParamID():
    """
    Class for doing parameter identification on a 0D cvs model
    """
    def __init__(self, model_path, param_id_model_type, param_id_method, mcmc_instead, file_name_prefix,
                 input_params_path=None,  sensitivity_params_path=None,
                 param_id_obs_path=None, sim_time=2.0, pre_time=20.0, maximumStep=0.0001, dt=0.01,
                 DEBUG=False):
        self.model_path = model_path
        self.param_id_method = param_id_method
        self.mcmc_instead = mcmc_instead
        self.param_id_model_type = param_id_model_type
        self.file_name_prefix = file_name_prefix

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.num_procs = self.comm.Get_size()

        self.dt = dt
        self.n_steps = int(sim_time/self.dt)

        param_id_obs_file_prefix = re.sub('\.json', '', os.path.split(param_id_obs_path)[1])
        case_type = f'{param_id_method}_{file_name_prefix}_{param_id_obs_file_prefix}'
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

        self.DEBUG = DEBUG
        if self.DEBUG:
            import resource

        # param names
        self.obs_names = None
        self.obs_types = None
        self.weight_const_vec = None
        self.weight_series_vec = None
        self.param_names = None
        self.sensitivity_param_names = None
        self.num_obs = None
        self.num_resistance_params = None
        self.gt_df = None
        if param_id_obs_path:
            self.__set_obs_names_and_df(param_id_obs_path)
        if input_params_path:
            self.__set_and_save_param_names(input_params_path=input_params_path)
        if sensitivity_params_path:
            self.__set_and_save_sensitivity_param_names(sensitivity_params_path=sensitivity_params_path)

        # ground truth values
        self.ground_truth_consts, self.ground_truth_series = self.__get_ground_truth_values()


        if self.mcmc_instead:
            # This mcmc_object will be an instance of the OpencorParamID class
            # it needs to be global so that it can be used in calculate_lnlikelihood()
            # without having its attributes pickled. opencor simulation objects
            # can't be pickled because they are pyqt.
            global mcmc_object 
            mcmc_object = OpencorMCMC(self.model_path,
                                           self.obs_names, self.obs_types,
                                           self.weight_const_vec, self.weight_series_vec,
                                           self.param_names, 
                                           self.ground_truth_consts, self.ground_truth_series,
                                           self.param_mins, self.param_maxs,
                                           sim_time=sim_time, pre_time=pre_time,
                                           dt=self.dt, maximumStep=maximumStep, DEBUG=self.DEBUG)
        else:
            if param_id_model_type == 'CVS0D':
                self.param_id = OpencorParamID(self.model_path, self.param_id_method,
                                               self.obs_names, self.obs_types,
                                               self.weight_const_vec, self.weight_series_vec,
                                               self.param_names, self.sensitivity_param_names,
                                               self.ground_truth_consts, self.ground_truth_series,
                                               self.param_mins, self.param_maxs,
                                               self.sensitivity_param_mins, self.sensitivity_param_maxs,
                                               sim_time=sim_time, pre_time=pre_time,
                                               dt=self.dt, maximumStep=maximumStep, DEBUG=self.DEBUG)
        if self.rank == 0:
            self.set_output_dir(self.output_dir)

        self.best_output_calculated = False
        self.sensitivity_calculated = False

    def run(self):
        self.param_id.run()

    def run_mcmc(self):
        mcmc_object.run()

    def run_single_sensitivity(self,sensitivity_output_path):
        self.param_id.run_single_sensitivity(sensitivity_output_path)

    def run_sensitivity(self,param_id_output_paths):
        self.param_id.run_sensitivity(param_id_output_paths)
        self.sensitivity_calculated = True

    def simulate_with_best_param_vals(self):
        self.param_id.simulate_with_best_param_vals()
        self.best_output_calculated = True

    def update_param_range(self, params_to_update_list_of_lists, mins, maxs):
        for params_to_update_list, min, max in zip(params_to_update_list_of_lists, mins, maxs):
            for JJ, param_name_list in enumerate(self.param_names):
                if param_name_list == params_to_update_list:
                    self.param_mins[JJ] = min
                    self.param_maxs[JJ] = max

    def update_sensitivity_param_range(self, params_to_update_list_of_lists, mins, maxs):
        for params_to_update_list, min, max in zip(params_to_update_list_of_lists, mins, maxs):
            for JJ, param_name_list in enumerate(self.sensitivity_param_names):
                if param_name_list == params_to_update_list:
                    self.sensitivity_param_mins[JJ] = min
                    self.sensitivity_param_maxs[JJ] = max

    def set_output_dir(self, path):
        self.output_dir = path
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        if self.mcmc_instead:
            mcmc_object.set_output_dir(self.output_dir)
        else:
            self.param_id.set_output_dir(self.output_dir)
    
    def set_best_param_vals(self, best_param_vals):
        if self.mcmc_instead:
            mcmc_object.set_best_param_vals = best_param_vals
        else:
            self.param_id.set_best_param_vals = best_param_vals

    def plot_outputs(self):
        if not self.best_output_calculated:
            print('simulate_with_best_param_vals must be done first '
                  'before plotting output values')
            print('running simulate_with_best_param_vals ')
            self.simulate_with_best_param_vals()

        print('plotting best observables')
        m3_to_cm3 = 1e6
        Pa_to_kPa = 1e-3
        no_conv = 1.0

        best_fit_obs = self.param_id.sim_helper.get_results(self.obs_names)
        best_fit_obs_consts, best_fit_obs_series = self.param_id.get_pred_obs_vec_and_array(best_fit_obs)

        # _________ Plot best comparison _____________
        subplot_width = 2
        fig, axs = plt.subplots(subplot_width, subplot_width)

        obs_names_unique = []
        for obs_name in self.obs_names:
            if obs_name not in obs_names_unique:
                obs_names_unique.append(obs_name)

        col_idx = 0
        row_idx = 0
        plot_idx = 0
        tSim = self.param_id.sim_helper.tSim - self.param_id.pre_time
        consts_plot_gt = np.tile(self.ground_truth_consts.reshape(-1, 1), (1, self.param_id.sim_helper.n_steps + 1))
        consts_plot_bf = np.tile(best_fit_obs_consts.reshape(-1, 1), (1, self.param_id.sim_helper.n_steps + 1))

        series_plot_gt = np.array(self.ground_truth_series)


        for unique_obs_count in range(len(obs_names_unique)):
            this_obs_waveform_plotted = False
            obs_name_for_plot = obs_names_unique[unique_obs_count].replace('/', '_')
            consts_idx = -1
            series_idx = -1
            percent_error_vec = np.zeros((self.num_obs,))
            for II in range(self.num_obs):
                # TODO the below counting is hacky, store the constant and series data in one list of arrays
                if self.gt_df.iloc[II]["data_type"] == "constant":
                    consts_idx += 1
                elif self.gt_df.iloc[II]["data_type"] == "series":
                    series_idx += 1
                # TODO generalise this for not just flows and pressures
                if self.obs_names[II] == obs_names_unique[unique_obs_count]:
                    for JJ in range(self.num_obs):
                        if self.obs_names[II] == self.gt_df.iloc[JJ]['variable'] and \
                                self.obs_types[II] == self.gt_df.iloc[JJ]['obs_type']:
                            break

                    if self.gt_df.iloc[JJ]["unit"] == 'm3_per_s':
                        conversion = m3_to_cm3
                        axs[row_idx, col_idx].set_ylabel(f'{obs_name_for_plot} [$cm^3/s$]', fontsize=14)
                    elif self.gt_df.iloc[JJ]["unit"] == 'm_per_s':
                        conversion = no_conv
                        axs[row_idx, col_idx].set_ylabel(f'{obs_name_for_plot} [$m/s$]', fontsize=14)
                    elif self.gt_df.iloc[JJ]["unit"] == 'm3':
                        conversion = m3_to_cm3
                        axs[row_idx, col_idx].set_ylabel(f'{obs_name_for_plot} [$cm^3$]', fontsize=14)
                    elif self.gt_df.iloc[JJ]["unit"] == 'J_per_m3':
                        conversion = Pa_to_kPa
                        axs[row_idx, col_idx].set_ylabel(f'{obs_name_for_plot} [$kPa$]', fontsize=14)
                    else:
                        print(f'variable with unit of {self.gt_df.iloc[JJ]["unit"]} is not implemented'
                              f'for plotting')
                        exit()
                    if not this_obs_waveform_plotted:
                        axs[row_idx, col_idx].plot(tSim, conversion*best_fit_obs[II, :], 'k', label='bf')
                        this_obs_waveform_plotted = True

                    if self.obs_types[II] == 'mean':
                        axs[row_idx, col_idx].plot(tSim, conversion*consts_plot_gt[consts_idx, :], 'b--', label='gt mean')
                        axs[row_idx, col_idx].plot(tSim, conversion*consts_plot_bf[consts_idx, :], 'b', label='bf mean')
                    elif self.obs_types[II] == 'max':
                        axs[row_idx, col_idx].plot(tSim, conversion*consts_plot_gt[consts_idx, :], 'r--', label='gt max')
                        axs[row_idx, col_idx].plot(tSim, conversion*consts_plot_bf[consts_idx, :], 'r', label='bf max')
                    elif self.obs_types[II] == 'min':
                        axs[row_idx, col_idx].plot(tSim, conversion*consts_plot_gt[consts_idx, :], 'g--', label='gt min')
                        axs[row_idx, col_idx].plot(tSim, conversion*consts_plot_bf[consts_idx, :], 'g', label='bf min')
                    elif self.obs_types[II] == 'series':
                        axs[row_idx, col_idx].plot(tSim, conversion*series_plot_gt[series_idx, :], 'k--', label='gt')

                #also calculate the RMS error for each observable
                if self.gt_df.iloc[II]["data_type"] == "constant":
                    percent_error_vec[II] = 100*np.abs((best_fit_obs_consts[II] - self.ground_truth_consts[II])/
                                                       self.ground_truth_consts[II])
                elif self.gt_df.iloc[II]["data_type"] == "series":
                    # rms_error_vec[II] = np.sqrt(consts_plot_gt[consts_idx, :] - consts_plot_bf[])
                    pass


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
                    plt.close()
                    plot_saved = True
                    col_idx = 0
                    row_idx = 0
                    plot_idx += 1
                    # create new plot
                    if unique_obs_count != len(obs_names_unique) - 1:
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
            plt.close()

        print('______observable errors______')
        for obs_idx in range(self.num_obs):
            if self.gt_df.iloc[obs_idx]["data_type"] == "constant":
                print(f'{self.obs_names[obs_idx]} {self.obs_types[obs_idx]} error:')
                print(f'{percent_error_vec[obs_idx]:.2f} %')
            if self.gt_df.iloc[II]["data_type"] == "series":
                # TODO
                pass

    def plot_mcmc(self):
    
        mcmc_chain_path = os.path.join(self.output_dir, 'mcmc_chain.npy')

        if not os.path.exists(mcmc_chain_path): 
            print('No mcmc results to plot')
            return
        
        print('plotting mcmc results')
        samples = np.load(os.path.join(self.output_dir, 'mcmc_chain.npy'))
        num_steps = samples.shape[0] 
        num_walkers = samples.shape[1] 
        num_params = samples.shape[2] # TODO check this is the same as objects num_params
        # discard first num_steps/10 samples
        # samples = samples[samples.shape[0]//10:, :, :]
        # thin = 10
        # samples = samples[::thin, :, :]
        # discarding samples isnt needed because we start an "optimal" point
        # TODO include a user defined burn in if we aren't starting from 
        # an optimal point.
        flat_samples = samples.reshape(-1, num_params)

        # TODO do this in plotting function instead
        fig, axes = plt.subplots(num_params, figsize=(10, 7), sharex=True)
        for i in range(num_params):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(self.param_names[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")
        # plt.savefig(os.path.join(self.output_dir, 'plots_param_id', 'mcmc_chain_plot.eps'))
        plt.savefig(os.path.join(self.output_dir, 'plots_param_id', 'mcmc_chain_plot.pdf'))
        plt.close()

        fig = corner.corner(flat_samples, bins=20, hist_bin_factor=2, smooth=0.5,
                            labels=self.param_names, truths=self.param_id.best_param_vals)
        # plt.savefig(os.path.join(self.output_dir, 'plots_param_id', 'mcmc_cornerplot.eps'))
        plt.savefig(os.path.join(self.output_dir, 'plots_param_id', 'mcmc_cornerplot.pdf'))
        # plt.savefig(os.path.join(self.plot_dir, 'mcmc_cornerplot.eps'))
        # plt.savefig(os.path.join(self.plot_dir, 'mcmc_cornerplot.pdf'))
        plt.close()

    def run_sensitivity(self, param_id_output_paths):
        print('running sensitivity analysis')
        if param_id_output_paths == None:
            sample_path_list = [self.output_dir]
        else:
            sample_path_list = []
            sample_paths = pd.read_csv(param_id_output_paths)
            for i in range(len(sample_paths)):
                sample_path_list.append(sample_paths.iat[i,0])
            if len(sample_path_list) < 1:
                sample_path_list = [self.output_dir]

        do_triples_and_quads = False

        for i in range(len(sample_path_list)):
            self.param_id.run_single_sensitivity(sample_path_list[i], do_triples_and_quads)
        # FA: Eventually we will automate the multiple runs of param_id and store the outputs in indexed
        # FA: directories that can all be accessed automatically by this function without defining input paths.
        m3_to_cm3 = 1e6
        Pa_to_kPa = 1e-3
        number_samples = len(sample_path_list)
        x_values = []
        for i in range(len(self.sensitivity_param_names)):
            x_values.append(self.sensitivity_param_names[i][0])


        for i in range(number_samples):
            sample_path = sample_path_list[i]
            normalised_jacobian = np.load(os.path.join(sample_path, 'normalised_jacobian_matrix.npy'))
            parameter_importance = np.load(os.path.join(sample_path, 'parameter_importance.npy'))
            collinearity_index = np.load(os.path.join(sample_path, 'collinearity_index.npy'))
            if i == 0:
                normalised_jacobian_average = np.zeros(normalised_jacobian.shape)
                parameter_importance_average = np.zeros(parameter_importance.shape)
                collinearity_index_average = np.zeros(collinearity_index.shape)
            normalised_jacobian_average = normalised_jacobian_average + normalised_jacobian/number_samples
            parameter_importance_average = parameter_importance_average + parameter_importance/number_samples
            collinearity_index_average = collinearity_index_average + collinearity_index/number_samples

        collinearity_max = collinearity_index_average.max()

        if do_triples_and_quads:
            #find maximum average value of collinearity triples
            for i in range(len(x_values)):
                for j in range(number_samples):
                    sample_path = sample_path_list[j]
                    collinearity_index_triple = np.load(os.path.join(sample_path, 'collinearity_triples' + str(i) + '.npy'))
                    if j==0:
                        collinearity_index_triple_average = np.zeros(collinearity_index_triple.shape)
                    collinearity_index_triple_average = collinearity_index_triple_average + collinearity_index_triple/number_samples
                if collinearity_index_triple_average.max() > collinearity_max:
                    collinearity_max = collinearity_index_triple_average.max()

            #find maximum value of collinearity quads
            for i in range(len(x_values)):
                for j in range(len(x_values)):
                    for k in range(number_samples):
                        sample_path = sample_path_list[k]
                        collinearity_index_quad = np.load(
                            os.path.join(sample_path, 'collinearity_quads' + str(i) + '_' + str(j) + '.npy'))
                        if k==0:
                            collinearity_index_quad_average = np.zeros(collinearity_index_quad.shape)
                        collinearity_index_quad_average = collinearity_index_quad_average + collinearity_index_quad / number_samples
                    if collinearity_index_quad_average.max() > collinearity_max:
                        collinearity_max = collinearity_index_quad_average.max()

        #find maximum average value and average values for collinearity index
        for i in range(number_samples):
            sample_path = sample_path_list[i]
            collinearity_index_pairs = np.load(
                os.path.join(sample_path, 'collinearity_pairs.npy'))
            if i==0:
                collinearity_index_pairs_average = np.zeros(collinearity_index_pairs.shape)
            collinearity_index_pairs_average = collinearity_index_pairs_average + collinearity_index_pairs/number_samples

        if collinearity_index_pairs_average.max() > collinearity_max:
            collinearity_max = collinearity_index_pairs_average.max()

        number_Params = len(normalised_jacobian_average)
        #plot jacobian
        subplot_height = 1
        subplot_width = 1
        plt.rc('xtick', labelsize=3)
        fig, axs = plt.subplots(subplot_height, subplot_width)

        x_labels = []
        subset = []
        x_index = 0
        for obs_idx in range(len(self.obs_names)):
            if self.obs_types[obs_idx] != "series":
                x_labels.append(self.obs_names[obs_idx] + " " + self.obs_types[obs_idx])
                subset.append(x_index)
                x_index = x_index + 1
        for param_idx in range(len(self.sensitivity_param_names)):
            y_values = []
            for obs_idx in range(len(x_labels)):
                y_values.append(abs((normalised_jacobian_average[param_idx][subset[obs_idx]])))
            axs.plot(x_labels, y_values, label=self.sensitivity_param_names[param_idx][0])
        axs.set_yscale('log')
        axs.legend(loc='lower left', fontsize=6)
        plt.savefig(os.path.join(self.plot_dir,
                                     f'reconstruct_{self.param_id_method}_'
                                     f'{self.file_name_prefix}_sensitivity_average.eps'))
        plt.savefig(os.path.join(self.plot_dir,
                                     f'reconstruct_{self.param_id_method}_'
                                     f'{self.file_name_prefix}_sensitivity_average.pdf'))
        plt.close()
        #plot parameter importance
        plt.rc('xtick', labelsize=6)
        plt.rc('ytick', labelsize=12)
        figB, axsB = plt.subplots(1, 1)


        axsB.bar(x_values, parameter_importance_average)
        axsB.set_ylabel("Parameter Importance", fontsize=12)
        plt.savefig(os.path.join(self.plot_dir,
                                     f'reconstruct_{self.param_id_method}_'
                                     f'{self.file_name_prefix}_parameter_importance_average.eps'))
        plt.savefig(os.path.join(self.plot_dir,
                                     f'reconstruct_{self.param_id_method}_'
                                     f'{self.file_name_prefix}_parameter_importance_average.pdf'))
        plt.close()
        #plot collinearity index average
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=4)
        figC, axsC = plt.subplots(1, 1)
        x_values_cumulative = []
        x_values_temp = x_values[0]
        for i in range(len(x_values)):
            x_values_cumulative.append(x_values_temp)
            if (i + 1) < len(x_values):
                x_values_temp = x_values[i + 1] + "\n" + x_values_temp
        axsC.barh(x_values_cumulative, collinearity_index_average)
        plt.savefig(os.path.join(self.plot_dir,
                                     f'reconstruct_{self.param_id_method}_'
                                     f'{self.file_name_prefix}_collinearity_index_average.eps'))
        plt.savefig(os.path.join(self.plot_dir,
                                     f'reconstruct_{self.param_id_method}_'
                                     f'{self.file_name_prefix}_collinearity_index_average.pdf'))
        plt.close()

        plt.rc('xtick', labelsize=4)
        plt.rc('ytick', labelsize=4)
        figD, axsD = plt.subplots(1, 1)

        X, Y = np.meshgrid(range(len(x_values)), range(len(x_values)))
        if do_triples_and_quads:
            collinearity_levels = np.linspace(0, collinearity_max, 20)
            co = axsD.contourf(X, Y, collinearity_index_pairs_average, levels=collinearity_levels)
        else:
            co = axsD.contourf(X, Y, collinearity_index_pairs_average)
        co = fig.colorbar(co, ax=axsD)
        axsD.set_xticks(range(len(x_values)))
        axsD.set_yticks(range(len(x_values)))
        axsD.set_xticklabels(x_values)
        axsD.set_yticklabels(x_values)

        plt.savefig(os.path.join(self.plot_dir,
                                     f'reconstruct_{self.param_id_method}_'
                                     f'{self.file_name_prefix}_collinearity_pairs_average.eps'))
        plt.savefig(os.path.join(self.plot_dir,
                                     f'reconstruct_{self.param_id_method}_'
                                     f'{self.file_name_prefix}_collinearity_pairs_average.pdf'))


        if do_triples_and_quads:
            #plot collinearity triples average
            for i in range(len(x_values)):
                for j in range(number_samples):
                    sample_path = sample_path_list[j]
                    collinearity_index_triple = np.load(os.path.join(sample_path, 'collinearity_triples' + str(i) + '.npy'))
                    if j == 0:
                        collinearity_index_triple_average = np.zeros(collinearity_index_triple.shape)
                    collinearity_index_triple_average = collinearity_index_triple_average + collinearity_index_triple/number_samples

                figE, axsE = plt.subplots(1, 1)
                co = axsE.contourf(X, Y, collinearity_index_triple_average, levels=collinearity_levels)
                co = fig.colorbar(co, ax=axsE)
                axsE.set_xticks(range(len(x_values)))
                axsE.set_yticks(range(len(x_values)))
                axsE.set_xticklabels(x_values)
                axsE.set_yticklabels(x_values)

                plt.savefig(os.path.join(self.plot_dir,
                                             f'reconstruct_{self.param_id_method}_'
                                             f'{self.file_name_prefix}_collinearity_triples_average' + str(i) + '.eps'))
                plt.savefig(os.path.join(self.plot_dir,
                                             f'reconstruct_{self.param_id_method}_'
                                             f'{self.file_name_prefix}_collinearity_triples_average' + str(i) + '.pdf'))
                plt.close()
            #plot collinearity quads average
            for i in range(len(x_values)):
                for j in range(len(x_values)):
                    for k in range(number_samples):
                        sample_path = sample_path_list[k]
                        collinearity_index_quad = np.load(
                            os.path.join(sample_path, 'collinearity_quads' + str(i) + '_' + str(j) + '.npy'))
                        if k==0:
                            collinearity_index_quad_average = np.zeros(collinearity_index_quad.shape)
                        collinearity_index_quad_average = collinearity_index_quad_average + collinearity_index_quad / number_samples
                    figE, axsE = plt.subplots(1, 1)
                    co = axsE.contourf(X, Y, collinearity_index_quad_average, levels=collinearity_levels)
                    co = fig.colorbar(co, ax=axsE)
                    axsE.set_xticks(range(len(x_values)))
                    axsE.set_yticks(range(len(x_values)))
                    axsE.set_xticklabels(x_values)
                    axsE.set_yticklabels(x_values)

                    plt.savefig(os.path.join(self.plot_dir,
                                                 f'reconstruct_{self.param_id_method}_'
                                                 f'{self.file_name_prefix}collinearity_quads_average' + str(i) + '_' + str(
                                                     j) + '.eps'))
                    plt.savefig(os.path.join(self.plot_dir,
                                                 f'reconstruct_{self.param_id_method}_'
                                                 f'{self.file_name_prefix}collinearity_quads_average' + str(i) + '_' + str(
                                                     j) + '.pdf'))
                    plt.close()

        print('sensitivity analysis complete')


    def save_prediction_data(self):
        pred_variables_path = os.path.join(resources_dir, f'{self.file_name_prefix}_prediction_variables.csv')
        if os.path.exists(pred_variables_path):
            print('Saving prediction data')
            pred_variables_names = genfromtxt(pred_variables_path,
                                      delimiter=',', dtype=None, encoding='UTF-8').flatten()
            pred_variables_names = np.array([pred_variables_names[II].strip()
                                             for II in range(pred_variables_names.shape[0])])

            tSim = self.param_id.sim_helper.tSim - self.param_id.pre_time
            pred_output = self.param_id.sim_helper.get_results(pred_variables_names)
            time_and_pred = np.concatenate((tSim.reshape(1, -1), pred_output))

            #save the prediction output
            np.save(os.path.join(self.output_dir, 'prediction_variable_data'), time_and_pred)
            print('Prediction data saved')

        else:
            print(f'prediction variables have not been defined, if you want to save predicition variables,',
                  f'create a file {pred_variables_path}, with the names of the desired prediction variables')

        return

    def set_genetic_algorithm_parameters(self, n_calls):
        self.param_id.set_genetic_algorithm_parameters(n_calls)

    def set_bayesian_parameters(self, n_calls, n_initial_points, acq_func, random_state, acq_func_kwargs={}):
        self.param_id.set_bayesian_parameters(n_calls, n_initial_points, acq_func, random_state,
                                              acq_func_kwargs=acq_func_kwargs)

    def close_simulation(self):
        self.param_id.close_simulation()

    def __set_obs_names_and_df(self, param_id_obs_path):
        with open(param_id_obs_path, encoding='utf-8-sig') as rf:
            json_obj = json.load(rf)
        self.gt_df = pd.DataFrame(json_obj)
        if self.gt_df.columns[0] == 'data_item':
            self.gt_df = self.gt_df['data_item']

        self.obs_names = [self.gt_df.iloc[II]["variable"] for II in range(self.gt_df.shape[0])]

        self.obs_types = [self.gt_df.iloc[II]["obs_type"] for II in range(self.gt_df.shape[0])]

        self.num_obs = len(self.obs_names)

        # how much to weight the different observable errors by
        self.weight_const_vec = np.array([self.gt_df.iloc[II]["weight"] for II in range(self.gt_df.shape[0])
                                          if self.gt_df.iloc[II]["data_type"] == "constant"])

        self.weight_series_vec = np.array([self.gt_df.iloc[II]["weight"] for II in range(self.gt_df.shape[0])
                                           if self.gt_df.iloc[II]["data_type"] == "series"])

        return

    def __set_and_save_param_names(self, input_params_path=None):

        # Each entry in param_names is a name or list of names that gets modified by one parameter
        if input_params_path:
            csv_parser = CSVFileParser()
            input_params = csv_parser.get_data_as_dataframe_multistrings(input_params_path)
            self.param_names = []
            param_names_for_gen = []
            param_state_names_for_gen = []
            param_const_names_for_gen = []
            for II in range(input_params.shape[0]):
                self.param_names.append([input_params["vessel_name"][II][JJ] + '/' +
                                               input_params["param_name"][II]for JJ in
                                               range(len(input_params["vessel_name"][II]))])

                if input_params["vessel_name"][II][0] == 'heart':
                    param_names_for_gen.append([input_params["param_name"][II]])

                    if input_params["param_type"][II] == 'state':
                        param_state_names_for_gen.append([input_params["param_name"][II]])

                    if input_params["param_type"][II] == 'const':
                        param_const_names_for_gen.append([input_params["param_name"][II]])

                else:
                    param_names_for_gen.append([input_params["param_name"][II] + '_' +
                                                re.sub('_T$', '', input_params["vessel_name"][II][JJ])
                                                for JJ in range(len(input_params["vessel_name"][II]))])

                    param_state_names_for_gen.append([input_params["param_name"][II] + '_' +
                                                      re.sub('_T$', '', input_params["vessel_name"][II][JJ])
                                                      for JJ in range(len(input_params["vessel_name"][II]))
                                                      if input_params["param_type"][II] == 'state'])

                    param_const_names_for_gen.append([input_params["param_name"][II] + '_' +
                                                      re.sub('_T$', '', input_params["vessel_name"][II][JJ])
                                                      for JJ in range(len(input_params["vessel_name"][II]))
                                                      if input_params["param_type"][II] == 'const'])


            # set param ranges from file
            self.param_mins = np.array([float(input_params["min"][JJ]) for JJ in range(input_params.shape[0])])
            self.param_maxs = np.array([float(input_params["max"][JJ]) for JJ in range(input_params.shape[0])])
        else:
            print(f'input_params_path cannot be None, exiting')

        if self.rank == 0:
            with open(os.path.join(self.output_dir, 'param_names.csv'), 'w') as f:
                wr = csv.writer(f)
                wr.writerows(self.param_names)
            with open(os.path.join(self.output_dir, 'param_names_for_gen.csv'), 'w') as f:
                wr = csv.writer(f)
                wr.writerows(param_names_for_gen)
            with open(os.path.join(self.output_dir, 'param_state_names_for_gen.csv'), 'w') as f:
                wr = csv.writer(f)
                wr.writerows(param_state_names_for_gen)
            with open(os.path.join(self.output_dir, 'param_const_names_for_gen.csv'), 'w') as f:
                wr = csv.writer(f)
                wr.writerows(param_const_names_for_gen)

        return

    def __set_and_save_sensitivity_param_names(self, sensitivity_params_path=None):

        # Each entry in sensitivity_param_names is a name or list of names that gets modified by one parameter
        if sensitivity_params_path:
            csv_parser = CSVFileParser()
            sensitivity_params = csv_parser.get_data_as_dataframe_multistrings(sensitivity_params_path)
            self.sensitivity_param_names = []
            for II in range(sensitivity_params.shape[0]):
                self.sensitivity_param_names.append([sensitivity_params["vessel_name"][II][JJ] + '/' +
                                               sensitivity_params["param_name"][II]for JJ in
                                               range(len(sensitivity_params["vessel_name"][II]))])

            # set param ranges from file
            self.sensitivity_param_mins = np.array([float(sensitivity_params["min"][JJ]) for JJ in range(sensitivity_params.shape[0])])
            self.sensitivity_param_maxs = np.array([float(sensitivity_params["max"][JJ]) for JJ in range(sensitivity_params.shape[0])])

        else:
            pass

        if self.rank == 0:
            with open(os.path.join(self.output_dir, 'sensitivity_param_names.csv'), 'w') as f:
                wr = csv.writer(f)
                wr.writerows(self.sensitivity_param_names)
            with open(os.path.join(self.output_dir, 'sensitivity_param_names.csv'), 'w') as f:
                wr = csv.writer(f)
                wr.writerows(self.sensitivity_param_names)

        return

    def __get_ground_truth_values(self):

        # _______ First we access data for mean values

        ground_truth_consts = np.array([self.gt_df.iloc[II]["value"] for II in range(self.gt_df.shape[0])
                                        if self.gt_df.iloc[II]["data_type"] == "constant"])

        ground_truth_series = np.array([self.gt_df.iloc[II]["series"] for II in range(self.gt_df.shape[0])
                                        if self.gt_df.iloc[II]["data_type"] == "series"])


        if self.rank == 0:
            np.save(os.path.join(self.output_dir, 'ground_truth_consts'), ground_truth_consts)
            # np.save(os.path.join(self.output_dir, 'ground_truth_series'), ground_truth_series)

        return ground_truth_consts, ground_truth_series
    
    def get_best_param_vals(self):
        return self.param_id.best_param_vals



class OpencorParamID():
    """
    Class for doing parameter identification on opencor models
    """
    def __init__(self, model_path, param_id_method,
                 obs_names, obs_types, weight_const_vec, weight_series_vec,
                 param_names, sensitivity_param_names,
                 ground_truth_consts, ground_truth_series,
                 param_mins, param_maxs, sensitivity_param_mins, sensitivity_param_maxs,
                 sim_time=2.0, pre_time=20.0, dt=0.01, maximumStep=0.0001,
                 DEBUG=False):

        self.model_path = model_path
        self.param_id_method = param_id_method
        self.output_dir = None

        self.obs_names = obs_names
        self.obs_types = obs_types
        self.weight_const_vec = weight_const_vec
        self.weight_series_vec = weight_series_vec
        self.param_names = param_names
        self.sensitivity_param_names = sensitivity_param_names
        self.num_obs = len(self.obs_names)
        self.num_params = len(self.param_names)
        self.sensitivity_num_params = len(self.sensitivity_param_names)
        self.ground_truth_consts = ground_truth_consts
        self.ground_truth_series = ground_truth_series
        self.param_mins = param_mins
        self.param_maxs = param_maxs
        self.sensitivity_param_mins = sensitivity_param_mins
        self.sensitivity_param_maxs = sensitivity_param_maxs
        self.param_norm_obj = None
        self.param_norm_obj = Normalise_class(self.param_mins, self.param_maxs)

        # set up opencor simulation
        self.dt = dt  # TODO this could be optimised
        self.maximumStep = maximumStep
        self.point_interval = self.dt
        self.sim_time = sim_time
        self.pre_time = pre_time
        self.n_steps = int(self.sim_time/self.dt)

        self.sim_helper = self.initialise_sim_helper()

        # initialise
        self.param_init = None
        self.best_param_vals = None
        self.best_cost = 999999

        # genetic algorithm constants TODO add more of the constants to this so they can be modified by the user
        self.n_calls = 10000
        self.acq_func = 'EI'  # the acquisition function
        self.n_initial_points = 5
        self.acq_func_kwargs = {}
        self.random_state = 1234 # random seed

        # mcmc
        self.sampler = None

        self.DEBUG = DEBUG

    def initialise_sim_helper(self):
        return SimulationHelper(self.model_path, self.dt, self.sim_time,
                                maximumNumberofSteps=100000000,
                                maximumStep=self.maximumStep, pre_time=self.pre_time)

    def run(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        num_procs = comm.Get_size()
        
        if num_procs == 1:
            print('WARNING Running in serial, are you sure you want to be a snail?')

        if rank == 0:
            # save date as identifier for the param_id
            np.save(os.path.join(self.output_dir, 'date'), date.today().strftime("%d_%m_%Y"))

        print('starting param id run for rank = {} process'.format(rank))

        # ________ Do parameter identification ________

        # Don't remove the get_init_param_vals, this also checks the parameters names are correct.
        self.param_init = self.sim_helper.get_init_param_vals(self.param_names)

        # C_T min and max was 1e-9 and 1e-5 before


        cost_convergence = 0.0001
        if self.param_id_method == 'bayesian':
            print('WARNING bayesian will be deprecated')
            if rank == 0:
                print('Running bayesian optimisation')
            param_ranges = [a for a in zip(self.param_mins, self.param_maxs)]
            updated_version = True # TODO remove this and remove the gp_minimize version
            if not updated_version:
                res = gp_minimize(self.get_cost_from_params,  # the function to minimize
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
                cost = np.zeros(num_procs)
                while call_num < self.n_calls:
                    if rank == 0:
                        if self.DEBUG:
                            zero_time = time.time()
                        if num_procs > 1:
                            # points = [opt.ask() for II in range(num_procs)]
                            # TODO figure out why the below call slows down so much as the number of calls increases
                            #  and whether it can give improvements
                            points = opt.ask(n_points=num_procs)
                            print(points)
                            points_np = np.array(points)
                        else:
                            points = opt.ask()
                        if self.DEBUG:
                            ask_time = time.time() - zero_time
                            print(f'Time to calculate new param values = {ask_time}')
                    else:
                        points_np = np.zeros((num_procs, self.num_params))

                    if num_procs > 1:
                        # broadcast points so every processor has all of the points. TODO This could be optimized for memory
                        comm.Bcast(points_np, root=0)
                        cost_proc = self.get_cost_from_params(points_np[rank, :])
                        # print(f'cost for rank = {rank} is {cost_proc}')

                        recv_buf_cost = np.zeros(num_procs)
                        send_buf_cost = cost_proc
                        # gather results from simulation
                        comm.Gatherv(send_buf_cost, [recv_buf_cost, 1,
                                                      None, MPI.DOUBLE], root=0)
                        cost_np = recv_buf_cost
                        cost = cost_np.tolist()
                    else:
                        cost = self.get_cost_from_params(points)


                    if rank == 0:
                        if self.DEBUG:
                            zero_time = time.time()
                        opt.tell(points, cost)
                        if self.DEBUG:
                            tell_time = time.time() - zero_time
                            print(f'Time to set the calculated cost and param values '
                                  f'and fit the gaussian = {tell_time}')
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

                    # Check resource usage
                    if self.DEBUG:
                        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        print(f'rank={rank} memory={mem}')

                    # TODO save results here every few iterations


            if rank == 0:
                print(res)
                self.best_cost = res.fun
                self.best_param_vals = res.x
                np.save(os.path.join(self.output_dir, 'best_cost'), self.best_cost)
                np.save(os.path.join(self.output_dir, 'best_param_vals'), self.best_param_vals)

        elif self.param_id_method == 'genetic_algorithm':
            num_elite = 12
            num_survivors = 48
            num_mutations_per_survivor = 12
            num_cross_breed = 120
            num_pop = num_survivors + num_survivors*num_mutations_per_survivor + \
                   num_cross_breed
            if self.n_calls < num_pop:
                print(f'Number of calls (n_calls={self.n_calls}) must be greater than the '
                      f'gen alg population (num_pop={num_pop}), exiting')
                exit()
            if num_procs > num_pop:
                print(f'Number of processors must be less than number of population, exiting')
                exit()
            self.max_generations = math.floor(self.n_calls/num_pop)
            if rank == 0:
                print(f'Running genetic algorithm with a population size of {num_pop},\n'
                      f'and a maximum number of generations of {self.max_generations}')
            simulated_bools = [False]*num_pop
            gen_count = 0

            if rank == 0:
                param_vals_norm = np.random.rand(self.num_params, num_pop)
                param_vals = self.param_norm_obj.unnormalise(param_vals_norm)
            else:
                param_vals = None

            cost = np.zeros(num_pop)
            cost[0] = 9999

            while cost[0] > cost_convergence and gen_count < self.max_generations:
                if gen_count > 30:
                   mutation_weight = 0.04
                elif gen_count > 60 :
                   mutation_weight = 0.02
                else:
                   mutation_weight = 0.08
                # TODO make these modifiable to the user
                # TODO make this more general for the automated approach
                # if gen_count > 100:
                #    mutation_weight = 0.01
                # elif gen_count > 160:
                #    mutation_weight = 0.005
                # elif gen_count > 220:
                #    mutation_weight = 0.002
                # elif gen_count > 300:
                #    mutation_weight = 0.001
                # else:
                #    mutation_weight = 0.02
                #
                # elif gen_count > 280:
                #     mutation_weight = 0.0003

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
                        self.sim_helper.set_param_vals(self.param_names, param_vals_proc[:, II])

                        success = self.sim_helper.run()
                        if success:
                            pred_obs = self.sim_helper.get_results(self.obs_names)
                            # calculate error between the observables of this set of parameters
                            # and the ground truth
                            cost_proc[II] = self.get_cost_from_obs(pred_obs)

                            # reset params
                            self.sim_helper.reset_and_clear()
                            bools_proc[II] = True # this point has now been simulated

                        else:
                            # simulation failed, choose a new random point
                            print('simulation failed with params...')
                            print(param_vals_proc[:, II])
                            print('... choosing a new random point')
                            param_vals_proc[:, II:II + 1] = self.param_norm_obj.unnormalise(np.random.rand(self.num_params, 1))
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
                    param_vals_norm = self.param_norm_obj.normalise(param_vals)
                    print('worst survivor params normed : {}'.format(param_vals_norm[:, num_survivors - 1]))
                    print('best params normed : {}'.format(param_vals_norm[:, 0]))
                    np.save(os.path.join(self.output_dir, 'best_cost'), cost[0])
                    np.save(os.path.join(self.output_dir, 'best_param_vals'), param_vals[:, 0])

                    # At this stage all of the population has been simulated
                    simulated_bools = [True]*num_pop
                    # keep the num_survivors best param_vals, replace these with mutations
                    param_idx = num_elite

                    # for idx in range(num_elite, num_survivors):
                    #     survive_prob = cost[num_elite:num_pop]**-1/sum(cost[num_elite:num_pop]**-1)
                    #     rand_survivor_idx = np.random.choice(np.arange(num_elite, num_pop), p=survive_prob)
                    #     param_vals_norm[:, param_idx] = param_vals_norm[:, rand_survivor_idx]
                    #
                    survive_prob = cost[num_elite:num_pop]**-1/sum(cost[num_elite:num_pop]**-1)
                    rand_survivor_idxs = np.random.choice(np.arange(num_elite, num_pop),
                                                         size=num_survivors-num_elite, p=survive_prob)
                    param_vals_norm[:, num_elite:num_survivors] = param_vals_norm[:, rand_survivor_idxs]

                    param_idx = num_survivors

                    for survivor_idx in range(num_survivors):
                        for JJ in range(num_mutations_per_survivor):
                            simulated_bools[param_idx] = False
                            fifty_fifty = np.random.rand()
                            if fifty_fifty < 0.5:
                              ## This accounts for smaller changes when the value is smaller
                              param_vals_norm[:, param_idx] = param_vals_norm[:, survivor_idx]* \
                                                              (1.0 + mutation_weight*np.random.randn(self.num_params))
                            else:
                              ## This doesn't account for smaller changes when the value is smaller
                              param_vals_norm[:, param_idx] = param_vals_norm[:, survivor_idx] + \
                                                              mutation_weight*np.random.randn(self.num_params)
                            param_idx += 1

                    # now do cross breeding
                    cross_breed_indices = np.random.randint(0, num_survivors, (num_cross_breed, 2))
                    for couple in cross_breed_indices:
                        if couple[0] == couple[1]:
                            couple[1] += 1  # this allows crossbreeding out of the survivors but that's ok
                        simulated_bools[param_idx] = False

                        fifty_fifty = np.random.rand()
                        if fifty_fifty < 0.5:
                          ## This accounts for smaller changes when the value is smaller
                          param_vals_norm[:, param_idx] = (param_vals_norm[:, couple[0]] +
                                                           param_vals_norm[:, couple[1]])/2* \
                                                          (1 + mutation_weight*np.random.randn(self.num_params))
                        else:
                          ## This doesn't account for smaller changes when the value is smaller,
                          ## which is needed to make sure values dont get stuck when they are small
                          param_vals_norm[:, param_idx] = (param_vals_norm[:, couple[0]] +
                                                           param_vals_norm[:, couple[1]])/2 + \
                                                          mutation_weight*np.random.randn(self.num_params)
                        param_idx += 1

                    param_vals = self.param_norm_obj.unnormalise(param_vals_norm)

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

    def run_single_sensitivity(self, sensitivity_output_path, do_triples_and_quads):
        if sensitivity_output_path == None:
            output_path = self.output_dir
        else:
            output_path = sensitivity_output_path

        self.param_init = self.sim_helper.get_init_param_vals(self.sensitivity_param_names)
        param_vec_init = np.array(self.param_init).flatten()
        self.best_param_vals = np.load(os.path.join(output_path, 'best_param_vals.npy'))

        # print(self.best_param_vals)

        #merge best found and initial values
        master_param_names = []
        master_param_values = []
        sensitivity_index = []

        for i in range(len(self.sensitivity_param_names)):
            master_param_names.append(self.sensitivity_param_names[i])
            master_param_values.append(param_vec_init[i])
            sensitivity_index.append(i)

        for i in range(len(self.param_names)):
            element_index = -1
            for j in range(len(master_param_names)):
                if master_param_names[j]==self.param_names[i]:
                    element_index = j
                    break
            if element_index >= 0:
                master_param_values[element_index] = self.best_param_vals[i]
            else:
                master_param_names.append(self.param_names[i])
                master_param_values.append(self.best_param_vals[i])

        num_sensitivity_params = len(self.sensitivity_param_names)
        gt_scalefactor = []
        x_index = 0
        objs_index = []

        for obs_idx in range(self.num_obs):
            #ignore series data for now
            if self.obs_types[obs_idx]!="series":
                #part of scale factor for normalising jacobain
                #gt_scalefactor.append(self.weight_const_vec[x_index]/self.ground_truth_consts[x_index])
                gt_scalefactor.append(1/self.ground_truth_consts[x_index])
                objs_index.append(x_index)
                x_index = x_index + 1

        jacobian_sensitivity = np.zeros((num_sensitivity_params,self.num_obs))

        for i in range(num_sensitivity_params):
            #central difference calculation of derivative
            param_vec_up = copy.deepcopy(master_param_values)
            param_vec_down = copy.deepcopy(master_param_values)
            # FA: It might be worth testing this out with a value smaller than 0.01 here
            param_vec_diff = (self.sensitivity_param_maxs[i] - self.sensitivity_param_mins[i])*0.001
            param_vec_range = self.sensitivity_param_maxs[i] - self.sensitivity_param_mins[i]
            param_vec_up[sensitivity_index[i]] = param_vec_up[sensitivity_index[i]] + param_vec_diff
            param_vec_down[sensitivity_index[i]] = param_vec_down[sensitivity_index[i]] - param_vec_diff
            up_pred_obs = self.sim_helper.modify_params_and_run_and_get_results(master_param_names,
                                                                                 param_vec_up, self.obs_names,
                                                                                 absolute=True)
            down_pred_obs = self.sim_helper.modify_params_and_run_and_get_results(master_param_names,
                                                                                 param_vec_down, self.obs_names,
                                                                                 absolute=True)

            up_pred_obs_consts_vec, up_pred_obs_series_array = self.get_pred_obs_vec_and_array(up_pred_obs)
            down_pred_obs_consts_vec, down_pred_obs_series_array = self.get_pred_obs_vec_and_array(down_pred_obs)
            for j in range(len(up_pred_obs_consts_vec)+len(up_pred_obs_series_array)):
                #normalise derivative
                if j < len(up_pred_obs_consts_vec):
                    dObs_param = (up_pred_obs_consts_vec[j]-down_pred_obs_consts_vec[j])/(param_vec_up[sensitivity_index[i]]-param_vec_down[sensitivity_index[i]])
                    dObs_param = dObs_param*param_vec_range*gt_scalefactor[objs_index[j]]
                else:
                    dObs_param = 0
                jacobian_sensitivity[i,j] = dObs_param

        np.save(os.path.join(output_path, 'normalised_jacobian_matrix.npy'), jacobian_sensitivity)

        #calculate parameter importance
        param_importance = np.zeros(num_sensitivity_params)
        for param_idx in range(num_sensitivity_params):
            sensitivity = 0
            for objs in range(self.num_obs):
                sensitivity = sensitivity + jacobian_sensitivity[param_idx][objs] * jacobian_sensitivity[param_idx][objs]
            sensitivity = math.sqrt(sensitivity / self.num_obs)
            param_importance[param_idx] = sensitivity
        np.save(os.path.join(output_path, 'parameter_importance.npy'), param_importance)

        #calculate S-norm
        S_norm = np.zeros((num_sensitivity_params,self.num_obs))
        for param_idx in range(num_sensitivity_params):
            for objs_idx in range(self.num_obs):
                S_norm[param_idx][objs_idx] = jacobian_sensitivity[param_idx][objs_idx]/\
                                             (param_importance[param_idx]*math.sqrt(self.num_obs))

        collinearity_eigvals = []
        for i in range(num_sensitivity_params):
            Sl = S_norm[:(i+1),:]
            Sll = Sl@Sl.T
            eigvals, eigvecs = la.eig(Sll)
            real_eigvals = eigvals.real
            collinearity_eigvals.append(min(real_eigvals))
        #calculate collinearity
        collinearity_index = np.zeros(len(collinearity_eigvals))
        for i in range(len(collinearity_eigvals)):
            collinearity_index[i] = 1/math.sqrt(collinearity_eigvals[i])

        np.save(os.path.join(output_path, 'collinearity_index.npy'), collinearity_index)


        collinearity_index_pairs = np.zeros((num_sensitivity_params,num_sensitivity_params))
        for i in range(num_sensitivity_params):
            for j in range(num_sensitivity_params):
                if i!=j:
                    Sl = S_norm[[i,j],:]
                    Sll = Sl@Sl.T
                    eigvals_pairs, eigvecs_pairs = la.eig(Sll)
                    real_eigvals_pairs = eigvals_pairs.real
                    collinearity_index_pairs[i][j] = 1/math.sqrt(min(real_eigvals_pairs))
                else:
                    collinearity_index_pairs[i][j] = 0

        np.save(os.path.join(output_path, 'collinearity_pairs.npy'), collinearity_index_pairs)

        if do_triples_and_quads:
            collinearity_index_triple = np.zeros((num_sensitivity_params, num_sensitivity_params))
            for i in range(num_sensitivity_params):
                for j in range(num_sensitivity_params):
                    for k in range(num_sensitivity_params):
                        if ((i!=j) and (i!=k) and (j!=k)):
                            Sl = S_norm[[i,j,k],:]
                            Sll = Sl@Sl.T
                            eigvals_pairs, eigvecs_pairs = la.eig(Sll)
                            real_eigvals_pairs = eigvals_pairs.real
                            collinearity_index_triple[j][k] = 1/math.sqrt(min(real_eigvals_pairs))
                        else:
                            collinearity_index_triple[j][k] = 0
                np.save(os.path.join(output_path, 'collinearity_triples'+str(i)+'.npy'), collinearity_index_triple)

            collinearity_index_quad = np.zeros((num_sensitivity_params, num_sensitivity_params))
            for i in range(num_sensitivity_params):
                for j in range(num_sensitivity_params):
                    for k in range(num_sensitivity_params):
                        for l in range(num_sensitivity_params):
                            if ((i!=j) and (i!=k) and (i!=l) and (j!=k) and (j!=l) and (k!=l)):
                                Sl = S_norm[[i,j,k,l],:]
                                Sll = Sl@Sl.T
                                eigvals_pairs, eigvecs_pairs = la.eig(Sll)
                                real_eigvals_pairs = eigvals_pairs.real
                                collinearity_index_quad[k][l] = 1/math.sqrt(min(real_eigvals_pairs))
                        else:
                            collinearity_index_quad[k][l] = 0
                    np.save(os.path.join(output_path, 'collinearity_quads'+str(i)+'_'+str(j)+'.npy'), collinearity_index_quad)

        return

    def get_cost_from_params(self, param_vals, reset=True, param_vals_are_normalised=False):

        # set params for this case
        if param_vals_are_normalised:
            param_vals = self.param_norm_obj.unnormalise(param_vals)

        self.sim_helper.set_param_vals(self.param_names, param_vals)
        
        success = self.sim_helper.run()
        if success:
            pred_obs = self.sim_helper.get_results(self.obs_names)

            cost = self.get_cost_from_obs(pred_obs)

            # reset params
            if reset:
                self.sim_helper.reset_and_clear()

        else:
            # simulation set cost to large,
            print('simulation failed with params...')
            print(param_vals)
            cost = np.inf

        return cost


    def get_cost_and_obs_from_params(self, param_vals, reset=True):
        # set params for this case
        self.sim_helper.set_param_vals(self.param_names, param_vals)

        success = self.sim_helper.run()
        if success:
            pred_obs = self.sim_helper.get_results(self.obs_names)

            cost = self.get_cost_from_obs(pred_obs)

            # reset params
            if reset:
                self.sim_helper.reset_and_clear()

        else:
            # simulation set cost to large,
            print('simulation failed with params...')
            print(param_vals)
            cost = 9999

        return cost, pred_obs

    def get_cost_from_obs(self, pred_obs):

        pred_obs_consts_vec, pred_obs_series_array = self.get_pred_obs_vec_and_array(pred_obs)
        # calculate error between the observables of this set of parameters
        # and the ground truth
        cost = self.cost_calc(pred_obs_consts_vec, pred_obs_series_array)

        return cost

    def cost_calc(self, prediction_consts, prediction_series=None):
        # cost = np.sum(np.power(self.weight_const_vec*(prediction_consts -
        #                        self.ground_truth_consts)/np.minimum(prediction_consts,
        #                                                             self.ground_truth_consts), 2))/(self.num_obs)
        cost = np.sum(np.power(self.weight_const_vec*(prediction_consts -
                               self.ground_truth_consts)/self.ground_truth_consts, 2))/(self.num_obs)
        # if prediction_series:
            # TODO Have not included cost from series error yet
            # cost +=
            # pass

        return cost

    def get_pred_obs_vec_and_array(self, pred_obs):

        pred_obs_consts_vec = np.zeros((len(self.ground_truth_consts), ))
        pred_obs_series_array = np.zeros((len(self.ground_truth_series), self.n_steps + 1))
        const_count = 0
        series_count = 0
        for JJ in range(len(pred_obs)):
            if self.obs_types[JJ] == 'mean':
                pred_obs_consts_vec[const_count] = np.mean(pred_obs[JJ, :])
                const_count += 1
            elif self.obs_types[JJ] == 'max':
                pred_obs_consts_vec[const_count] = np.max(pred_obs[JJ, :])
                const_count += 1
            elif self.obs_types[JJ] == 'min':
                pred_obs_consts_vec[const_count] = np.min(pred_obs[JJ, :])
                const_count += 1
            elif self.obs_types[JJ] == 'series':
                pred_obs_series_array[series_count, :] = pred_obs[JJ, :]
                series_count += 1
                pass
        return pred_obs_consts_vec, pred_obs_series_array

    def simulate_with_best_param_vals(self):
        if MPI.COMM_WORLD.Get_rank() != 0:
            print('simulate once should only be done on one rank')
            exit()
        if self.best_param_vals is None:
            self.best_cost = np.load(os.path.join(self.output_dir, 'best_cost.npy'))
            self.best_param_vals = np.load(os.path.join(self.output_dir, 'best_param_vals.npy'))
        else:
            # The sim object has already been opened so the best cost doesn't need to be opened
            pass

        # ___________ Run model with new parameters ________________

        self.sim_helper.update_times(self.dt, 0.0, self.sim_time, self.pre_time)

        # run simulation and check cost
        cost_check, pred_obs = self.get_cost_and_obs_from_params(self.best_param_vals, reset=False)
        pred_obs_constants_vec, pred_obs_series_array = self.get_pred_obs_vec_and_array(pred_obs)

        print(f'cost should be {self.best_cost}')
        print('cost check after single simulation is {}'.format(cost_check))

        print(f'final obs values :')
        print(pred_obs_constants_vec)
        # TODO print all const outputs with their variable name

    def simulate_once(self):
        if MPI.COMM_WORLD.Get_rank() != 0:
            print('simulate once should only be done on one rank')
            exit()
        else:
            # The sim object has already been opened so the best cost doesn't need to be opened
            pass

        # ___________ Run model with new parameters ________________

        self.sim_helper.update_times(self.dt, 0.0, self.sim_time, self.pre_time)

        # run simulation and check cost
        cost_check, pred_obs = self.get_cost_and_obs_from_params(self.best_param_vals, reset=False)
        pred_obs_constants_vec, pred_obs_series_array = self.get_pred_obs_vec_and_array(pred_obs)

        print(f'cost should be {self.best_cost}')
        print('cost check after single simulation is {}'.format(cost_check))

        print(f'final obs values :')
        print(pred_obs_constants_vec)

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

def calculate_lnlikelihood(param_vals):
    """
    This function is a wrapper around the mcmc_object method
    to calculate the lnlikelihood from model simulation.
    It allows the emcee algorithm to only pickle the param_vals
    and not all the attributes of the class instance.
    """
    return mcmc_object.get_lnlikelihood_from_params(param_vals)

class OpencorMCMC():
    """
    Class for doing mcmc on opencor models
    """

    def __init__(self, model_path,
                 obs_names, obs_types, weight_const_vec, weight_series_vec,
                 param_names,
                 ground_truth_consts, ground_truth_series,
                 param_mins, param_maxs,
                 sim_time=2.0, pre_time=20.0, dt=0.01, maximumStep=0.0001,
                 DEBUG=False):

        self.model_path = model_path
        self.output_dir = None

        self.obs_names = obs_names
        self.obs_types = obs_types
        self.weight_const_vec = weight_const_vec
        self.weight_series_vec = weight_series_vec
        self.param_names = param_names
        self.num_obs = len(self.obs_names)
        self.num_params = len(self.param_names)
        self.ground_truth_consts = ground_truth_consts
        self.ground_truth_series = ground_truth_series
        self.param_mins = param_mins
        self.param_maxs = param_maxs
        self.param_norm_obj = None
        self.param_norm_obj = Normalise_class(self.param_mins, self.param_maxs)

        # TODO load this from file. Make the user define the priors
        self.param_prior_dists = None# ['uniform', 'uniform', 'exponential']

        # set up opencor simulation
        self.dt = dt  # TODO this could be optimised
        self.maximumStep = maximumStep
        self.point_interval = self.dt
        self.sim_time = sim_time
        self.pre_time = pre_time
        self.n_steps = int(self.sim_time/self.dt)
        self.sim_helper = self.initialise_sim_helper()

        # initialise
        self.param_init = None
        self.best_param_vals = None
        self.best_cost = 999999

        # mcmc
        self.sampler = None
        self.num_steps = 400

        self.DEBUG = DEBUG

    def initialise_sim_helper(self):
        return SimulationHelper(self.model_path, self.dt, self.sim_time,
                                maximumNumberofSteps=100000000,
                                maximumStep=self.maximumStep, pre_time=self.pre_time)

    def set_best_param_vals(self, best_param_vals):
        self.best_param_vals = best_param_vals

    def run(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        num_procs = comm.Get_size()
        if rank == 0:
            print('Running mcmc')

        if num_procs > 1:
            # from pathos import multiprocessing
            # from pathos.multiprocessing import ProcessPool
            from schwimmbad import MPIPool

            num_walkers = 128 # TODO make this user definable or change back to max(4*self.num_params, num_procs)

            if rank == 0:
                if self.best_param_vals:
                    best_param_vals_norm = self.param_norm_obj.normalise(self.best_param_vals)
                    # create initial params in gaussian ball around best_param_vals estimate
                    init_param_vals_norm = (np.ones((num_walkers, self.num_params))*best_param_vals_norm).T + \
                                       0.01*np.random.randn(self.num_params, num_walkers)
                    init_param_vals = self.param_norm_obj.unnormalise(init_param_vals_norm)
                else:
                    init_param_vals_norm = np.random.rand(self.num_params, num_walkers)
                    init_param_vals = self.param_norm_obj.unnormalise(init_param_vals_norm)

            try:
                pool = MPIPool() # workers dont get past this line in this try, they wait for work to do
                if mcmc_lib == 'emcee':
                    self.sampler = emcee.EnsembleSampler(num_walkers, self.num_params, calculate_lnlikelihood,
                                                pool=pool)
                elif mcmc_lib == 'zeus':
                    self.sampler = zeus.EnsembleSampler(num_walkers, self.num_params, calculate_lnlikelihood,
                                                         pool=pool)

                start_time = time.time()
                self.sampler.run_mcmc(init_param_vals.T, self.num_steps, progress=True)
                print(f'mcmc time = {time.time() - start_time}')
            except:
                if rank == 0:
                    sys.exit()
                else:
                    # workers pass to here
                    pass

        else:
            num_walkers = 2*self.num_params

            if self.best_param_vals:
                best_param_vals_norm = self.param_norm_obj.normalise(self.best_param_vals)
                init_param_vals_norm = (np.ones((num_walkers, self.num_params))*best_param_vals_norm).T + \
                                   0.01*np.random.randn(self.num_params, num_walkers)
                init_param_vals = self.param_norm_obj.unnormalise(init_param_vals_norm)
            else:
                init_param_vals_norm = np.random.rand(self.num_params, num_walkers)
                init_param_vals = self.param_norm_obj.unnormalise(init_param_vals_norm)

            if mcmc_lib == 'emcee':
                self.sampler = emcee.EnsembleSampler(num_walkers, self.num_params, calculate_lnlikelihood)
            elif mcmc_lib == 'zeus':
                self.sampler = zeus.EnsembleSampler(num_walkers, self.num_params, calculate_lnlikelihood)

            start_time = time.time()
            self.sampler.run_mcmc(init_param_vals.T, self.num_steps) # , progress=True)
            print(f'mcmc time = {time.time()-start_time}')

        if rank == 0:
            # TODO save chains
            if mcmc_lib == 'emcee':
                print(f'acceptance fraction was {self.sampler.acceptance_fraction}')
                print(f'autocorrelation time was {self.sampler.get_autocorr_time}')
            samples = self.sampler.get_chain()
            mcmc_chain_path = os.path.join(self.output_dir, 'mcmc_chain.npy')
            np.save(mcmc_chain_path, samples)
            print('mcmc complete')
            print(f'mcmc chain saved in {mcmc_chain_path}')

    def get_lnprior_from_params(self, param_vals):
        lnprior = 0
        for idx, param_val in enumerate(param_vals):
            # TODO input param_prior_dists
            if self.param_prior_dists:
                prior_dist = self.param_prior_dists[idx]
            else:
                prior_dist = None

            if not prior_dist or prior_dist == 'uniform':
                if param_val < self.param_mins[idx] or param_val > self.param_maxs[idx]:
                    return -np.inf
                else:
                    #prior += 0
                    pass
            
            elif prior_dist == 'exponential':
                lamb = 1.0 # TODO make this user modifiable
                if param_val < self.param_mins[idx]:
                    return -np.inf
                else:
                    # the normalisation isnt needed here but might be nice to
                    # make sure prior for each param is between 0 and 1
                    lnprior += -lamb*param_val/self.param_maxs[idx]
        return lnprior

    def get_lnlikelihood_from_params(self, param_vals, reset=True, param_vals_are_normalised=False):
        # set params for this case
        if param_vals_are_normalised:
            param_vals = self.param_norm_obj.unnormalise(param_vals)

        lnprior = self.get_lnprior_from_params(param_vals)

        if not np.isfinite(lnprior):
            return -np.inf

        self.sim_helper.set_param_vals(self.param_names, param_vals)
        success = self.sim_helper.run()
        if success:
            pred_obs = self.sim_helper.get_results(self.obs_names)

            lnlikelihood = self.get_lnlikelihood_from_obs(pred_obs)

            # reset params
            if reset:
                self.sim_helper.reset_and_clear()

        else:
            # simulation set cost to large,
            print('simulation failed with params...')
            print(param_vals)
            return -np.inf

        return lnprior + lnlikelihood


    def get_lnlikelihood_from_obs(self, pred_obs):

        pred_obs_consts_vec, pred_obs_series_array = self.get_pred_obs_vec_and_array(pred_obs)
        # calculate error between the observables of this set of parameters
        # and the ground truth
        cost = self.lnlikelihood_calc(pred_obs_consts_vec, pred_obs_series_array)

        return cost


    def lnlikelihood_calc(self, prediction_consts, prediction_series=None):
        # cost = np.sum(np.power(self.weight_const_vec*(prediction_consts -
        #                        self.ground_truth_consts)/np.minimum(prediction_consts,
        #                                                             self.ground_truth_consts), 2))/(self.num_obs)
        lnlikelihood = -0.5*np.sum(np.power(self.weight_const_vec*(prediction_consts -
                               self.ground_truth_consts)/self.ground_truth_consts, 2))
        # if prediction_series:
            # TODO Have not included cost from series error yet
            # cost +=
            # pass

        return lnlikelihood
    
    def get_pred_obs_vec_and_array(self, pred_obs):

        pred_obs_consts_vec = np.zeros((len(self.ground_truth_consts), ))
        pred_obs_series_array = np.zeros((len(self.ground_truth_series), self.n_steps + 1))
        const_count = 0
        series_count = 0
        for JJ in range(len(pred_obs)):
            if self.obs_types[JJ] == 'mean':
                pred_obs_consts_vec[const_count] = np.mean(pred_obs[JJ, :])
                const_count += 1
            elif self.obs_types[JJ] == 'max':
                pred_obs_consts_vec[const_count] = np.max(pred_obs[JJ, :])
                const_count += 1
            elif self.obs_types[JJ] == 'min':
                pred_obs_consts_vec[const_count] = np.min(pred_obs[JJ, :])
                const_count += 1
            elif self.obs_types[JJ] == 'series':
                pred_obs_series_array[series_count, :] = pred_obs[JJ, :]
                series_count += 1
                pass
        return pred_obs_consts_vec, pred_obs_series_array
    
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

