'''
@author: Finbar J. Argus
'''

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
import matplotlib.ticker as tick
import paperPlotSetup
import stat_distributions
import diagnostics
import utilities
import traceback
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
import math
import scipy.linalg as la
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
# set resource limit to inf to stop seg fault problem #TODO remove this, I don't think it does much
# import resource
# curlimit = resource.getrlimit(resource.RLIMIT_STACK)
# resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY,resource.RLIM_INFINITY))

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
                 input_params_path=None,
                 param_id_obs_path=None, sim_time=2.0, pre_time=20.0,
                 pre_heart_periods=None, sim_heart_periods=None,
                 maximum_step=0.0001, dt=0.01, mcmc_options=None, ga_options=None, DEBUG=False):
        self.model_path = model_path
        self.param_id_method = param_id_method
        self.mcmc_instead = mcmc_instead
        self.param_id_model_type = param_id_model_type
        self.file_name_prefix = file_name_prefix

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.num_procs = self.comm.Get_size()

        self.dt = dt

        self.param_id_obs_file_prefix = re.sub('\.json', '', os.path.split(param_id_obs_path)[1])
        case_type = f'{param_id_method}_{file_name_prefix}_{self.param_id_obs_file_prefix}'
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
        # if self.DEBUG:
        #     import resource

        # TODO I should have a separate class for parsing the observable info from param_id_obs_path
        #  and param info from input_params_path
        # obs info
        self.obs_names = None
        self.obs_types = None
        self.obs_operations = None
        self.obs_operands = None
        self.weight_const_vec = None
        self.weight_series_vec = None
        self.weight_amp_vec = None
        self.weight_phase_vec = None
        self.std_const_vec = None
        self.std_series_vec = None
        self.std_amp_vec = None
        # param names
        self.param_names = None
        self.param_mins = None
        self.param_maxs = None
        self.param_prior_types = None
        self.param_names_for_plotting = None
        self.num_obs = None
        self.gt_df = None
        self.input_params_path = input_params_path
        if param_id_obs_path:
            self.__set_obs_names_and_df(param_id_obs_path)
        if self.input_params_path:
            self.__set_and_save_param_names()

        # ground truth values
        self.ground_truth_const, self.ground_truth_series, self.ground_truth_amp, self.ground_truth_phase = \
            self.__get_ground_truth_values()

        # get prediction variables
        self.pred_var_names = None
        self.__set_prediction_var()

        if self.mcmc_instead:
            # This mcmc_object will be an instance of the OpencorParamID class
            # it needs to be global so that it can be used in calculate_lnlikelihood()
            # without having its attributes pickled. opencor simulation objects
            # can't be pickled because they are pyqt.
            global mcmc_object 
            mcmc_object = OpencorMCMC(self.model_path,
                                           self.obs_names, self.obs_types, self.obs_freqs,
                                           self.obs_operations, self.obs_operands,
                                           self.weight_const_vec, self.weight_series_vec, 
                                           self.weight_amp_vec, self.weight_phase_vec,
                                           self.std_const_vec, self.std_series_vec, self.std_amp_vec,
                                           self.param_names,
                                           self.ground_truth_const, self.ground_truth_series,
                                           self.ground_truth_amp, self.ground_truth_phase,
                                           self.param_mins, self.param_maxs, self.param_prior_types,
                                           sim_time=sim_time, pre_time=pre_time,
                                           pre_heart_periods=pre_heart_periods, sim_heart_periods=sim_heart_periods,
                                           dt=self.dt, maximum_step=maximum_step, mcmc_options=mcmc_options,
                                           DEBUG=self.DEBUG)
            self.n_steps = mcmc_object.n_steps
        else:
            if param_id_model_type == 'cellml_only':
                self.param_id = OpencorParamID(self.model_path, self.param_id_method,
                                               self.obs_names, self.obs_types, self.obs_freqs,
                                               self.obs_operations, self.obs_operands,
                                               self.weight_const_vec, self.weight_series_vec,
                                               self.weight_amp_vec, self.weight_phase_vec,
                                               self.std_const_vec, self.std_series_vec, self.std_amp_vec,
                                               self.param_names, self.pred_var_names,
                                               self.ground_truth_const, self.ground_truth_series,
                                               self.ground_truth_amp, self.ground_truth_phase,
                                               self.param_mins, self.param_maxs,
                                               sim_time=sim_time, pre_time=pre_time,
                                               pre_heart_periods=pre_heart_periods, sim_heart_periods=sim_heart_periods,
                                               dt=self.dt, maximum_step=maximum_step, ga_options=ga_options,
                                               DEBUG=self.DEBUG)
            self.n_steps = self.param_id.n_steps
        if self.rank == 0:
            self.set_output_dir(self.output_dir)
        
        self.best_output_calculated = False
        self.sensitivity_calculated = False

    def temp_test(self):
        self.param_id.temp_test()
    def temp_test2(self):
        self.param_id.temp_test2()

    def run(self):
        self.param_id.run()

    def run_mcmc(self):
        mcmc_object.run()

    def run_single_sensitivity(self,sensitivity_output_path):
        self.param_id.run_single_sensitivity(sensitivity_output_path)

    def simulate_with_best_param_vals(self):
        self.param_id.simulate_with_best_param_vals()
        self.best_output_calculated = True

    def update_param_range(self, params_to_update_list_of_lists, mins, maxs):
        for params_to_update_list, min, max in zip(params_to_update_list_of_lists, mins, maxs):
            for JJ, param_name_list in enumerate(self.param_names):
                if param_name_list == params_to_update_list:
                    self.param_mins[JJ] = min
                    self.param_maxs[JJ] = max

    def set_output_dir(self, path):
        if self.rank != 0:
            return
        self.output_dir = path
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        if self.mcmc_instead:
            mcmc_object.set_output_dir(self.output_dir)
        else:
            self.param_id.set_output_dir(self.output_dir)
    
    def set_param_names(self, param_names):
        if self.mcmc_instead:
            mcmc_object.set_param_names(param_names)
        else:
            self.param_id.set_param_names(param_names)

        # TODO have to save param names as in __set_and_save_param_names!!

    def set_best_param_vals(self, best_param_vals):
        if self.mcmc_instead:
            mcmc_object.set_best_param_vals(best_param_vals)
        else:
            self.param_id.set_best_param_vals(best_param_vals)

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
        if len(self.ground_truth_phase) == 0:
            phase = False
        elif self.ground_truth_phase.all() == None:
            phase = False
        else: 
            phase = True

        if np.any(np.array(self.obs_types) == 'frequency') and np.any(np.array(self.obs_operations) != None):
            best_fit_obs, best_fit_temp_obs = self.param_id.sim_helper.get_results(self.obs_names,
                                                                                   output_temp_results=True)
        else:
            best_fit_obs = self.param_id.sim_helper.get_results(self.obs_names)
            best_fit_temp_obs = None

        obs_dict = self.param_id.get_obs_vec_and_array(best_fit_obs, temp_obs=best_fit_temp_obs)
        best_fit_obs_const = obs_dict['const']
        best_fit_obs_series = obs_dict['series']
        best_fit_obs_amp = obs_dict['amp']
        best_fit_obs_phase = obs_dict['phase']

        # _________ Plot best comparison _____________
        subplot_width = 1
        fig, axs = plt.subplots(subplot_width, subplot_width, squeeze=False)
        fig_phase, axs_phase = plt.subplots(subplot_width, subplot_width, squeeze=False)

        obs_names_unique = []
        for obs_name in self.obs_names:
            if obs_name not in obs_names_unique:
                obs_names_unique.append(obs_name)

        col_idx = 0
        row_idx = 0
        plot_idx = 0
        tSim = self.param_id.sim_helper.tSim - self.param_id.pre_time
        const_plot_gt = np.tile(self.ground_truth_const.reshape(-1, 1), (1, self.param_id.sim_helper.n_steps + 1))
        const_plot_bf = np.tile(best_fit_obs_const.reshape(-1, 1), (1, self.param_id.sim_helper.n_steps + 1))

        if len(self.ground_truth_series) > 0:
            min_len_series = min(self.ground_truth_series.shape[1], best_fit_obs_series.shape[1])


        for unique_obs_count in range(len(obs_names_unique)):
            this_obs_waveform_plotted = False
            const_idx = -1
            series_idx = -1
            freq_idx = -1
            percent_error_vec = np.zeros((self.num_obs,))
            phase_error_vec = np.zeros((self.num_obs,))
            std_error_vec = np.zeros((self.num_obs,))
            for II in range(self.num_obs):
                # TODO the below counting is hacky, store the constant and series data in one list of arrays
                if self.gt_df.iloc[II]["data_type"] == "constant":
                    const_idx += 1
                elif self.gt_df.iloc[II]["data_type"] == "series":
                    series_idx += 1
                elif self.gt_df.iloc[II]["data_type"] == "frequency":
                    freq_idx += 1
                # TODO generalise this for not just flows and pressures
                if self.obs_names[II] == obs_names_unique[unique_obs_count]:
                    for JJ in range(self.num_obs):
                        if self.obs_names[II] == self.gt_df.iloc[JJ]['variable'] and \
                                self.obs_types[II] == self.gt_df.iloc[JJ]['obs_type']:
                            break

                    if "name_for_plotting" in self.gt_df.iloc[II].keys():
                        obs_name_for_plot = self.gt_df.iloc[II]["name_for_plotting"]
                    else:
                        obs_name_for_plot = self.obs_names[II]

                    if self.gt_df.iloc[JJ]["unit"] == 'm3_per_s':
                        conversion = m3_to_cm3
                        unit_label = '[cm^3/s]'
                    elif self.gt_df.iloc[JJ]["unit"] == 'm_per_s':
                        conversion = no_conv
                        unit_label = '[m/s]'
                    elif self.gt_df.iloc[JJ]["unit"] == 'm3':
                        conversion = m3_to_cm3
                        unit_label = '[cm^3]'
                    elif self.gt_df.iloc[JJ]["unit"] == 'J_per_m3':
                        conversion = Pa_to_kPa
                        unit_label = '[kPa]'
                    else:
                        conversion = 1.0
                        unit_label = f'[{self.gt_df.iloc[JJ]["unit"]}]'

                    axs[row_idx, col_idx].set_ylabel(f'${obs_name_for_plot}$ ${unit_label}$', fontsize=18)
                    if phase:
                        axs_phase[row_idx, col_idx].set_ylabel(f'${obs_name_for_plot}$ phase', fontsize=18)
                    
                    if not this_obs_waveform_plotted:
                        if not self.obs_types[II] == 'frequency':
                            axs[row_idx, col_idx].plot(tSim, conversion*best_fit_obs[II, :], 'k', label='output')
                        else:
                            axs[row_idx, col_idx].plot(self.obs_freqs[II], conversion * best_fit_obs_amp[freq_idx],
                                                       'kv', label='model output')
                            if phase:
                                axs_phase[row_idx, col_idx].plot(self.obs_freqs[II], conversion * best_fit_obs_phase[freq_idx],
                                                        'kv', label='model output')
                        this_obs_waveform_plotted = True

                    if self.obs_types[II] == 'mean':
                        axs[row_idx, col_idx].plot(tSim, conversion*const_plot_gt[const_idx, :],
                                                   'b--', label='mean measurement')
                        axs[row_idx, col_idx].plot(tSim, conversion*const_plot_bf[const_idx, :],
                                                   'b', label='mean output')
                    elif self.obs_types[II] == 'max':
                        axs[row_idx, col_idx].plot(tSim, conversion*const_plot_gt[const_idx, :],
                                                   'r--', label='max measurement')
                        axs[row_idx, col_idx].plot(tSim, conversion*const_plot_bf[const_idx, :],
                                                   'r', label='max output')
                    elif self.obs_types[II] == 'min':
                        axs[row_idx, col_idx].plot(tSim, conversion*const_plot_gt[const_idx, :],
                                                   'g--', label='min measurement')
                        axs[row_idx, col_idx].plot(tSim, conversion*const_plot_bf[const_idx, :], 'g', label='min output')
                    elif self.obs_types[II] == 'series':
                        axs[row_idx, col_idx].plot(tSim[:min_len_series],
                                                   conversion*self.ground_truth_series[series_idx, :min_len_series],
                                                   'k--', label='measurement')
                    elif self.obs_types[II] == 'frequency':
                        axs[row_idx, col_idx].plot(self.obs_freqs[II],
                                                   conversion*self.ground_truth_amp[freq_idx],
                                                   'kx', label='measurement')
                        if phase:
                            axs_phase[row_idx, col_idx].plot(self.obs_freqs[II],
                                                    conversion*self.ground_truth_phase[freq_idx],
                                                    'kx', label='measurement')
                    if self.gt_df.iloc[II]["data_type"] == "frequency":
                        axs[row_idx, col_idx].set_xlim(0.0, self.obs_freqs[II][-1])
                        axs[row_idx, col_idx].set_xlabel('frequency [$Hz$]', fontsize=18)
                    else:
                        axs[row_idx, col_idx].set_xlim(0.0, self.param_id.sim_time)
                        axs[row_idx, col_idx].set_xlabel('Time [$s$]', fontsize=18)

                #also calculate the RMS error for each observable
                if self.gt_df.iloc[II]["data_type"] == "constant":
                    percent_error_vec[II] = 100*(best_fit_obs_const[const_idx] - self.ground_truth_const[const_idx])/ \
                                                       self.ground_truth_const[const_idx]
                    std_error_vec[II] = (best_fit_obs_const[const_idx] - self.ground_truth_const[const_idx])/ \
                                                       self.std_const_vec[const_idx]
                elif self.gt_df.iloc[II]["data_type"] == "series":
                    percent_error_vec[II] = 100*np.sum(np.abs((self.ground_truth_series[series_idx, :min_len_series] -
                                                               best_fit_obs_series[series_idx, :min_len_series]) /
                                                              (np.mean(self.ground_truth_series[series_idx, :min_len_series]))))/min_len_series
                    std_error_vec[II] = np.sum(np.abs((self.ground_truth_series[series_idx, :min_len_series] -
                                                       best_fit_obs_series[series_idx, :min_len_series]) /
                                                      (self.std_series_vec[series_idx]))/min_len_series)
                elif self.gt_df.iloc[II]["data_type"] == "frequency":
                    std_error_vec[II] = np.sum(np.abs((best_fit_obs_amp[freq_idx] - self.ground_truth_amp[freq_idx]) *
                                               self.weight_amp_vec[freq_idx] /
                                               self.std_amp_vec[freq_idx]) / len(best_fit_obs_amp[freq_idx]))
                    percent_error_vec[II] = 100*np.sum(np.abs((best_fit_obs_amp[freq_idx] - self.ground_truth_amp[freq_idx]) /
                                                       np.mean(self.ground_truth_amp[freq_idx]))
                                                       / len(best_fit_obs_amp[freq_idx]))
                    if phase:
                        phase_error_vec[II] = np.sum(np.abs((best_fit_obs_phase[freq_idx] - self.ground_truth_phase[freq_idx])*
                                                            self.weight_phase_vec[freq_idx]))/len(best_fit_obs_phase[freq_idx])


            # axs[row_idx, col_idx].set_ylim(ymin=0.0)
            # axs[row_idx, col_idx].set_yticks(np.arange(0, 21, 10))

            plot_saved = False
            col_idx = col_idx + 1
            if col_idx%subplot_width == 0:
                col_idx = 0
                row_idx += 1
                if row_idx%subplot_width == 0:
                    for JJ in range(subplot_width):
                        fig.align_ylabels(axs[:, JJ])
                        if phase:
                            fig_phase.align_ylabels(axs_phase[:, JJ])
                    axs[subplot_width-1, subplot_width-1].legend(loc='upper right', fontsize=12)
                    if phase:
                        axs_phase[subplot_width-1, subplot_width-1].legend(loc='upper right', fontsize=12)
                    fig.tight_layout()
                    if phase:
                        fig_phase.tight_layout()
                    fig.savefig(os.path.join(self.plot_dir,
                                             f'reconstruct_{self.file_name_prefix}_'
                                             f'{self.param_id_obs_file_prefix}_{plot_idx}.eps'))
                    fig.savefig(os.path.join(self.plot_dir,
                                             f'reconstruct_{self.file_name_prefix}_'
                                             f'{self.param_id_obs_file_prefix}_{plot_idx}.pdf'))
                    fig.savefig(os.path.join(self.plot_dir,
                                             f'reconstruct_{self.file_name_prefix}_'
                                             f'{self.param_id_obs_file_prefix}_{plot_idx}.png'))
                    plt.close(fig)
                    
                    if phase:
                        fig_phase.savefig(os.path.join(self.plot_dir,
                                                f'phase_reconstruct_{self.file_name_prefix}_'
                                                f'{self.param_id_obs_file_prefix}_{plot_idx}.eps'))
                        fig_phase.savefig(os.path.join(self.plot_dir,
                                                f'phase_reconstruct_{self.file_name_prefix}_'
                                                f'{self.param_id_obs_file_prefix}_{plot_idx}.pdf'))
                        fig_phase.savefig(os.path.join(self.plot_dir,
                                                f'phase_reconstruct_{self.file_name_prefix}_'
                                                f'{self.param_id_obs_file_prefix}_{plot_idx}.png'))

                        plt.close(fig_phase)

                    plot_saved = True
                    col_idx = 0
                    row_idx = 0
                    plot_idx += 1
                    # create new plot
                    if unique_obs_count != len(obs_names_unique) - 1:
                        fig, axs = plt.subplots(subplot_width, subplot_width, squeeze=False)
                        if phase:
                            fig_phase, axs_phase = plt.subplots(subplot_width, subplot_width, squeeze=False)
                        plot_saved = False

        # save final plot if it is not a full set of subplots
        if not plot_saved:
            for JJ in range(subplot_width):
                fig.align_ylabels(axs[:, JJ])
            axs[0, 0].legend(loc='lower right', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir,
                                     f'reconstruct_{self.file_name_prefix}_'
                                     f'{self.param_id_obs_file_prefix}_{plot_idx}.eps'))
            plt.savefig(os.path.join(self.plot_dir,
                                     f'reconstruct_{self.file_name_prefix}_'
                                     f'{self.param_id_obs_file_prefix}_{plot_idx}.pdf'))
            plt.close()

        # Make a bar plot with all percentage errors.
        fig, axs = plt.subplots()
        obs_names_for_plot_list = []

        for II in range(self.num_obs):
            if "name_for_plotting" in self.gt_df.iloc[0].keys():
                obs_names_for_plot_list.append(f'${self.gt_df.iloc[II]["name_for_plotting"]}\,{self.gt_df.iloc[II]["obs_type"]}$')
            else:
                obs_names_for_plot_list.append(self.obs_names[II])
        obs_names_for_plot = np.array(obs_names_for_plot_list)

        bar_list = axs.bar(obs_names_for_plot, percent_error_vec, label='% error', width=1.0, color='b', edgecolor='black')
        axs.axhline(y=0.0,linewidth= 3, color='k', linestyle= 'dotted')

        # bar_list[0].set_facecolor('r')
        # bar_list[1].set_facecolor('r')

        # axs.legend()
        axs.set_ylabel('E$_{\%}$')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir,
                                 f'error_bars_{self.file_name_prefix}_'
                                 f'{self.param_id_obs_file_prefix}.eps'))
        plt.savefig(os.path.join(self.plot_dir,
                                 f'error_bars_{self.file_name_prefix}_'
                                 f'{self.param_id_obs_file_prefix}.pdf'))
        plt.savefig(os.path.join(self.plot_dir,
                                 f'error_bars_{self.file_name_prefix}_'
                                 f'{self.param_id_obs_file_prefix}.png'))
        plt.close()

        #plot error as number of standard deviations of
        fig, axs = plt.subplots()
        bar_list = axs.bar(obs_names_for_plot, std_error_vec, label='% error', width=1.0, color='b', edgecolor='black')
        axs.axhline(y=0.0,linewidth=3, color='k', linestyle= 'dotted')

        # bar_list[0].set_facecolor('r')
        # bar_list[1].set_facecolor('r')

        # axs.legend()
        axs.set_ylabel('E$_{std}$')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir,
                                 f'std_error_bars_{self.file_name_prefix}_'
                                 f'{self.param_id_obs_file_prefix}.eps'))
        plt.savefig(os.path.join(self.plot_dir,
                                 f'std_error_bars_{self.file_name_prefix}_'
                                 f'{self.param_id_obs_file_prefix}.pdf'))
        plt.savefig(os.path.join(self.plot_dir,
                                 f'std_error_bars_{self.file_name_prefix}_'
                                 f'{self.param_id_obs_file_prefix}.png'))
        plt.close()

        print('______observable errors______')
        for obs_idx in range(self.num_obs):
            if self.gt_df.iloc[obs_idx]["data_type"] == "constant":
                print(f'{self.obs_names[obs_idx]} {self.obs_types[obs_idx]} error:')
                print(f'{percent_error_vec[obs_idx]:.2f} %')
            if self.gt_df.iloc[obs_idx]["data_type"] == "series":
                print(f'{self.obs_names[obs_idx]} {self.obs_types[obs_idx]} error:')
                print(f'{percent_error_vec[obs_idx]:.2f} %')
            if self.gt_df.iloc[obs_idx]["data_type"] == "frequency":
                print(f'{self.obs_names[obs_idx]} {self.obs_types[obs_idx]} error:')
                print(f'{percent_error_vec[obs_idx]:.2f} %')
                if phase:
                    print(f'{self.obs_names[obs_idx]} {self.obs_types[obs_idx]} phase error:')
                    print(f'{phase_error_vec[obs_idx]:.2f}')

    def get_mcmc_samples(self):
        mcmc_chain_path = os.path.join(self.output_dir, 'mcmc_chain.npy')

        if not os.path.exists(mcmc_chain_path):
            print('No mcmc results to get chain')
            return

        samples = np.load(os.path.join(self.output_dir, 'mcmc_chain.npy'))
        num_steps = samples.shape[0]
        num_walkers = samples.shape[1]
        num_params = samples.shape[2]  #
        if self.mcmc_instead:
            if num_params != mcmc_object.num_params:
                print('num params in mcmc chain doesn\'t equal mcmc_object number of params')
        else:
            if num_params != self.param_id.num_params:
                print('num params in mcmc chain doesn\'t equal param_id number of params')

        # TODO fix the below
        # for some reason some chains get stuck for long times, remove the chains that get stuck
        # I think this occurs when initialisation happens outside of the prior distribution
        walkers_to_remove = []
        for walker_idx in range(num_walkers):
            for param_idx in range(num_params):
                block_size = 200
                for step_block_idx in range(num_steps//block_size):
                    # get std of the block and remove that chain it if is zero
                    block_std = np.std(samples[step_block_idx*block_size:(step_block_idx+1)*block_size, walker_idx, param_idx])
                    if block_std == 0:
                        walkers_to_remove.append(walker_idx)

        walkers_to_remove = list(set(walkers_to_remove))
        if len(walkers_to_remove) > 0:
            print('There is a bug where chains can get stuck, removing walkers with stuck parameters. removed walker idxs:')
            print(walkers_to_remove)
            samples = np.delete(samples, walkers_to_remove, axis=1)

        # discard first num_steps/2 samples
        # TODO include a user defined burn in if we aren't starting from
        samples = samples[samples.shape[0]//2:, :, :]
        # thin = 5
        # samples = samples[::thin, :, :]
        flat_samples = samples.reshape(-1, num_params)

        return flat_samples, samples, num_params

    def plot_mcmc(self):

        flat_samples, samples, num_params = self.get_mcmc_samples()
        if self.rank != 0:
            return

        means = np.zeros((num_params))
        conf_ivals = np.zeros((num_params, 3))

        for param_idx in range(num_params):
            means[param_idx] = np.mean(flat_samples[:, param_idx])
            conf_ivals[param_idx, :] = np.percentile(flat_samples[:, param_idx], [5, 50, 95])

        print('5th, 50th, and 95th percentile parameter values are:')
        print(conf_ivals)

        fig, axes = plt.subplots(num_params, figsize=(10, 7), sharex=True)
        for i in range(num_params):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(f'${self.param_names_for_plotting[i]}$')
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")
        # plt.savefig(os.path.join(self.output_dir, 'plots_param_id', 'mcmc_chain_plot.eps'))
        plt.savefig(os.path.join(self.output_dir, 'plots_param_id', 'mcmc_chain_plot.pdf'))
        plt.close()

        label_list = [f'${self.param_names_for_plotting[II]}$' for II in range(len(self.param_names_for_plotting))]
        if self.mcmc_instead:
            if mcmc_object.best_param_vals is None:
                best_param_vals = np.load(os.path.join(self.output_dir, 'best_param_vals.npy'))
                mcmc_object.set_best_param_vals(best_param_vals)
        else:
            if self.param_id.best_param_vals is None:
                best_param_vals = np.load(os.path.join(self.output_dir, 'best_param_vals.npy'))
                self.param_id.set_best_param_vals(best_param_vals)

        overwrite_params_to_plot_idxs = [II for II in range(num_params)] # This plots all param distributions
        if self.mcmc_instead:
            fig = corner.corner(flat_samples[:, overwrite_params_to_plot_idxs], bins=20, hist_bin_factor=2, smooth=0.5, quantiles=(0.05, 0.5, 0.95),
                                labels=[label_list[II] for II in overwrite_params_to_plot_idxs],
                                truths=mcmc_object.best_param_vals[overwrite_params_to_plot_idxs],
                                fontsize=20)
        else:
            fig = corner.corner(flat_samples[:, overwrite_params_to_plot_idxs], bins=20, hist_bin_factor=2, smooth=0.5, quantiles=(0.05, 0.5, 0.95),
                                labels=[label_list[II] for II in overwrite_params_to_plot_idxs],
                                truths=self.param_id.best_param_vals[overwrite_params_to_plot_idxs],
                                fontsize=20)
        axes = fig.get_axes()
        for idx, ax in enumerate(axes):
            if idx >= num_params*(num_params - 1):

                ax.tick_params(axis='both', rotation=0)
                formatterx = matplotlib.ticker.ScalarFormatter()
                ax.xaxis.set_major_formatter(formatterx)
                ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            if idx%num_params == 0:

                ax.tick_params(axis='both', rotation=0)
                formattery = matplotlib.ticker.ScalarFormatter()
                ax.yaxis.set_major_formatter(formattery)
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        plt.subplots_adjust(hspace=0.12, wspace=0.1)

        plt.savefig(os.path.join(self.plot_dir, f'mcmc_cornerplot_{self.file_name_prefix}_'
                                                f'{self.param_id_obs_file_prefix}.pdf'))
        plt.close()

        # do another corner plot with just a subset of params
        # overwrite_params_to_plot_idxs = [0,1, 4, 7] # This chooses a subset of params to plot
        if self.mcmc_instead:
            fig = corner.corner(flat_samples[:, overwrite_params_to_plot_idxs], bins=20, hist_bin_factor=2, smooth=0.5, quantiles=(0.05, 0.5, 0.95),
                                labels=[label_list[II] for II in overwrite_params_to_plot_idxs],
                                truths=mcmc_object.best_param_vals[overwrite_params_to_plot_idxs],
                                fontsize=20)
        else:
            fig = corner.corner(flat_samples[:, overwrite_params_to_plot_idxs], bins=20, hist_bin_factor=2, smooth=0.5, quantiles=(0.05, 0.5, 0.95),
                                labels=[label_list[II] for II in overwrite_params_to_plot_idxs],
                                truths=self.param_id.best_param_vals[overwrite_params_to_plot_idxs],
                                fontsize=20)
        axes = fig.get_axes()
        for idx, ax in enumerate(axes):
            if idx >= len(overwrite_params_to_plot_idxs)*(len(overwrite_params_to_plot_idxs) - 1):

                ax.tick_params(axis='both', rotation=0)
                formatterx = matplotlib.ticker.ScalarFormatter()
                ax.xaxis.set_major_formatter(formatterx)
                ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            if idx%len(overwrite_params_to_plot_idxs) == 0:

                ax.tick_params(axis='both', rotation=0)
                formattery = matplotlib.ticker.ScalarFormatter()
                ax.yaxis.set_major_formatter(formattery)
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        plt.subplots_adjust(hspace=0.12, wspace=0.1)

        plt.savefig(os.path.join(self.plot_dir, f'mcmc_cornerplot_subset_{self.file_name_prefix}_'
                                                f'{self.param_id_obs_file_prefix}.pdf'))
        plt.close()

        # Also check autocorrelation times for mcmc chain
        tau = self.calculate_autocorrelation_time(samples)

        # check geweke convergence
        acceptable = self.calculate_geweke_convergence(samples)
        if acceptable:
            print('chain passed geweke diagnostic with p>0.05')
        else:
            print('chain failed geweke diagnostic with p<0.05, USE CHAIN RESULTS WITH CARE')

    def calculate_autocorrelation_time(self, samples):
        tau = emcee.autocorr.integrated_time(samples, quiet=True)
        return tau

    def calculate_geweke_convergence(self, samples):
        d = diagnostics.Diagnostics()
        acceptable = d.geweke(samples, first=0.3, last=0.5)
        return acceptable

    def run_single_sensitivity(self, do_triples_and_quads):
        self.param_id.run_single_sensitivity(self.output_dir, do_triples_and_quads)

    def run_sensitivity(self, param_id_output_paths):
        # TODO change this function to plot_sensitivity
        if self.rank !=0:
            return

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


        # FA: Eventually we will automate the multiple runs of param_id and store the outputs in idxed
        # FA: directories that can all be accessed automatically by this function without defining input paths.
        m3_to_cm3 = 1e6
        Pa_to_kPa = 1e-3
        number_samples = len(sample_path_list)
        x_values = []
        for i in range(len(self.param_names)):
            x_values.append(self.param_names[i][0])

        for i in range(number_samples):
            sample_path = sample_path_list[i]
            normalised_jacobian = np.load(os.path.join(sample_path, 'normalised_jacobian_matrix.npy'))
            parameter_importance = np.load(os.path.join(sample_path, 'parameter_importance.npy'))
            collinearity_idx_i = np.load(os.path.join(sample_path, 'collinearity_idx.npy'))
            if i == 0:
                normalised_jacobian_average = np.zeros(normalised_jacobian.shape)
                parameter_importance_average = np.zeros(parameter_importance.shape)
                collinearity_idx_average = np.zeros(collinearity_idx_i.shape)
            normalised_jacobian_average = normalised_jacobian_average + normalised_jacobian/number_samples
            parameter_importance_average = parameter_importance_average + parameter_importance/number_samples
            collinearity_idx_average = collinearity_idx_average + collinearity_idx_i/number_samples
            self.collinearity_idx = collinearity_idx_average

        collinearity_max = collinearity_idx_average.max()

        if do_triples_and_quads:
            #find maximum average value of collinearity triples
            for i in range(len(x_values)):
                for j in range(number_samples):
                    sample_path = sample_path_list[j]
                    collinearity_idx_triple = np.load(os.path.join(sample_path, 'collinearity_triples' + str(i) + '.npy'))
                    if j==0:
                        collinearity_idx_triple_average = np.zeros(collinearity_idx_triple.shape)
                    collinearity_idx_triple_average = collinearity_idx_triple_average + collinearity_idx_triple/number_samples
                if collinearity_idx_triple_average.max() > collinearity_max:
                    collinearity_max = collinearity_idx_triple_average.max()

            #find maximum value of collinearity quads
            for i in range(len(x_values)):
                for j in range(len(x_values)):
                    for k in range(number_samples):
                        sample_path = sample_path_list[k]
                        collinearity_idx_quad = np.load(
                            os.path.join(sample_path, 'collinearity_quads' + str(i) + '_' + str(j) + '.npy'))
                        if k==0:
                            collinearity_idx_quad_average = np.zeros(collinearity_idx_quad.shape)
                        collinearity_idx_quad_average = collinearity_idx_quad_average + collinearity_idx_quad / number_samples
                    if collinearity_idx_quad_average.max() > collinearity_max:
                        collinearity_max = collinearity_idx_quad_average.max()

        #find maximum average value and average values for collinearity idx
        for i in range(number_samples):
            sample_path = sample_path_list[i]
            collinearity_idx_pairs_i = np.load(
                os.path.join(sample_path, 'collinearity_pairs.npy'))
            if i==0:
                collinearity_idx_pairs_average = np.zeros(collinearity_idx_pairs_i.shape)
            collinearity_idx_pairs_average = collinearity_idx_pairs_average + collinearity_idx_pairs_i/number_samples
            self.collinearity_idx_pairs = collinearity_idx_pairs_average

        if collinearity_idx_pairs_average.max() > collinearity_max:
            collinearity_max = collinearity_idx_pairs_average.max()

        number_Params = len(normalised_jacobian_average)
        #plot jacobian
        subplot_height = 1
        subplot_width = 1
        plt.rc('xtick', labelsize=3)
        fig, axs = plt.subplots(subplot_height, subplot_width)

        x_labels = []
        subset = []
        x_idx = 0
        for obs_idx in range(len(self.obs_names)):
            # if self.obs_types[obs_idx] != "series":
            x_labels.append(self.obs_names[obs_idx] + " " + self.obs_types[obs_idx])
            subset.append(x_idx)
            x_idx = x_idx + 1

        for param_idx in range(len(self.param_names)):
            y_values = []
            for obs_idx in range(len(x_labels)):
                y_values.append(abs((normalised_jacobian_average[param_idx][subset[obs_idx]])))
            axs.plot(x_labels, y_values, label=f'${self.param_names_for_plotting[param_idx][0]}$')
        axs.set_yscale('log')
        axs.legend(loc='lower left', fontsize=6)
        plt.savefig(os.path.join(self.plot_dir,
                                     f'reconstruct_{self.file_name_prefix}_'
                                     f'{self.param_id_obs_file_prefix}_sensitivity_average.eps'))
        plt.savefig(os.path.join(self.plot_dir,
                                     f'reconstruct_{self.file_name_prefix}_'
                                     f'{self.param_id_obs_file_prefix}_sensitivity_average.pdf'))
        plt.close()
        #plot parameter importance
        plt.rc('xtick', labelsize=6)
        plt.rc('ytick', labelsize=12)
        figB, axsB = plt.subplots(1, 1)


        axsB.bar(x_values, parameter_importance_average)
        axsB.set_ylabel("Parameter Importance", fontsize=12)
        plt.savefig(os.path.join(self.plot_dir,
                                     f'reconstruct_{self.file_name_prefix}_'
                                     f'{self.param_id_obs_file_prefix}_parameter_importance_average.eps'))
        plt.savefig(os.path.join(self.plot_dir,
                                     f'reconstruct_{self.file_name_prefix}_'
                                     f'{self.param_id_obs_file_prefix}_parameter_importance_average.pdf'))
        plt.close()
        #plot collinearity idx average
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=4)
        figC, axsC = plt.subplots(1, 1)
        x_values_cumulative = []
        x_values_temp = x_values[0]
        for i in range(len(x_values)):
            x_values_cumulative.append(x_values_temp)
            if (i + 1) < len(x_values):
                x_values_temp = x_values[i + 1] + "\n" + x_values_temp
        axsC.barh(x_values_cumulative, collinearity_idx_average)
        plt.savefig(os.path.join(self.plot_dir,
                                     f'reconstruct_{self.file_name_prefix}_'
                                     f'{self.param_id_obs_file_prefix}_collinearity_index_average.eps'))
        plt.savefig(os.path.join(self.plot_dir,
                                     f'reconstruct_{self.file_name_prefix}_'
                                     f'{self.param_id_obs_file_prefix}_collinearity_index_average.pdf'))
        plt.close()

        plt.rc('xtick', labelsize=4)
        plt.rc('ytick', labelsize=4)
        figD, axsD = plt.subplots(1, 1)

        X, Y = np.meshgrid(range(len(x_values)), range(len(x_values)))
        if do_triples_and_quads:
            collinearity_levels = np.linspace(0, collinearity_max, 20)
            co = axsD.contourf(X, Y, collinearity_idx_pairs_average, levels=collinearity_levels)
        else:
            co = axsD.contourf(X, Y, collinearity_idx_pairs_average)
        co = fig.colorbar(co, ax=axsD)
        axsD.set_xticks(range(len(x_values)))
        axsD.set_yticks(range(len(x_values)))
        axsD.set_xticklabels(x_values)
        axsD.set_yticklabels(x_values)

        plt.savefig(os.path.join(self.plot_dir,
                                     f'reconstruct_{self.file_name_prefix}_'
                                     f'{self.param_id_obs_file_prefix}_collinearity_pairs_average.eps'))
        plt.savefig(os.path.join(self.plot_dir,
                                     f'reconstruct_{self.file_name_prefix}_'
                                     f'{self.param_id_obs_file_prefix}_collinearity_pairs_average.pdf'))

        if do_triples_and_quads:
            #plot collinearity triples average
            for i in range(len(x_values)):
                for j in range(number_samples):
                    sample_path = sample_path_list[j]
                    collinearity_idx_triple = np.load(os.path.join(sample_path, 'collinearity_triples' + str(i) + '.npy'))
                    if j == 0:
                        collinearity_idx_triple_average = np.zeros(collinearity_idx_triple.shape)
                    collinearity_idx_triple_average = collinearity_idx_triple_average + collinearity_idx_triple/number_samples

                figE, axsE = plt.subplots(1, 1)
                co = axsE.contourf(X, Y, collinearity_idx_triple_average, levels=collinearity_levels)
                co = fig.colorbar(co, ax=axsE)
                axsE.set_xticks(range(len(x_values)))
                axsE.set_yticks(range(len(x_values)))
                axsE.set_xticklabels(x_values)
                axsE.set_yticklabels(x_values)

                plt.savefig(os.path.join(self.plot_dir,
                                             f'reconstruct_{self.file_name_prefix}_'
                                             f'{self.param_id_obs_file_prefix}_collinearity_triples_average' + str(i) + '.eps'))
                plt.savefig(os.path.join(self.plot_dir,
                                             f'reconstruct_{self.file_name_prefix}_'
                                             f'{self.param_id_obs_file_prefix}_collinearity_triples_average' + str(i) + '.pdf'))
                plt.close()
            #plot collinearity quads average
            for i in range(len(x_values)):
                for j in range(len(x_values)):
                    for k in range(number_samples):
                        sample_path = sample_path_list[k]
                        collinearity_idx_quad = np.load(
                            os.path.join(sample_path, 'collinearity_quads' + str(i) + '_' + str(j) + '.npy'))
                        if k==0:
                            collinearity_idx_quad_average = np.zeros(collinearity_idx_quad.shape)
                        collinearity_idx_quad_average = collinearity_idx_quad_average + collinearity_idx_quad / number_samples
                    figE, axsE = plt.subplots(1, 1)
                    co = axsE.contourf(X, Y, collinearity_idx_quad_average, levels=collinearity_levels)
                    co = fig.colorbar(co, ax=axsE)
                    axsE.set_xticks(range(len(x_values)))
                    axsE.set_yticks(range(len(x_values)))
                    axsE.set_xticklabels(x_values)
                    axsE.set_yticklabels(x_values)

                    plt.savefig(os.path.join(self.plot_dir,
                                                 f'reconstruct_{self.file_name_prefix}_'
                                                 f'{self.param_id_obs_file_prefix}collinearity_quads_average' + str(i) + '_' + str(
                                                     j) + '.eps'))
                    plt.savefig(os.path.join(self.plot_dir,
                                                 f'reconstruct_{self.file_name_prefix}_'
                                                 f'{self.param_id_obs_file_prefix}collinearity_quads_average' + str(i) + '_' + str(
                                                     j) + '.pdf'))
                    plt.close()

        self.sensitivity_calculated = True
        print('sensitivity analysis complete')

    def __get_prediction_data(self):
        if self.rank !=0:
            return

        tSim = self.param_id.sim_helper.tSim - self.param_id.pre_time
        pred_output = self.param_id.sim_helper.get_results(self.pred_var_names)
        time_and_pred = np.concatenate((tSim.reshape(1, -1), pred_output))
        return time_and_pred

    def save_prediction_data(self):
        if self.rank !=0:
            return
        if self.pred_var_names is not None:
            print('Saving prediction data')
            time_and_pred = self.__get_prediction_data()

            #save the prediction output
            np.save(os.path.join(self.output_dir, 'prediction_variable_data'), time_and_pred)
            print('single prediction data saved')

        else:
            pred_var_path = os.path.join(resources_dir, f'{self.file_name_prefix}_prediction_variables.csv')
            print(f'prediction variables have not been defined, if you want to save predicition variables,',
                  f'create a file {pred_var_path}, with the names of the desired prediction variables')

        return

    def set_genetic_algorithm_parameters(self, n_calls):
        self.param_id.set_genetic_algorithm_parameters(n_calls)

    def set_bayesian_parameters(self, n_calls, n_initial_points, acq_func, random_state, acq_func_kwargs={}):
        self.param_id.set_bayesian_parameters(n_calls, n_initial_points, acq_func, random_state,
                                              acq_func_kwargs=acq_func_kwargs)

    def close_simulation(self):
        if self.mcmc_instead:
            mcmc_object.close_simulation()
        else:
            self.param_id.close_simulation()

    def __set_obs_names_and_df(self, param_id_obs_path):
        # TODO this function should be in the parsing section. as it parses the 
        # ground truth data.
        with open(param_id_obs_path, encoding='utf-8-sig') as rf:
            json_obj = json.load(rf)
        self.gt_df = pd.DataFrame(json_obj)
        if self.gt_df.columns[0] == 'data_item':
            self.gt_df = self.gt_df['data_item']

        self.obs_names = [self.gt_df.iloc[II]["variable"] for II in range(self.gt_df.shape[0])]

        self.obs_types = [self.gt_df.iloc[II]["obs_type"] for II in range(self.gt_df.shape[0])]


        self.obs_operations = []
        self.obs_operands = []
        self.obs_freqs = []
        for II in range(self.gt_df.shape[0]):
            if "operation" not in self.gt_df.iloc[II].keys():
                self.obs_operations.append(None)
                self.obs_operands.append(None)
            elif self.gt_df.iloc[II]["operation"] == "Null":
                self.obs_operations.append(None)
                self.obs_operands.append(None)
            else:
                self.obs_operations.append(self.gt_df.iloc[II]["operation"])
                self.obs_operands.append(self.gt_df.iloc[II]["operands"])

            if "frequencies" not in self.gt_df.iloc[II].keys():
                self.obs_freqs.append(None)
            else:
                self.obs_freqs.append(self.gt_df.iloc[II]["frequencies"])

        self.num_obs = len(self.obs_names)

        # how much to weight the different observable errors by
        self.weight_const_vec = np.array([self.gt_df.iloc[II]["weight"] for II in range(self.gt_df.shape[0])
                                          if self.gt_df.iloc[II]["data_type"] == "constant"])

        self.weight_series_vec = np.array([self.gt_df.iloc[II]["weight"] for II in range(self.gt_df.shape[0])
                                           if self.gt_df.iloc[II]["data_type"] == "series"])

        self.weight_amp_vec = np.array([self.gt_df.iloc[II]["weight"] for II in range(self.gt_df.shape[0])
                                           if self.gt_df.iloc[II]["data_type"] == "frequency"])
        weight_phase_list = [] 
        for II in range(self.gt_df.shape[0]):
            if self.gt_df.iloc[II]["data_type"] == "frequency":
                if "phase_weight" not in self.gt_df.iloc[II].keys():
                    weight_phase_list.append(1)
                else:
                    weight_phase_list.append(self.gt_df.iloc[II]["phase_weight"])
        self.weight_phase_vec = np.array(weight_phase_list)

        return

    def __set_and_save_param_names(self, idxs_to_ignore=None):
        # This should also be a function under parsers.

        # Each entry in param_names is a name or list of names that gets modified by one parameter
        if self.input_params_path:
            csv_parser = CSVFileParser()
            input_params = csv_parser.get_data_as_dataframe_multistrings(self.input_params_path)
            self.param_names = []
            param_names_for_gen = []
            param_state_names_for_gen = []
            param_const_names_for_gen = []
            for II in range(input_params.shape[0]):
                if idxs_to_ignore is not None:
                    if II in idxs_to_ignore:
                        continue
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


            # set param ranges from file and strings for plotting parameter names
            if idxs_to_ignore is not None:
                self.param_mins = np.array([float(input_params["min"][JJ]) for JJ in range(input_params.shape[0])
                                            if JJ not in idxs_to_ignore])
                self.param_maxs = np.array([float(input_params["max"][JJ]) for JJ in range(input_params.shape[0])
                                            if JJ not in idxs_to_ignore])
                if "name_for_plotting" in input_params.columns:
                    self.param_names_for_plotting = np.array([input_params["name_for_plotting"][JJ]
                                                            for JJ in range(input_params.shape[0])
                                                            if JJ not in idxs_to_ignore])
                else:
                    self.param_names_for_plotting = np.array([self.param_names[JJ][0]
                                                            for JJ in range(len(self.param_names))
                                                            if JJ not in idxs_to_ignore])
            else:
                self.param_mins = np.array([float(input_params["min"][JJ]) for JJ in range(input_params.shape[0])])
                self.param_maxs = np.array([float(input_params["max"][JJ]) for JJ in range(input_params.shape[0])])
                if "name_for_plotting" in input_params.columns:
                    self.param_names_for_plotting = np.array([input_params["name_for_plotting"][JJ]
                                                            for JJ in range(input_params.shape[0])])
                else:
                    self.param_names_for_plotting = np.array([param_name[0] for param_name in self.param_names])

            # set param_priors
            if "prior" in input_params.columns:
                self.param_prior_types = np.array([input_params["prior"][JJ] for JJ in range(input_params.shape[0])])
            else:
                self.param_prior_types = np.array(["uniform" for JJ in range(input_params.shape[0])])


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

    def __get_ground_truth_values(self):

        # _______ First we access data for constant values

        ground_truth_const = np.array([self.gt_df.iloc[II]["value"] for II in range(self.gt_df.shape[0])
                                        if self.gt_df.iloc[II]["data_type"] == "constant"])

        # _______ Then for time series
        ground_truth_series = np.array([self.gt_df.iloc[II]["value"] for II in range(self.gt_df.shape[0])
                                        if self.gt_df.iloc[II]["data_type"] == "series"])

        # _______ Then for frequency series
        ground_truth_amp = np.array([self.gt_df.iloc[II]["value"] for II in range(self.gt_df.shape[0])
                                        if self.gt_df.iloc[II]["data_type"] == "frequency"])

        # _______ and the phase of the freq data
        ground_truth_phase_list = []
        for II in range(self.gt_df.shape[0]):
            if self.gt_df.iloc[II]["data_type"] == "frequency":
                if "phase" not in self.gt_df.iloc[II].keys():
                    ground_truth_phase_list.append(None)
                else:
                    ground_truth_phase_list.append(self.gt_df.iloc[II]["phase"])
        ground_truth_phase = np.array(ground_truth_phase_list)

        # The std for the different observables
        self.std_const_vec = np.array([self.gt_df.iloc[II]["std"] for II in range(self.gt_df.shape[0])
                                       if self.gt_df.iloc[II]["data_type"] == "constant"])

        self.std_series_vec = np.array([np.mean(self.gt_df.iloc[II]["std"]) for II in range(self.gt_df.shape[0])
                                        if self.gt_df.iloc[II]["data_type"] == "series"])

        self.std_amp_vec = np.array([np.mean(self.gt_df.iloc[II]["std"]) for II in range(self.gt_df.shape[0])
                                        if self.gt_df.iloc[II]["data_type"] == "frequency"])

        if len(ground_truth_series) > 0:
            # TODO what if we have ground truths of different size or sample rate?
            ground_truth_series = np.stack(ground_truth_series)

        if len(ground_truth_amp) > 0:
            ground_truth_amp = np.stack(ground_truth_amp)

        if len(ground_truth_phase) > 0:
            ground_truth_phase = np.stack(ground_truth_phase)

        if self.rank == 0:
            np.save(os.path.join(self.output_dir, 'ground_truth_const.npy'), ground_truth_const)
            if len(ground_truth_series) > 0:
                np.save(os.path.join(self.output_dir, 'ground_truth_series.npy'), ground_truth_series)
            if len(ground_truth_amp) > 0:
                np.save(os.path.join(self.output_dir, 'ground_truth_amp.npy'), ground_truth_amp)
            if len(ground_truth_phase) > 0:
                np.save(os.path.join(self.output_dir, 'ground_truth_phase.npy'), ground_truth_phase)

        return ground_truth_const, ground_truth_series, ground_truth_amp, ground_truth_phase

    
    def get_best_param_vals(self):
        if self.mcmc_instead:
            return mcmc_object.best_param_vals
        else:
            return self.param_id.best_param_vals

    def get_param_names(self):
        if self.mcmc_instead:
            return mcmc_object.param_names
        else:
            return self.param_id.param_names

    def get_param_importance(self):
        return self.param_id.param_importance

    def get_collinearity_idx(self):
        return self.param_id.collinearity_idx

    def get_collinearity_idx_pairs(self):
        return self.param_id.collinearity_idx_pairs

    def get_pred_param_importance(self):
        return self.param_id.pred_param_importance

    def get_pred_collinearity_idx_pairs(self):
        return self.param_id.pred_collinearity_idx_pairs

    def remove_params_by_idx(self, param_idxs_to_remove):
        self.__set_and_save_param_names(idxs_to_ignore=param_idxs_to_remove)
        if self.mcmc_instead:
            mcmc_object.remove_params_by_idx(param_idxs_to_remove)
        else:
            self.param_id.remove_params_by_idx(param_idxs_to_remove)

    def remove_params_by_name(self, param_names_to_remove):
        param_idxs_to_remove = []
        if self.mcmc_instead:
            num_params = mcmc_object.num_params
        else:
            num_params = self.param_id.num_params

        for II in range(num_params):
            if self.param_names[II] in param_names_to_remove:
                param_idxs_to_remove.append(II)

        self.remove_params_by_idx(param_idxs_to_remove)

    def __set_prediction_var(self):
        # prediction variables
        pred_var_path = os.path.join(resources_dir, f'{self.file_name_prefix}_prediction_variables.csv')
        if os.path.exists(pred_var_path):
            # TODO change this to loading with parser
            csv_parser = CSVFileParser()
            self.pred_var_df = csv_parser.get_data_as_dataframe_multistrings(pred_var_path)
            self.pred_var_names = np.array([self.pred_var_df["name"][II].strip()
                                            for II in range(self.pred_var_df.shape[0])])
            self.pred_var_units = np.array([self.pred_var_df["unit"][II].strip()
                                            for II in range(self.pred_var_df.shape[0])])
            self.pred_var_names_for_plotting = np.array([self.pred_var_df["name_for_plotting"][II].strip()
                                            for II in range(self.pred_var_df.shape[0])])
        else:
            self.pred_var_names = None
            self.pred_var_units = None
            self.pred_var_names_for_plotting = None

        #if len(self.pred_var_names) < 1:
        #    self.pred_var_names = None
        #    self.pred_var_units = None
        #    self.pred_var_names_for_plotting = None

        if self.pred_var_names is None:
            self.pred_var_names = None
            self.pred_var_units = None
            self.pred_var_names_for_plotting = None

    def postprocess_predictions(self):
        if self.pred_var_names == None:
            print('no prediction variables, not plotting predictions')
            return 0
        m3_to_cm3 = 1e6
        Pa_to_kPa = 1e-3

        flat_samples, _, _ = self.get_mcmc_samples()
        # this array is of size (num_pred_var, num_samples,
        if self.DEBUG:
            n_sims = 6
        else:
            n_sims = 100

        pred_array = mcmc_object.calculate_var_from_posterior_samples(self.pred_var_names, flat_samples, n_sims=n_sims)
        if self.mcmc_instead:
            tSim = mcmc_object.sim_helper.tSim - mcmc_object.pre_time
        else:
            tSim = self.param_id.sim_helper.tSim - self.param_id.pre_time

        fig, axs = plt.subplots(len(self.pred_var_names))
        if len(self.pred_var_names) == 1:
            axs = [axs]

        save_list = []
        for pred_idx in range(len(self.pred_var_names)):
            #TODO conversion
            if self.pred_var_units[pred_idx] == 'm3_per_s':
                conversion = m3_to_cm3
                unit_for_plot = '$cm^3/s$'
            elif self.pred_var_units[pred_idx] == 'm_per_s':
                conversion = 1.0
                unit_for_plot = '$m/s$'
            elif self.pred_var_units[pred_idx] == 'm3':
                conversion = m3_to_cm3
                unit_for_plot = '$cm^3$'
            elif self.pred_var_units[pred_idx] == 'J_per_m3':
                conversion = Pa_to_kPa
                unit_for_plot = '$kPa$'
            else:
                print(f'unit of {self.pred_var_units} not yet implemented for prediction variables plotting.')
                exit()
            # calculate mean and std of the ensemble
            pred_mean = np.mean(pred_array[pred_idx, :, :], axis=0)
            pred_std = np.std(pred_array[pred_idx, :, :], axis=0)

            # get idxs of max min and mean prediction to plot std bars
            idxs_to_plot_std = [np.argmax(pred_mean), np.argmin(pred_mean),
                                np.argmin(np.abs(pred_mean - np.mean(pred_mean)))]
            # idxs_to_plot_std = [self.n_steps//5*(II) for II in range(6)]
            # TODO put units in prediction file and use it here
            axs[pred_idx].set_xlabel('Time [$s$]', fontsize=14)
            axs[pred_idx].set_ylabel(f'${self.pred_var_names_for_plotting[pred_idx]}$ [{unit_for_plot}]', fontsize=14)
            # for sample_idx in range(pred_array.shape[1]):
            #     axs[pred_idx].plot(tSim, conversion*pred_array[pred_idx, sample_idx, :], 'k')

            axs[pred_idx].plot(tSim, conversion*pred_mean, 'b', label='mean', linewidth=1.5)
            axs[pred_idx].errorbar(tSim[idxs_to_plot_std], conversion*pred_mean[idxs_to_plot_std],
                                   yerr=conversion*pred_std[idxs_to_plot_std], ecolor='b', fmt='^', capsize=6, zorder=3)
            axs[pred_idx].set_xlim(0.0, 1.0)
            # z_star = 1.96 for 95% confidence interval. margin_of_error=z_star*std
            z_star = 1.96
            margin_of_error = z_star * pred_std
            conf_ival_up = pred_mean + margin_of_error
            conf_ival_down = pred_mean - margin_of_error
            axs[pred_idx].plot(tSim, conversion*conf_ival_up, 'r--', label='95% CI', linewidth=1.2)
            axs[pred_idx].plot(tSim, conversion*conf_ival_down, 'r--', linewidth=1.2)
            axs[pred_idx].legend()
            y_max = 10*math.ceil(max(conversion*conf_ival_up)/10)
            axs[pred_idx].set_ylim(ymin=0.0, ymax=y_max)
            # save prediction value, std, and CI of for max, min, and mean
            for idx in idxs_to_plot_std:
                save_list.append(pred_mean[idx])
                save_list.append(pred_std[idx])
                save_list.append(conf_ival_up[idx])
                save_list.append(conf_ival_down[idx])

        # save prediction value, std, and CI of for max, min, and mean
        pred_save_array = conversion*np.array(save_list)
        np.save(os.path.join(self.output_dir, 'prediction_vals_std_ci.npy'), pred_save_array)

        plt.savefig(os.path.join(self.plot_dir,
                                 f'prediction_{self.file_name_prefix}_'
                                 f'{self.param_id_obs_file_prefix}.eps'))
        plt.savefig(os.path.join(self.plot_dir,
                                 f'prediction_{self.file_name_prefix}_'
                                 f'{self.param_id_obs_file_prefix}.pdf'))

        # save param standard deviations
        param_std = np.std(flat_samples, axis=0)
        print(param_std)
        np.save(os.path.join(self.output_dir, 'params_std.npy'), param_std)

class OpencorParamID():
    """
    Class for doing parameter identification on opencor models
    """
    def __init__(self, model_path, param_id_method,
                 obs_names, obs_types, obs_freqs, obs_operations, obs_operands,
                 weight_const_vec, weight_series_vec, 
                 weight_amp_vec, weight_phase_vec,
                 std_const_vec, std_series_vec, std_amp_vec,
                 param_names, pred_var_names,
                 ground_truth_const, ground_truth_series, ground_truth_amp, ground_truth_phase,
                 param_mins, param_maxs,
                 sim_time=2.0, pre_time=20.0, pre_heart_periods=None, sim_heart_periods=None,
                 dt=0.01, maximum_step=0.0001, ga_options=None, DEBUG=False):

        self.model_path = model_path
        self.param_id_method = param_id_method
        self.output_dir = None

        self.obs_names = obs_names
        self.obs_types = obs_types
        self.obs_freqs = obs_freqs
        self.obs_operations = obs_operations
        # check whether we need to output temporary results for doing fft on an operation
        self.output_temp_results = False
        for type, operation in zip(self.obs_types, self.obs_operations):
            if type == 'frequency' and operation != None:
                self.output_temp_results = True
                break
        self.obs_operands = obs_operands
        self.weight_const_vec = weight_const_vec
        self.weight_series_vec = weight_series_vec
        self.weight_amp_vec = weight_amp_vec
        self.weight_phase_vec = weight_phase_vec
        self.std_const_vec = std_const_vec
        self.std_series_vec = std_series_vec
        self.std_amp_vec = std_amp_vec
        self.param_names = param_names
        self.pred_var_names = pred_var_names
        self.num_obs = len(self.obs_names)
        self.num_params = len(self.param_names)
        self.ground_truth_const = ground_truth_const
        self.ground_truth_series = ground_truth_series
        self.ground_truth_amp = ground_truth_amp
        self.ground_truth_phase = ground_truth_phase
        self.param_mins = param_mins
        self.param_maxs = param_maxs
        self.param_norm_obj = Normalise_class(self.param_mins, self.param_maxs)

        # set up opencor simulation
        self.dt = dt  # TODO this could be optimised
        self.maximum_step = maximum_step
        self.point_interval = self.dt
        if sim_time is not None:
            self.sim_time = sim_time
        else:
            # set temporary sim time, just to initialise the sim_helper
            self.sim_time = 0.001
        if pre_time is not None:
            self.pre_time = pre_time
        else:
            # set temporary pre time, just to initialise the sim_helper
            self.pre_time = 0.001

        self.sim_helper = self.initialise_sim_helper()
        # overwrite pre_time and sim_time if pre_heart_periods and sim_heart_periods are defined
        if pre_heart_periods is not None:
            try:
                T = self.sim_helper.get_init_param_vals(['heart/T'])[0]
            except:
                print('ERROR: heart/T not found in model parameters. You should be setting sim_time and pre_time'
                      'instead of pre_heart_periods and sim_heart_periods in user_inputs.yaml'
                      'if your model doesn\'t have a heart period. Exiting')
                exit()
            self.pre_time = T*pre_heart_periods
        if sim_heart_periods is not None:
            try:
                T = self.sim_helper.get_init_param_vals(['heart/T'])[0]
            except:
                print('ERROR: heart/T not found in model parameters. You should be setting sim_time and pre_time'
                      'instead of pre_heart_periods and sim_heart_periods in user_inputs.yaml'
                      'if your model doesn\'t have a heart period. Exiting')
                exit()
            self.sim_time = T*sim_heart_periods

        self.sim_helper.update_times(self.dt, 0.0, self.sim_time, self.pre_time)

        self.sim_helper.create_operation_variables(self.obs_names, self.obs_operations, self.obs_operands)

        self.n_steps = int(self.sim_time/self.dt)

        # initialise
        self.param_init = None
        self.best_param_vals = None
        self.best_cost = np.inf

        # bayesian optimisation constants TODO add more of the constants to this so they can be modified by the user
        # TODO or remove bayesian optimisation, as it is untested
        self.n_calls = 10000
        self.acq_func = 'EI'  # the acquisition function
        self.n_initial_points = 5
        self.acq_func_kwargs = {}
        self.random_state = 1234 # random seed

        # sensitivity
        self.param_importance = None
        self.collinearity_idx = None
        self.collinearity_idx_pairs = None
        self.pred_param_importance = None
        self.pred_collinearity_idx_pairs = None

        if ga_options is not None:
            self.cost_type = ga_options['cost_type']
        else:
            self.cost_type = 'MSE'

        self.DEBUG = DEBUG

    def initialise_sim_helper(self):
        return SimulationHelper(self.model_path, self.dt, self.sim_time,
                                maximumNumberofSteps=100000000,
                                maximum_step=self.maximum_step, pre_time=self.pre_time)
    
    def set_best_param_vals(self, best_param_vals):
        self.best_param_vals = best_param_vals
    
    def set_param_names(self, param_names):
        self.param_names = param_names
        self.num_params = len(self.param_names)

    def remove_params_by_idx(self, param_idxs_to_remove):
        if len(param_idxs_to_remove) > 0:
            self.param_names = [self.param_names[II] for II in range(self.num_params) if II not in param_idxs_to_remove]
            self.num_params = len(self.param_names)
            if self.best_param_vals is not None:
                self.best_param_vals = np.delete(self.best_param_vals, param_idxs_to_remove)
            self.param_mins = np.delete(self.param_mins, param_idxs_to_remove)
            self.param_maxs = np.delete(self.param_maxs, param_idxs_to_remove)
            self.param_norm_obj = Normalise_class(self.param_mins, self.param_maxs)
            self.param_init = None

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
            print('WARNING bayesian will be deprecated and is untested')
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
                    # if self.DEBUG:
                    #     mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    #     print(f'rank={rank} memory={mem}')

                    # TODO save results here every few iterations


            if rank == 0:
                print(res)
                self.best_cost = res.fun
                self.best_param_vals = res.x
                np.save(os.path.join(self.output_dir, 'best_cost'), self.best_cost)
                np.save(os.path.join(self.output_dir, 'best_param_vals'), self.best_param_vals)

        elif self.param_id_method == 'genetic_algorithm':
            if self.DEBUG:
                num_elite = 1
                num_survivors = 2
                num_mutations_per_survivor = 2
                num_cross_breed = 0
            else:
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
            cost[0] = np.inf

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
                    # pop_per_proc = np.array([ave + 1 if p < res else ave for p in range(num_procs)])
                    # IMPORTANT: the above type of list comprehension breaks opencor if the opencor object
                    # has already been called in another function
                    pop_per_proc = np.zeros(num_procs, dtype=int)
                    for II in range(num_procs):
                        if II < res:
                            pop_per_proc[II] = ave + 1
                        else:
                            pop_per_proc[II] = ave
                    
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
                            if self.output_temp_results:
                                obs, temp_obs = self.sim_helper.get_results(self.obs_names, output_temp_results=True)
                            else:
                                obs = self.sim_helper.get_results(self.obs_names)
                                temp_obs = None
                            # calculate error between the observables of this set of parameters
                            # and the ground truth
                            cost_proc[II] = self.get_cost_from_obs(obs, temp_obs=temp_obs)

                            # reset params
                            self.sim_helper.reset_and_clear()
                            bools_proc[II] = True # this point has now been simulated

                        else:
                            # simulation failed, choose a new random point
                            print('simulation failed with params...')
                            print(param_vals_proc[:, II])
                            print('... choosing a new random point')
                            param_vals_proc[:, II:II + 1] = self.param_norm_obj.unnormalise(np.random.rand(self.num_params, 1))
                            cost_proc[II] = np.inf
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
                best_cost_in_array = np.array([self.best_cost])
                self.best_param_vals = param_vals[:, 0]
            else:
                best_cost_in_array = np.empty(1, dtype=float)
                self.best_param_vals = np.empty(self.num_params, dtype=float)

            comm.Bcast(best_cost_in_array, root=0)
            self.best_cost = best_cost_in_array[0]
            comm.Bcast(self.best_param_vals, root=0)



        else:
            print(f'param_id_method {self.param_id_method} hasn\'t been implemented')
            exit()

        if rank == 0:
            print('')
            print(f'{self.param_id_method} is complete')
            # print init params and final params
            print('init params     : {}'.format(self.param_init))
            print('best fit params : {}'.format(self.best_param_vals))
            print('best cost       : {}'.format(self.best_cost))

        return

    def run_single_sensitivity(self, sensitivity_output_path, do_triples_and_quads):
        # TODO this may need to be cleaned up or removed
        #  Doesn't work with frequency based obs
        if sensitivity_output_path == None:
            output_path = self.output_dir
        else:
            output_path = sensitivity_output_path

        self.best_param_vals = np.load(os.path.join(output_path, 'best_param_vals.npy'))

        gt_scalefactor = []
        const_idx = 0
        series_idx = 0

        for obs_idx in range(self.num_obs):
            if self.obs_types[obs_idx] != "series":
                #part of scale factor for normalising jacobain
                gt_scalefactor.append(self.weight_const_vec[const_idx]/self.std_const_vec[const_idx])
                # gt_scalefactor.append(1/self.ground_truth_const[x_idx])
                const_idx = const_idx + 1
            else: 
                #part of scale factor for normalising jacobain
                gt_scalefactor.append(self.weight_series_vec[series_idx]/self.std_const_vec[series_idx])
                # gt_scalefactor.append(1/self.ground_truth_const[x_idx])
                series_idx = series_idx + 1


        jacobian_sensitivity = np.zeros((self.num_params,self.num_obs))
        if self.pred_var_names == None:
            num_preds = 0
        else:
            num_preds = len(self.pred_var_names)*3 # *3 for the min max and mean of the pred trace
            pred_jacobian_sensitivity = np.zeros((self.num_params, num_preds))

        for i in range(self.num_params):
            #central difference calculation of derivative
            param_vec_up = self.best_param_vals.copy()
            param_vec_down = self.best_param_vals.copy()
            param_vec_up[i] = param_vec_up[i]*1.1
            param_vec_down[i] = param_vec_down[i]*0.9
            
            self.sim_helper.set_param_vals(self.param_names, param_vec_up)
            success = self.sim_helper.run()
            if success:
                up_obs = self.sim_helper.get_results(self.obs_names)
                if num_preds > 0:
                    up_preds = self.sim_helper.get_results(self.pred_var_names)
                self.sim_helper.reset_and_clear()
            else:
                print('sim failed on sensitivity run, reseting to new param_vec_up')
                while not success:
                    # keep slightly increasing param_vec_up until simulation runs
                    param_vec_up[i] = param_vec_up[i]*1.01
                    self.sim_helper.set_param_vals(self.param_names, param_vec_up)
                    success = self.sim_helper.run()
                up_obs = self.sim_helper.get_results(self.obs_names)
                if num_preds > 0:
                    up_preds = self.sim_helper.get_results(self.pred_var_names)
                self.sim_helper.reset_and_clear()

            self.sim_helper.set_param_vals(self.param_names, param_vec_down)
            success = self.sim_helper.run()
            if success:
                down_obs = self.sim_helper.get_results(self.obs_names)
                if num_preds > 0:
                    down_preds = self.sim_helper.get_results(self.pred_var_names)
                self.sim_helper.reset_and_clear()
            else:
                print('sim failed on sensitivity run, reseting to new param_vec_down')
                while not success:
                    # keep slightly decreasing param_vec_down until simulation runs
                    param_vec_up[i] = param_vec_down[i]*0.99
                    self.sim_helper.set_param_vals(self.param_names, param_vec_down)
                    success = self.sim_helper.run()
                down_obs = self.sim_helper.get_results(self.obs_names)
                if num_preds > 0:
                    down_preds = self.sim_helper.get_results(self.pred_var_names)
                self.sim_helper.reset_and_clear()

            print('sensitivity analysis needs to be updated to new version. Exiting')
            exit()
            up_obs_const_vec, up_obs_series_array = self.get_obs_vec_and_array(up_obs)
            down_obs_const_vec, down_obs_series_array = self.get_obs_vec_and_array(down_obs)
            for j in range(len(up_obs_const_vec)+len(up_obs_series_array)):
                dObs_param = 0
                #normalise derivative
                if j < len(up_obs_const_vec):
                    dObs_param = (up_obs_const_vec[j]-down_obs_const_vec[j])/(param_vec_up[i]-param_vec_down[i])
                    dObs_param = dObs_param*self.best_param_vals[i]*gt_scalefactor[j]
                else:
                    dObs_param = 0
                jacobian_sensitivity[i, j] = dObs_param

            if num_preds > 0:
                up_preds_const_vec = self.get_preds_min_max_mean(up_preds)
                down_preds_const_vec = self.get_preds_min_max_mean(down_preds)
                for j in range(num_preds):
                    dPreds_param = 0
                    #normalise derivative
                    dPreds_param = (up_preds_const_vec[j]-down_preds_const_vec[j])/(param_vec_up[i]-param_vec_down[i])
                    # TODO could I normalise the below better? an estimated std maybe?
                    dPreds_param = dPreds_param*self.best_param_vals[i]/up_preds_const_vec[j]
                    if dPreds_param == 0:
                        # avoid nan errors if the param has no effect on the prediction
                        dPreds_param = 1e-14
                    pred_jacobian_sensitivity[i, j] = dPreds_param

        np.save(os.path.join(output_path, 'normalised_jacobian_matrix.npy'), jacobian_sensitivity)
        if num_preds > 0:
            np.save(os.path.join(output_path, 'normalised_prediction_jacobian_matrix.npy'), pred_jacobian_sensitivity)

        #calculate parameter importance
        self.param_importance = np.zeros(self.num_params)
        if num_preds > 0:
            self.pred_param_importance = np.zeros(self.num_params)
        else:
            self.pred_param_importance = None
        for param_idx in range(self.num_params):
            sensitivity = 0
            pred_sensitivity = 0
            for obj_idx in range(self.num_obs):
                sensitivity += jacobian_sensitivity[param_idx][obj_idx] \
                              * jacobian_sensitivity[param_idx][obj_idx]
            sensitivity = math.sqrt(sensitivity / self.num_obs)
            self.param_importance[param_idx] = sensitivity

            if num_preds > 0:
                for pred_idx in range(num_preds):
                    pred_sensitivity += pred_jacobian_sensitivity[param_idx][pred_idx] \
                                       * pred_jacobian_sensitivity[param_idx][pred_idx]
                pred_sensitivity = math.sqrt(pred_sensitivity / num_preds)
                self.pred_param_importance[param_idx] = pred_sensitivity

        np.save(os.path.join(output_path, 'parameter_importance.npy'), self.param_importance)
        if num_preds > 0:
            np.save(os.path.join(output_path, 'parameter_importance_for_prediction.npy'),
                self.pred_param_importance)

        #calculate S-norm
        S_norm = np.zeros((self.num_params,self.num_obs))
        pred_S_norm = np.zeros((self.num_params, num_preds))
        for param_idx in range(self.num_params):
            for objs_idx in range(self.num_obs):
                S_norm[param_idx][objs_idx] = jacobian_sensitivity[param_idx][objs_idx]/\
                                             (self.param_importance[param_idx]*math.sqrt(self.num_obs))
            if num_preds > 0:
                for preds_idx in range(num_preds):
                    pred_S_norm[param_idx][preds_idx] = pred_jacobian_sensitivity[param_idx][preds_idx]/ \
                                                  (self.pred_param_importance[param_idx]*math.sqrt(num_preds))


        collinearity_eigvals = []
        pred_collinearity_eigvals = []
        for i in range(self.num_params):
            Sl = S_norm[:(i+1),:]
            Sll = Sl@Sl.T
            eigvals, eigvecs = la.eig(Sll)
            real_eigvals = eigvals.real
            collinearity_eigvals.append(min(real_eigvals))

        #calculate collinearity
        self.collinearity_idx = np.zeros(len(collinearity_eigvals))
        for i in range(len(collinearity_eigvals)):
            if collinearity_eigvals[i] < 1e-12:
                self.collinearity_idx[i] = 1e6
            else:
                self.collinearity_idx[i] = 1/math.sqrt(collinearity_eigvals[i])

        np.save(os.path.join(output_path, 'collinearity_idx.npy'), self.collinearity_idx)


        self.collinearity_idx_pairs = np.zeros((self.num_params,self.num_params))
        if num_preds > 0:
            self.pred_collinearity_idx_pairs = np.zeros((self.num_params,self.num_params))
        else:
            self.pred_collinearity_idx_pairs = None
        for i in range(self.num_params):
            for j in range(self.num_params):
                if i!=j:
                    Sl = S_norm[[i,j],:]
                    Sll = Sl@Sl.T
                    eigvals_pairs, eigvecs_pairs = la.eig(Sll)
                    real_eigvals_pairs = eigvals_pairs.real
                    self.collinearity_idx_pairs[i][j] = 1/math.sqrt(max(min(real_eigvals_pairs), 1e-12))

                    if num_preds > 0:
                        pred_Sl = pred_S_norm[[i,j],:]
                        pred_Sll = pred_Sl@pred_Sl.T
                        pred_eigvals_pairs, pred_eigvecs_pairs = la.eig(pred_Sll)
                        pred_real_eigvals_pairs = pred_eigvals_pairs.real
                        self.pred_collinearity_idx_pairs[i][j] = 1/math.sqrt(max(min(pred_real_eigvals_pairs), 1e-12))
                else:
                    self.collinearity_idx_pairs[i][j] = 0
                    if num_preds > 0:
                        self.pred_collinearity_idx_pairs[i][j] = 0

        np.save(os.path.join(output_path, 'collinearity_pairs.npy'), self.collinearity_idx_pairs)
        if num_preds > 0:
            np.save(os.path.join(output_path, 'pred_collinearity_pairs.npy'), self.pred_collinearity_idx_pairs)

        if do_triples_and_quads:
            collinearity_idx_triple = np.zeros((self.num_params, self.num_params))
            for i in range(self.num_params):
                for j in range(self.num_params):
                    for k in range(self.num_params):
                        if ((i!=j) and (i!=k) and (j!=k)):
                            Sl = S_norm[[i,j,k],:]
                            Sll = Sl@Sl.T
                            eigvals_pairs, eigvecs_pairs = la.eig(Sll)
                            real_eigvals_pairs = eigvals_pairs.real
                            collinearity_idx_triple[j][k] = 1/math.sqrt(max(min(real_eigvals_pairs), 1e-12))
                        else:
                            collinearity_idx_triple[j][k] = 0
                np.save(os.path.join(output_path, 'collinearity_triples'+str(i)+'.npy'), collinearity_idx_triple)

            collinearity_idx_quad = np.zeros((self.num_params, self.num_params))
            for i in range(self.num_params):
                for j in range(self.num_params):
                    for k in range(self.num_params):
                        for l in range(self.num_params):
                            if ((i!=j) and (i!=k) and (i!=l) and (j!=k) and (j!=l) and (k!=l)):
                                Sl = S_norm[[i,j,k,l],:]
                                Sll = Sl@Sl.T
                                eigvals_pairs, eigvecs_pairs = la.eig(Sll)
                                real_eigvals_pairs = eigvals_pairs.real
                                collinearity_idx_quad[k][l] = 1/math.sqrt(max(min(real_eigvals_pairs), 1e-12))
                        else:
                            collinearity_idx_quad[k][l] = 0
                    np.save(os.path.join(output_path, 'collinearity_quads'+str(i)+'_'+str(j)+'.npy'), collinearity_idx_quad)

        return

    def get_cost_from_params(self, param_vals, reset=True, param_vals_are_normalised=False):

        # set params for this case
        if param_vals_are_normalised:
            param_vals = self.param_norm_obj.unnormalise(param_vals)

        self.sim_helper.set_param_vals(self.param_names, param_vals)
        
        success = self.sim_helper.run()
        if success:
            if self.output_temp_results:
                obs, temp_obs = self.sim_helper.get_results(self.obs_names, output_temp_results=True)
            else:
                obs = self.sim_helper.get_results(self.obs_names)
                temp_obs = None

            cost = self.get_cost_from_obs(obs, temp_obs=temp_obs)

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
            if self.output_temp_results:
                obs, temp_obs = self.sim_helper.get_results(self.obs_names, output_temp_results=True)
            else:
                obs = self.sim_helper.get_results(self.obs_names)
                temp_obs = None

            cost = self.get_cost_from_obs(obs, temp_obs=temp_obs)

            # reset params
            if reset:
                self.sim_helper.reset_and_clear()

        else:
            # simulation set cost to large,
            print('simulation failed with params...')
            print(param_vals)
            cost = np.inf

        return cost, obs, temp_obs

    def get_cost_from_obs(self, obs, temp_obs=None):

        obs_dict = self.get_obs_vec_and_array(obs, temp_obs=temp_obs)
        # calculate error between the observables of this set of parameters
        # and the ground truth
        
        cost = self.cost_calc(obs_dict)

        return cost

    def cost_calc(self, obs_dict):
        # cost = np.sum(np.power(self.weight_const_vec*(const -
        #                        self.ground_truth_const)/np.minimum(const,
        #                                                             self.ground_truth_const), 2))/(self.num_obs)
        const = obs_dict['const']
        series = obs_dict['series']
        amp = obs_dict['amp']
        phase = obs_dict['phase']
        if len(self.ground_truth_phase) == 0:
            phase = None
        if self.ground_truth_phase.all() == None:
            phase = None
        if self.cost_type == 'MSE':
            cost = np.sum(np.power(self.weight_const_vec*(const -
                               self.ground_truth_const)/self.std_const_vec, 2))
        elif self.cost_type == 'AE':
            cost = np.sum(np.abs(self.weight_const_vec*(const -
                                                          self.ground_truth_const)/self.std_const_vec))
        else:
            print(f'cost type of {self.cost_type} not implemented')
            exit()

        if series is not None:
            #print(series)
            min_len_series = min(self.ground_truth_series.shape[1], series.shape[1])
            # calculate sum of squares cost and divide by number data points in series data
            # divide by number data points in series data
            if self.cost_type == 'MSE':
                series_cost = np.sum(np.power((series[:, :min_len_series] -
                                               self.ground_truth_series[:,
                                               :min_len_series]) * self.weight_series_vec.reshape(-1, 1) /
                                              self.std_series_vec.reshape(-1, 1), 2)) / min_len_series
            elif self.cost_type == 'AE':
                series_cost = np.sum(np.abs((series[:, :min_len_series] -
                                             self.ground_truth_series[:,
                                             :min_len_series]) * self.weight_series_vec.reshape(-1, 1) /
                                            self.std_series_vec.reshape(-1, 1))) / min_len_series
        else:
            series_cost = 0

        if amp is not None:
            # calculate sum of squares cost and divide by number data points in freq data
            # divide by number data points in series data
            if self.cost_type == 'MSE':
                amp_cost = np.sum([np.power((amp[JJ] - self.ground_truth_amp[JJ]) *
                                             self.weight_amp_vec[JJ] /
                                             self.std_amp_vec[JJ], 2) / len(amp[JJ]) for JJ in range(len(amp))])
            elif self.cost_type == 'AE':
                amp_cost = np.sum([np.abs((amp[JJ] - self.ground_truth_amp[JJ]) *
                                             self.weight_amp_vec[JJ] /
                                             self.std_amp_vec[JJ]) / len(amp[JJ]) for JJ in range(len(amp))])
        else:
            amp_cost = 0

        if phase is not None:
            # calculate sum of squares cost and divide by number data points in freq data
            # divide by number data points in series data
            # TODO figure out how to properly weight this compared to the frequency weight.
            if self.cost_type == 'MSE':
                phase_cost = np.sum([np.power((phase[JJ] - self.ground_truth_phase[JJ]) *
                                             self.weight_phase_vec[JJ], 2) / len(phase[JJ]) for JJ in
                                    range(len(phase))])
            if self.cost_type == 'AE':
                phase_cost = np.sum([np.abs((phase[JJ] - self.ground_truth_phase[JJ]) *
                                              self.weight_phase_vec[JJ]) / len(phase[JJ]) for JJ in
                                     range(len(phase))])
        else:
            phase_cost = 0

        cost = (cost + series_cost + amp_cost + phase_cost) / self.num_obs

        return cost

    def get_obs_vec_and_array(self, obs, temp_obs=None):

        obs_const_vec = np.zeros((len(self.ground_truth_const), ))
        obs_series_array = np.zeros((len(self.ground_truth_series), self.n_steps + 1))
        # TODO series array should also be a list of arrays for if the series are of variable lengths
        obs_amp_list_of_arrays = [np.zeros(len(self.obs_freqs[JJ])) for JJ in range(len(obs))
                                   if self.obs_types[JJ] == 'frequency']
        obs_phase_list_of_arrays = [np.zeros(len(self.obs_freqs[JJ])) for JJ in range(len(obs))
                                  if self.obs_types[JJ] == 'frequency']

        const_count = 0
        series_count = 0
        freq_count = 0
        for JJ in range(len(obs)):
            if self.obs_types[JJ] == 'mean':
                obs_const_vec[const_count] = np.mean(obs[JJ, :])
                const_count += 1
            elif self.obs_types[JJ] == 'max':
                obs_const_vec[const_count] = np.max(obs[JJ, :])
                const_count += 1
            elif self.obs_types[JJ] == 'min':
                obs_const_vec[const_count] = np.min(obs[JJ, :])
                const_count += 1
            elif self.obs_types[JJ] == 'series':
                obs_series_array[series_count, :] = obs[JJ, :]
                series_count += 1
            elif self.obs_types[JJ] == 'frequency':
                # TODO copy this to mcmc
                if self.obs_operations[JJ] == None:
                    time_domain_obs = obs[JJ, :]

                    complex_num = np.fft.fft(time_domain_obs)/len(time_domain_obs)
                    amp = np.abs(complex_num)[0:len(time_domain_obs)//2]
                    # make sure the first amplitude is negative if it is a negative signal
                    amp[0] = amp[0] * np.sign(np.mean(time_domain_obs))
                    phase = np.angle(complex_num)[0:len(time_domain_obs)//2]
                    freqs = np.fft.fftfreq(time_domain_obs.shape[-1], d=self.dt)[:len(time_domain_obs)//2]
                else:
                    time_domain_obs_0 = temp_obs[JJ, 0, :]
                    time_domain_obs_1 = temp_obs[JJ, 1, :]

                    complex_num_0 = np.fft.fft(time_domain_obs_0)/len(time_domain_obs_0)
                    complex_num_1 = np.fft.fft(time_domain_obs_1)/len(time_domain_obs_1)

                    if (self.obs_operations[JJ] == 'multiplication' and temp_obs is not None):
                        complex_num = complex_num_0 * complex_num_1
                        sign_signal = np.sign(np.mean(time_domain_obs_0) * np.mean(time_domain_obs_1))
                    elif (self.obs_operations[JJ] == 'division' and temp_obs is not None):
                        complex_num = complex_num_0 / complex_num_1
                        sign_signal = np.sign(np.mean(time_domain_obs_0) / np.mean(time_domain_obs_1))
                    elif (self.obs_operations[JJ] == 'addition' and temp_obs is not None):
                        complex_num = complex_num_0 + complex_num_1
                        sign_signal = np.sign(np.mean(time_domain_obs_0) + np.mean(time_domain_obs_1))
                    elif (self.obs_operations[JJ] == 'subtraction' and temp_obs is not None):
                        complex_num = complex_num_0 - complex_num_1
                        sign_signal = np.sign(np.mean(time_domain_obs_0) - np.mean(time_domain_obs_1))

                    amp = np.abs(complex_num)[0:len(time_domain_obs_0)//2]
                    # make sure the first amplitude is negative if it is a negative signal
                    amp[0] = amp[0] * sign_signal
                    phase = np.angle(complex_num)[0:len(time_domain_obs_0)//2]

                    freqs = np.fft.fftfreq(time_domain_obs_0.shape[-1], d=self.dt)[:len(time_domain_obs_0)//2]


                # now interpolate to defined frequencies
                obs_amp_list_of_arrays[freq_count][:] = np.interp(self.obs_freqs[JJ], freqs, amp)
                # and phase
                obs_phase_list_of_arrays[freq_count][:] = np.interp(self.obs_freqs[JJ], freqs, phase)

                freq_count += 1

        if series_count == 0:
            obs_series_array = None
        if freq_count == 0:
            obs_freq_list_of_arrays = None
            obs_phase_list_of_arrays = None
        obs_dict = {'const': obs_const_vec, 'series': obs_series_array,
                    'amp': obs_amp_list_of_arrays, 'phase': obs_phase_list_of_arrays}
        return obs_dict

    def get_preds_min_max_mean(self, preds):

        preds_const_vec = np.zeros((preds.shape[0]*3, ))
        for JJ in range(len(preds)):
            preds_const_vec[JJ] = np.min(preds[JJ, :])
            preds_const_vec[JJ + 1] = np.max(preds[JJ, :])
            preds_const_vec[JJ + 2] = np.mean(preds[JJ, :])
        return preds_const_vec

    def simulate_with_best_param_vals(self, use_mcmc_chain=False):
        if MPI.COMM_WORLD.Get_rank() != 0:
            print('simulate once should only be done on one rank')
            exit()

        if self.best_param_vals is None:
            self.best_param_vals = np.load(os.path.join(self.output_dir, 'best_param_vals.npy'))
        else:
            # The sim object has already been opened so the best cost doesn't need to be opened
            pass

        if not np.isfinite(self.best_cost):
            self.best_cost = np.load(os.path.join(self.output_dir, 'best_cost.npy'))

        # ___________ Run model with new parameters ________________

        self.sim_helper.update_times(self.dt, 0.0, self.sim_time, self.pre_time)

        # run simulation and check cost
        cost_check, obs, temp_obs = self.get_cost_and_obs_from_params(self.best_param_vals, reset=False)
        obs_dict = self.get_obs_vec_and_array(obs, temp_obs=temp_obs)

        print(f'cost should be {self.best_cost}')
        print('cost check after single simulation is {}'.format(cost_check))


        print(f'final obs values :')
        print(obs_dict['const'])
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
        cost_check, obs, temp_obs = self.get_cost_and_obs_from_params(self.best_param_vals, reset=False)
        obs_dict = self.get_obs_vec_and_array(obs)

        print(f'cost should be {self.best_cost}')
        print('cost check after single simulation is {}'.format(cost_check))

        print(f'final obs values :')
        print(obs_dict['const'])

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
                 obs_names, obs_types, obs_freqs, obs_operations, obs_operands,
                 weight_const_vec, weight_series_vec, 
                 weight_amp_vec, weight_phase_vec,
                 std_const_vec, std_series_vec, std_amp_vec,
                 param_names,
                 ground_truth_const, ground_truth_series, ground_truth_amp, ground_truth_phase,
                 param_mins, param_maxs, param_prior_types,
                 sim_time=2.0, pre_time=20.0, pre_heart_periods=None, sim_heart_periods=None,
                 dt=0.01, maximum_step=0.0001, mcmc_options=None, DEBUG=False):

        self.model_path = model_path
        self.output_dir = None

        self.obs_names = obs_names
        self.obs_types = obs_types
        self.obs_freqs = obs_freqs
        self.obs_operations = obs_operations
        self.obs_operands = obs_operands
        self.weight_const_vec = weight_const_vec
        self.weight_series_vec = weight_series_vec
        self.weight_amp_vec = weight_amp_vec
        self.weight_phase_vec = weight_phase_vec
        self.std_const_vec = std_const_vec
        self.std_series_vec = std_series_vec
        self.std_amp_vec = std_amp_vec
        self.param_names = param_names
        self.num_obs = len(self.obs_names)
        self.num_params = len(self.param_names)
        self.ground_truth_const = ground_truth_const
        self.ground_truth_series = ground_truth_series
        self.ground_truth_amp = ground_truth_amp
        self.ground_truth_phase = ground_truth_phase
        self.param_mins = param_mins
        self.param_maxs = param_maxs
        self.param_prior_types = param_prior_types
        self.param_norm_obj = Normalise_class(self.param_mins, self.param_maxs)
        
        for type, operation in zip(self.obs_types, self.obs_operations):
            if type == 'frequency' and operation != None:
                print('have not implemented frequency with operations in mcmc yet. EXITING')
                exit()

        # set up opencor simulation
        self.dt = dt  # TODO this could be optimised
        self.maximum_step = maximum_step
        self.point_interval = self.dt
        if sim_time is not None:
            self.sim_time = sim_time
        else:
            # set temporary sim time, just to initialise the sim_helper
            self.sim_time = 0.001
        if pre_time is not None:
            self.pre_time = pre_time
        else:
            # set temporary pre time, just to initialise the sim_helper
            self.pre_time = 0.001
        self.sim_helper = self.initialise_sim_helper()

        if pre_heart_periods is not None:
            T = self.sim_helper.get_init_param_vals(['heart/T'])[0]
            self.pre_time = T*pre_heart_periods
        if sim_heart_periods is not None:
            T = self.sim_helper.get_init_param_vals(['heart/T'])[0]
            self.sim_time = T*sim_heart_periods

        self.n_steps = int(self.sim_time/self.dt)

        self.sim_helper.update_times(self.dt, 0.0, self.sim_time, self.pre_time)
        self.sim_helper.create_operation_variables(self.obs_names, self.obs_operations, self.obs_operands)

        # initialise
        self.param_init = None
        self.best_param_vals = None
        self.best_cost = np.inf

        # mcmc
        self.sampler = None
        if mcmc_options is not None:
            self.num_steps = mcmc_options['num_steps']
            self.num_walkers = mcmc_options['num_walkers']
            self.cost_type = mcmc_options['cost_type']
        else:
            self.num_steps = 5000
            self.num_walkers = 2*self.num_params
            self.cost_type = 'MSE'
            print('number of mcmc steps and walkers is not set, choosing defaults of 5000 and 2*num_params')

        self.DEBUG = DEBUG

    def initialise_sim_helper(self):
        return SimulationHelper(self.model_path, self.dt, self.sim_time,
                                maximumNumberofSteps=100000000,
                                maximum_step=self.maximum_step, pre_time=self.pre_time)

    def set_best_param_vals(self, best_param_vals):
        self.best_param_vals = best_param_vals

    def set_param_names(self, param_names):
        self.param_names = param_names
        self.num_params = len(self.param_names)

    def remove_params_by_idx(self, param_idxs_to_remove):
        if len(param_idxs_to_remove) > 0:
            self.param_names = [self.param_names[II] for II in range(self.num_params) if II not in param_idxs_to_remove]
            self.num_params = len(self.param_names)
            if self.best_param_vals is not None:
                self.best_param_vals = np.delete(self.best_param_vals, param_idxs_to_remove)
            self.param_mins = np.delete(self.param_mins, param_idxs_to_remove)
            self.param_maxs = np.delete(self.param_maxs, param_idxs_to_remove)
            self.param_prior_types = np.delete(self.param_prior_types, param_idxs_to_remove)
            self.param_norm_obj = Normalise_class(self.param_mins, self.param_maxs)
            self.param_init = None

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

            if rank == 0:
                if self.best_param_vals is not None:
                    best_param_vals_norm = self.param_norm_obj.normalise(self.best_param_vals)
                    # create initial params in gaussian ball around best_param_vals estimate
                    init_param_vals_norm = (np.ones((self.num_walkers, self.num_params))*best_param_vals_norm).T + \
                                       0.1*np.random.randn(self.num_params, self.num_walkers)
                    init_param_vals_norm = np.clip(init_param_vals_norm, 0.001, 0.999)
                    init_param_vals = self.param_norm_obj.unnormalise(init_param_vals_norm)
                else:
                    init_param_vals_norm = np.random.rand(self.num_params, self.num_walkers)
                    init_param_vals = self.param_norm_obj.unnormalise(init_param_vals_norm)

            try:
                pool = MPIPool() # workers dont get past this line in this try, they wait for work to do
                if mcmc_lib == 'emcee':
                    self.sampler = emcee.EnsembleSampler(self.num_walkers, self.num_params, calculate_lnlikelihood,
                                                pool=pool)
                elif mcmc_lib == 'zeus':
                    self.sampler = zeus.EnsembleSampler(self.num_walkers, self.num_params, calculate_lnlikelihood,
                                                         pool=pool)

                start_time = time.time()
                self.sampler.run_mcmc(init_param_vals.T, self.num_steps, progress=True, tune=True)
                print(f'mcmc time = {time.time() - start_time}')
            except:
                if rank == 0:
                    sys.exit()
                else:
                    # workers pass to here
                    pass

        else:
            if self.best_param_vals is not None:
                best_param_vals_norm = self.param_norm_obj.normalise(self.best_param_vals)
                init_param_vals_norm = (np.ones((self.num_walkers, self.num_params))*best_param_vals_norm).T + \
                                   0.01*np.random.randn(self.num_params, self.num_walkers)
                init_param_vals_norm = np.clip(init_param_vals_norm, 0.001, 0.999)
                init_param_vals = self.param_norm_obj.unnormalise(init_param_vals_norm)
            else:
                init_param_vals_norm = np.random.rand(self.num_params, self.num_walkers)
                init_param_vals = self.param_norm_obj.unnormalise(init_param_vals_norm)

            if mcmc_lib == 'emcee':
                self.sampler = emcee.EnsembleSampler(self.num_walkers, self.num_params, calculate_lnlikelihood)
            elif mcmc_lib == 'zeus':
                self.sampler = zeus.EnsembleSampler(self.num_walkers, self.num_params, calculate_lnlikelihood)

            start_time = time.time()
            self.sampler.run_mcmc(init_param_vals.T, self.num_steps) # , progress=True)
            print(f'mcmc time = {time.time()-start_time}')

        if rank == 0:
            # TODO save chains
            if mcmc_lib == 'emcee':
                print(f'acceptance fraction was {self.sampler.acceptance_fraction}')
            samples = self.sampler.get_chain()
            mcmc_chain_path = os.path.join(self.output_dir, 'mcmc_chain.npy')
            np.save(mcmc_chain_path, samples)
            print('mcmc complete')
            print(f'mcmc chain saved in {mcmc_chain_path}')

            # save best param vals and best cost from mcmc mean
            samples = samples[samples.shape[0]//2:, :, :]
            # thin = 10
            # samples = samples[::thin, :, :]
            flat_samples = samples.reshape(-1, self.num_params)
            means = np.zeros((self.num_params))
            medians = np.zeros((self.num_params))
            for param_idx in range(self.num_params):
                means[param_idx] = np.mean(flat_samples[:, param_idx])
                medians[param_idx] = np.median(flat_samples[:, param_idx])

            # rerun with original and mcmc optimal param vals
            mcmc_best_param_vals = medians  # means
            mcmc_best_cost, obs, temp_obs = self.get_cost_and_obs_from_params(mcmc_best_param_vals, reset=True)
            if self.best_param_vals is None:
                self.best_param_vals = mcmc_best_param_vals
                self.best_cost = mcmc_best_cost
                print('cost from mcmc median param vals is {}'.format(self.best_cost))
                print('saving best_param_vals and best_cost from mcmc medians')

                np.save(os.path.join(self.output_dir, 'best_cost'), self.best_cost)
                np.save(os.path.join(self.output_dir, 'best_param_vals'), self.best_param_vals)
            else:
                original_best_cost, obs, temp_obs = self.get_cost_and_obs_from_params(self.best_param_vals, reset=True)
                if mcmc_best_cost < original_best_cost:
                    self.best_param_vals = mcmc_best_param_vals
                    self.best_cost = mcmc_best_cost
                    print('cost from mcmc median param vals is {}'.format(self.best_cost))
                    print('resaving best_param_vals and best_cost from mcmc medians')

                    np.save(os.path.join(self.output_dir, 'best_cost'), self.best_cost)
                    np.save(os.path.join(self.output_dir, 'best_param_vals'), self.best_param_vals)
                else:
                    self.best_cost = original_best_cost
                    # leave the original best fit param val as the best fit value, mcmc just gives distributions
                    print('cost from mcmc median param vals is {}'.format(mcmc_best_cost))
                    print('Keeping the genetic algorithm best fit as it is lower, ({})'.format(self.best_cost))

    def get_lnprior_from_params(self, param_vals):
        lnprior = 0
        for idx, param_val in enumerate(param_vals):
            if self.param_prior_types is not None:
                prior_dist = self.param_prior_types[idx]
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
                if param_val < self.param_mins[idx] or param_val > self.param_maxs[idx]:
                    return -np.inf
                else:
                    # the normalisation isnt needed here but might be nice to
                    # make sure prior for each param is between 0 and 1
                    lnprior += -lamb*param_val/self.param_maxs[idx]

            elif prior_dist == 'normal':
                if param_val < self.param_mins[idx] or param_val > self.param_maxs[idx]:
                    return -np.inf
                else:
                    # temporarily make the std 1/6 of the user defined range and the mean the centre of the range
                    std = 1/6*(self.param_maxs[idx] - self.param_mins[idx])
                    mean = 0.5*(self.param_maxs[idx] + self.param_mins[idx])
                    lnprior += -0.5*((param_val - mean)/std)**2


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
            obs = self.sim_helper.get_results(self.obs_names)

            lnlikelihood = self.get_lnlikelihood_from_obs(obs)

            # reset params
            if reset:
                self.sim_helper.reset_and_clear()

        else:
            # simulation set cost to large,
            print('simulation failed with params...')
            print(param_vals)
            return -np.inf

        return lnprior + lnlikelihood


    def get_lnlikelihood_from_obs(self, obs):

        # calculate error between the observables of this set of parameters
        # and the ground truth
        cost = self.get_cost_from_obs(obs)
        lnlikelihood = -0.5*cost

        return lnlikelihood

    def get_cost_and_obs_from_params(self, param_vals, reset=True):
        # set params for this case
        self.sim_helper.set_param_vals(self.param_names, param_vals)

        success = self.sim_helper.run()
        if success:
            obs = self.sim_helper.get_results(self.obs_names)

            cost = self.get_cost_from_obs(obs)

            # reset params
            if reset:
                self.sim_helper.reset_and_clear()

        else:
            # simulation set cost to large,
            print('simulation failed with params...')
            print(param_vals)
            cost = np.inf

        temp_obs = None # TODO implement operands with frequency domain in MCMC
        return cost, obs, temp_obs

    def get_cost_from_obs(self, obs):

        obs_dict = self.get_obs_vec_and_array(obs)
        # calculate error between the observables of this set of parameters
        # and the ground truth
        cost = self.cost_calc(obs_dict)

        return cost

    def cost_calc(self, obs_dict):
        # cost = np.sum(np.power(self.weight_const_vec*(const -
        #                        self.ground_truth_const)/np.minimum(const,
        #                                                             self.ground_truth_const), 2))/(self.num_obs)
        const = obs_dict['const']
        series = obs_dict['series']
        amp = obs_dict['amp']
        phase = obs_dict['phase']
        if len(self.ground_truth_phase) == 0:
            phase = None
        if self.ground_truth_phase.all() == None:
            phase = None
        if self.cost_type == 'MSE':
            cost = np.sum(np.power(self.weight_const_vec*(const -
                               self.ground_truth_const)/self.std_const_vec, 2))
        elif self.cost_type == 'AE':
            cost = np.sum(np.abs(self.weight_const_vec*(const -
                                                          self.ground_truth_const)/self.std_const_vec))
        else:
            print(f'cost type of {self.cost_type} not implemented')
            exit()

        if series is not None:
            #print(series)
            min_len_series = min(self.ground_truth_series.shape[1], series.shape[1])
            # calculate sum of squares cost and divide by number data points in series data
            # divide by number data points in series data
            if self.cost_type == 'MSE':
                series_cost = np.sum(np.power((series[:, :min_len_series] -
                                               self.ground_truth_series[:,
                                               :min_len_series]) * self.weight_series_vec.reshape(-1, 1) /
                                              self.std_series_vec.reshape(-1, 1), 2)) / min_len_series
            elif self.cost_type == 'AE':
                series_cost = np.sum(np.abs((series[:, :min_len_series] -
                                             self.ground_truth_series[:,
                                             :min_len_series]) * self.weight_series_vec.reshape(-1, 1) /
                                            self.std_series_vec.reshape(-1, 1))) / min_len_series
        else:
            series_cost = 0

        if amp is not None:
            # calculate sum of squares cost and divide by number data points in freq data
            # divide by number data points in series data
            if self.cost_type == 'MSE':
                amp_cost = np.sum([np.power((amp[JJ] - self.ground_truth_amp[JJ]) *
                                             self.weight_amp_vec[JJ] /
                                             self.std_amp_vec[JJ], 2) / len(amp[JJ]) for JJ in range(len(amp))])
            elif self.cost_type == 'AE':
                amp_cost = np.sum([np.abs((amp[JJ] - self.ground_truth_amp[JJ]) *
                                             self.weight_amp_vec[JJ] /
                                             self.std_amp_vec[JJ]) / len(amp[JJ]) for JJ in range(len(amp))])
        else:
            amp_cost = 0

        if phase is not None:
            # calculate sum of squares cost and divide by number data points in freq data
            # divide by number data points in series data
            # TODO figure out how to properly weight this compared to the frequency weight.
            if self.cost_type == 'MSE':
                phase_cost = np.sum([np.power((phase[JJ] - self.ground_truth_phase[JJ]) *
                                             self.weight_phase_vec[JJ], 2) / len(phase[JJ]) for JJ in
                                    range(len(phase))])
            if self.cost_type == 'AE':
                phase_cost = np.sum([np.abs((phase[JJ] - self.ground_truth_phase[JJ]) *
                                              self.weight_phase_vec[JJ]) / len(phase[JJ]) for JJ in
                                     range(len(phase))])
        else:
            phase_cost = 0

        cost = (cost + series_cost + amp_cost + phase_cost) / self.num_obs

        return cost

    def get_obs_vec_and_array(self, obs, temp_obs=None):

        obs_const_vec = np.zeros((len(self.ground_truth_const), ))
        obs_series_array = np.zeros((len(self.ground_truth_series), self.n_steps + 1))
        # TODO series array should also be a list of arrays for if the series are of variable lengths
        obs_amp_list_of_arrays = [np.zeros(len(self.obs_freqs[JJ])) for JJ in range(len(obs))
                                   if self.obs_types[JJ] == 'frequency']
        obs_phase_list_of_arrays = [np.zeros(len(self.obs_freqs[JJ])) for JJ in range(len(obs))
                                  if self.obs_types[JJ] == 'frequency']

        const_count = 0
        series_count = 0
        freq_count = 0
        for JJ in range(len(obs)):
            if self.obs_types[JJ] == 'mean':
                obs_const_vec[const_count] = np.mean(obs[JJ, :])
                const_count += 1
            elif self.obs_types[JJ] == 'max':
                obs_const_vec[const_count] = np.max(obs[JJ, :])
                const_count += 1
            elif self.obs_types[JJ] == 'min':
                obs_const_vec[const_count] = np.min(obs[JJ, :])
                const_count += 1
            elif self.obs_types[JJ] == 'series':
                obs_series_array[series_count, :] = obs[JJ, :]
                series_count += 1
            elif self.obs_types[JJ] == 'frequency':
                # TODO copy this to mcmc
                if self.obs_operations[JJ] == None:
                    time_domain_obs = obs[JJ, :]

                    complex_num = np.fft.fft(time_domain_obs)/len(time_domain_obs)
                    amp = np.abs(complex_num)[0:len(time_domain_obs)//2]
                    # make sure the first amplitude is negative if it is a negative signal
                    amp[0] = amp[0] * np.sign(np.mean(time_domain_obs))
                    phase = np.angle(complex_num)[0:len(time_domain_obs)//2]
                    freqs = np.fft.fftfreq(time_domain_obs.shape[-1], d=self.dt)[:len(time_domain_obs)//2]
                else:
                    time_domain_obs_0 = temp_obs[JJ, 0, :]
                    time_domain_obs_1 = temp_obs[JJ, 1, :]

                    complex_num_0 = np.fft.fft(time_domain_obs_0)/len(time_domain_obs_0)
                    complex_num_1 = np.fft.fft(time_domain_obs_1)/len(time_domain_obs_1)

                    if (self.obs_operations[JJ] == 'multiplication' and temp_obs is not None):
                        complex_num = complex_num_0 * complex_num_1
                        sign_signal = np.sign(np.mean(time_domain_obs_0) * np.mean(time_domain_obs_1))
                    elif (self.obs_operations[JJ] == 'division' and temp_obs is not None):
                        complex_num = complex_num_0 / complex_num_1
                        sign_signal = np.sign(np.mean(time_domain_obs_0) / np.mean(time_domain_obs_1))
                    elif (self.obs_operations[JJ] == 'addition' and temp_obs is not None):
                        complex_num = complex_num_0 + complex_num_1
                        sign_signal = np.sign(np.mean(time_domain_obs_0) + np.mean(time_domain_obs_1))
                    elif (self.obs_operations[JJ] == 'subtraction' and temp_obs is not None):
                        complex_num = complex_num_0 - complex_num_1
                        sign_signal = np.sign(np.mean(time_domain_obs_0) - np.mean(time_domain_obs_1))

                    amp = np.abs(complex_num)[0:len(time_domain_obs_0)//2]
                    # make sure the first amplitude is negative if it is a negative signal
                    amp[0] = amp[0] * sign_signal
                    phase = np.angle(complex_num)[0:len(time_domain_obs_0)//2]

                    freqs = np.fft.fftfreq(time_domain_obs_0.shape[-1], d=self.dt)[:len(time_domain_obs_0)//2]

                # now interpolate to defined frequencies
                obs_amp_list_of_arrays[freq_count][:] = np.interp(self.obs_freqs[JJ], freqs, amp)
                # and phase
                obs_phase_list_of_arrays[freq_count][:] = np.interp(self.obs_freqs[JJ], freqs, phase)

                freq_count += 1

        if series_count == 0:
            obs_series_array = None
        if freq_count == 0:
            obs_freq_list_of_arrays = None
            obs_phase_list_of_arrays = None
        obs_dict = {'const': obs_const_vec, 'series': obs_series_array,
                    'amp': obs_amp_list_of_arrays, 'phase': obs_phase_list_of_arrays}
        return obs_dict

    def set_output_dir(self, output_dir):
        self.output_dir = output_dir

    def calculate_var_from_posterior_samples(self, var_names, flat_samples, n_sims=100):
        var_array = np.zeros((len(var_names), n_sims, self.n_steps + 1))
        for II in range(n_sims):
            rand_idx = np.random.randint(0, len(flat_samples)-1)
            sample_param_vals = flat_samples[rand_idx, :]
            self.sim_helper.set_param_vals(self.param_names, sample_param_vals)
            success = self.sim_helper.run()
            if success:
                var_array[:, II, :] = self.sim_helper.get_results(var_names)
                self.sim_helper.reset_and_clear()
            else:
                print("sim_helper failed when running sample, this shouldn't happen, exiting")
                exit()

        return var_array


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

