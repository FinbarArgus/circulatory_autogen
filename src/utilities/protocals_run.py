import numpy as np
import os, sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from opencor_helper import SimulationHelper
import paperPlotSetup
paperPlotSetup.Setup_Plot(3)


# USER SHOULD ONLY NEED TO CHANGE BETWEEN HERE
##########################################################

model_path = "/home/farg967/Documents/git_projects/circulatory_autogen/generated_models/SN_to_cAMP/SN_to_cAMP.cellml"
plot_dir = "/home/farg967/Documents/git_projects/circulatory_autogen/generated_models/SN_to_cAMP/plots"
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

dt = 0.0001
# the times for each section of specific param values in each experiment. i.e, experiment 1
# has the first set of param values for 1 second, then the second set for 2 seconds.
sim_times = [[1,2], [1,2],
             [1,2], [1,2],
             [10, 60, 180, 60, 60],
             [4, 4, 4]]
params_to_change_dict = {'SN/I_const': [[0, -0.15],   [0, -0.15],  
                                        [0, -0.15],   [0, -0.15],
                                        [-0.15, -0.15, -0.15, -0.15, -0.15],
                                        [0.0, -0.04, -0.4]], 
                         'SN/g_M': [    [0.08, 0.08], [0.12, 0.12], 
                                        [0.08, 0.08], [0.08, 0.08],
                                        [0.08, 0.08, 0.08, 0.08, 0.08],
                                        [0.08, 0.08, 0.08]],
                         'SN/gna':  [   [14.0, 14.0], [14.0, 14.0], 
                                        [14.0, 14.0], [7.0, 7.0],
                                        [14.0, 14.0, 14.0, 14.0, 14.0],
                                        [14.0, 14.0, 14.0]],
                         'SN/ko':  [    [4.5, 4.5], [4.5, 4.5],
                                        [4.5, 4.5], [4.5, 4.5],
                                        [4.5, 54.5, 4.5, 54.5, 4.5],
                                        [4.5, 4.5, 4.5]],
                         'SN/cAMP': [   [0.2, 0.2], [0.2, 0.2],
                                        [0.2, 0.2], [0.2, 0.2],
                                        [0.2, 0.2, 0.6, 0.6, 0.6],
                                        [0.2, 0.2, 0.2]]}

variables_to_plot = ['SN/V', 'SN/Ca_ter', 'SN/Ca_ER']

######################################
# AND HERE (as well as the plots at the end)

t_list = []
res_list = []

for exp_idx in range(num_experiments):
    current_time = 0
    for idx, sim_time  in enumerate(sim_times[exp_idx]):
        if idx == 0:
            sim_helper = SimulationHelper(model_path, dt, sim_time, maximumNumberofSteps=1000, maximum_step=0.0001, pre_time=pre_times[exp_idx])
            current_time += pre_times[exp_idx]
        else:
            sim_helper.update_times(dt, current_time, sim_time, pre_time=0)
        # change parameters
        sim_helper.set_param_vals(list(params_to_change_dict.keys()), 
                                  [list(params_to_change_dict.values())[II][exp_idx][idx] for \
                                      II in range(len(params_to_change_dict.keys()))])
        sim_helper.run()
        current_time += sim_time
        if idx == 0:
            t_vec = sim_helper.tSim
            res_vec = sim_helper.get_results(variables_to_plot, flatten=True)
        else:
            t_vec = np.concatenate((t_vec, sim_helper.tSim[1:]))
            for var_idx in range(len(variables_to_plot)):
                res_vec[var_idx] = np.concatenate((res_vec[var_idx], 
                                                   sim_helper.get_results(variables_to_plot, 
                                                                          flatten=True)[var_idx][1:]))


    # get results
    t_vec = t_vec - pre_times[exp_idx]
    t_list.append(t_vec)
    res_list.append(res_vec)

    sim_helper.reset_and_clear()

# CHANGE THE PLOTS FOR YOUR SPECIFIC EXPERIMENTS
    
# plot V for M-type activation
var_idx = 0
fig, ax = plt.subplots(2, 1)
for idx in range(2):
    ax[idx].plot(t_list[idx], res_list[idx][var_idx], color=experiment_colors[idx], label=experiment_labels[idx])
    ax[idx].set_xlim([0, max_times[idx]])
    ax[idx].legend(loc='upper left')
    ax[idx].set_ylabel('V [mV]')

# set x and y labels
ax[-1].set_xlabel('time [s]')
fig.savefig(os.path.join(plot_dir, 'V_M_type.png'))

# plot V for Na block
var_idx = 0
fig, ax = plt.subplots(2, 1)
for idx in range(2,4):
    ax[idx-2].plot(t_list[idx], res_list[idx][var_idx], color=experiment_colors[idx], label=experiment_labels[idx])
    ax[idx-2].set_xlim([0, max_times[idx]])
    ax[idx-2].legend(loc='upper left')
    ax[idx-2].set_ylabel('V [mV]')

# set x and y labels
ax[-1].set_xlabel('time [s]')
fig.savefig(os.path.join(plot_dir, 'V_Na.png'))

# plot Cai for K+ and iso 
var_idx = 1
fig, ax = plt.subplots(1, 1)
for idx in range(4,5):
    ax.plot(t_list[idx], mM_to_uM*res_list[idx][var_idx], color=experiment_colors[idx], label=experiment_labels[idx])
    ax.set_xlim([0, max_times[idx]])
    ax.legend(loc='upper left')
    ax.set_ylabel('Ca$_{ter}$ [$\mu$M]')

# set x and y labels
ax.set_xlabel('time [s]')
fig.savefig(os.path.join(plot_dir, 'Cai_K_iso.png'))

# plot Ca_ER for K+ and iso 
var_idx = 2
fig, ax = plt.subplots(1, 1)
for idx in range(4,5):
    ax.plot(t_list[idx], mM_to_uM*res_list[idx][var_idx], color=experiment_colors[idx], label=experiment_labels[idx])
    ax.set_xlim([0, max_times[idx]])
    ax.legend(loc='upper left')
    ax.set_ylabel('Ca$_{ER}$ [$\mu$M]')

# set x and y labels
ax.set_xlabel('time [s]')
fig.savefig(os.path.join(plot_dir, 'Ca_ER_K_iso.png'))

# plot V for diff current inputs
var_idx = 0
fig, ax = plt.subplots(1, 1)
for idx in range(5,6):
    ax.plot(t_list[idx], res_list[idx][var_idx], color=experiment_colors[idx], label=experiment_labels[idx])
    ax.set_xlim([0, max_times[idx]])
    ax.legend(loc='upper left')
    ax.set_ylabel('V [mV]')

# set x and y labels
ax.set_xlabel('time [s]')
fig.savefig(os.path.join(plot_dir, 'V_I_const.png'))

print('plotting done')
