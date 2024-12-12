# import opencor as oc
import numpy as np
from opencor_helper import SimulationHelper
import csv
import paperPlotSetup
paperPlotSetup.Setup_Plot(3)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# TODO get all of the info from the user_inputs.yaml file

file_path = "/home/farg967/Documents/git_projects/circulatory_autogen/generated_models/lung_control_lv_estimation_observables_BB044/lung_control.cellml"
output_file_path = "/home/farg967/Documents/git_projects/circulatory_autogen/param_id_output/genetic_algorithm_lung_control_lv_estimation_observables_BB044/plots_param_id"
pre_time = 30.5
period = 1.0

sim_object = SimulationHelper(file_path, 0.01, period, solver_info={'MaximumStep':0.001, 'MaximumNumberofSteps':500000}, pre_time=pre_time)
sim_object.run()

# IF you want to change parameters and run again, do it here. 


y = sim_object.get_results(['LPA_A/u', 'RPA_A/u', 'RPV_V/u', 'heart/u_rv'])
t = sim_object.tSim - pre_time

conversion = 1/133.322

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
axs[0,0].plot(t, y[0]*conversion)#, label='LPA_A/u')
axs[0,1].plot(t, y[1]*conversion)#, label='RPA_A/u')
axs[1,0].plot(t, y[2]*conversion)#, label='LPV_V/u')
axs[1,1].plot(t, y[3]*conversion)#, label='heart/u_rv')

axs[0,0].set_xlim([0, period])
axs[0,0].set_ylim([0, 50])
axs[0,0].set_xlabel('Time [s]')
axs[0,0].set_ylabel('LPA P [mmHg]')
axs[0,1].set_xlim([0, period])
axs[0,1].set_ylim([0, 50])
axs[0,1].set_xlabel('Time [s]')
axs[0,1].set_ylabel('RPA P [mmHg]')
axs[1,0].set_xlim([0, period])
axs[1,0].set_ylim([0, 50])
axs[1,0].set_xlabel('Time [s]')
axs[1,0].set_ylabel('LPV P [mmHg]')
axs[1,1].set_xlim([0, period])
axs[1,1].set_ylim([0, 50])
axs[1,1].set_xlabel('Time [s]')
axs[1,1].set_ylabel('RV P [mmHg]')

plt.savefig(os.path.join(output_file_path, 'model_outputs_u.png'))
plt.savefig(os.path.join(output_file_path, 'model_outputs_u.eps'))





