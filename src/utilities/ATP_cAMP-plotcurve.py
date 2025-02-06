# import opencor as oc
import sys
import os
import numpy as np
from opencor_helper import SimulationHelper
import csv
#from PyQt5.QtWidgets import QApplication
import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import paperPlotSetup
matplotlib.use('Agg')

# TODO 
#preset some address, such as input model address, results saved address
file_path = "/home/cdon822/Documents/OpenCOR/ISO_cAMP_modelv3.cellml"
Outputfile_address = "/home/cdon822/Documents/GIT_files/my_project/simulation results/SAN_model/"
#create a new temporary files to save the results
save_newfilename1 = "ATP_cAMP-test"
save_address1 = Outputfile_address + "/"+save_newfilename1
if not os.path.exists(save_address1):
    os.mkdir(save_address1)
#spliced into new address
#save_addr01 = Outputfile_address + "/"+save_newfilename1+"/ATP_0_cAMP_plot.png"


#those are parameters that need to test in this scripts
#exp:[Membrane/V_node] represent the [module/variable_name]

#obs_list = ['Rate_modulation_experiments/q_I','Rate_modulation_experiments/q_b1GsI']
#str_title = "ISO-beta1AR relationship"
#obs_list = ['Rate_modulation_experiments/q_I','Rate_modulation_experiments/q_b2GsI']
#str_title = "ISO-beta2AR relationship"
#obs_list = ['Rate_modulation_experiments/q_I','Rate_modulation_experiments/q_Gs']
#str_title = "ISO-Gs relationship"
#obs_list = ['Rate_modulation_experiments/q_I','Rate_modulation_experiments/q_Gi']
#str_title = "ISO-Gi relationship"
#obs_list = ['Rate_modulation_experiments/q_I','Rate_modulation_experiments/q_ACGs']
#str_title = "ISO-ACGs relationship"
obs_list = ['Rate_modulation_experiments/q_ATP','Rate_modulation_experiments/q_cAMP']
str_title = "ATP-cAMP relationship"


#save the results about ATP=0, ATP-cAMP curve
#preset the fontsize for specific parameters, only need set once
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 24

#I don't know why cannot show the curve, but can save the curve
#app = QApplication(sys.argv) if QApplication.instance() is None else QApplication.instance()

#which parameter need to extract and reset the value during iteration
#type: ['function_name/variable_name']
reset_param_names = ['Rate_modulation_experiments/q_ATP']
#set initial value
reset_param_val = [1]
#define a variable that save result from iteration
inputs = []
outputs = []

#preset iteration times, to realized that increase the Iso concentration for different simualtion
for ii in range(1,55):
    x = SimulationHelper(file_path, 0.01, 10, pre_time=0)
    #used for debug
    print("ii=",ii)
    #notice, in here, must be cautious to set the formula about change of variable parameters
    reset_param_val[0] = reset_param_val[0]+2
    print("reset_value:",reset_param_val)
    x.set_param_vals(reset_param_names, reset_param_val)
    x.run()
    z = x.get_results(obs_list)
    z1 = z[1][0]
    #z_out = z[1][100]
    #print the output
    inputs.append(reset_param_val[0])
    outputs.append(z1[1000])
    print("z=",z1[1000])
    
#set the final curve saved address and image name
save_addr1 = save_address1+"/"+str_title+".png"

#set what we need to show at the curve image
plt.plot(inputs,outputs,marker='.',linestyle='-',color='r',label='curve-relationship')
plt.title(str_title)
#plt.xscale('log')
#plt.xticks([0.1, 1, 10, 100, 1000], labels=["0.1", "1", "10", "100", "1000"])

plt.xlabel('ATP [nM]')
plt.ylabel('cAMP [nM]')
plt.grid(True)
plt.savefig(save_addr1,bbox_inches='tight')
#plt.savefig(save_addr1)
plt.clf()
print("results saved as 'png' type")
print("[in,out]=",inputs, outputs)

# I don't know why cannot plot the result curve, but can save it in the script address
#plt.show()



