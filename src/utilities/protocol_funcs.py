import numpy as np
import os, sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
sys.path.append(os.path.dirname(__file__))
root_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir, 'src'))
user_inputs_dir = os.path.join(root_dir, 'user_run_files')
from opencor_helper import SimulationHelper
import paperPlotSetup
paperPlotSetup.Setup_Plot(3)

def run_protocols(model_path, variables_to_plot, protocol_info=None, inp_data_dict=None):
    """ runs the protocols defigned by protocol infor or if not defined, gets 
    protocol info from the obs_data.json file.

    Args:
        variables_to_plot (list[string]): list of variables to plot
        protocol_info (dict, optional): _description_. includes all information for simulating
            the protocols.
        inp_data_dict (dict, optional): _description_. If you don't want to use the circulatory_autogen user_inputs
            dict, you can define your own.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    if inp_data_dict is None:
        import yaml
        with open(os.path.join(user_inputs_dir, 'user_inputs.yaml'), 'r') as file:
            inp_data_dict = yaml.load(file, Loader=yaml.FullLoader)
    
    dt = inp_data_dict['dt']

    if protocol_info is None:
        param_id_obs_path = inp_data_dict['param_id_obs_path']
        with open(param_id_obs_path, encoding='utf-8-sig') as rf:
            json_obj = json.load(rf)

        protocol_info = json_obj['protocol_info']

    sim_times = protocol_info['sim_times']
    pre_times = protocol_info['pre_times']
    experiment_colors = protocol_info['experiment_colors']
    params_to_change_dict = protocol_info['params_to_change']


    num_experiments = len(sim_times)
    if num_experiments != len(pre_times):
        raise ValueError('pre_times and sim_times must be the same length')
        
    max_times = [np.sum(sim_times[II]) for II in range(num_experiments)]

    ######################################
    # AND HERE (as well as the plots at the end)

    t_list = []
    res_list = []

    for exp_idx in range(num_experiments):
        current_time = 0
        for idx, sim_time  in enumerate(sim_times[exp_idx]):
            if idx == 0:
                sim_helper = SimulationHelper(model_path, dt, sim_time, solver_info={'maximumNumberofSteps':1000, 'maximum_step':0.0001}, 
                                              pre_time=pre_times[exp_idx])
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
    return t_list, res_list
