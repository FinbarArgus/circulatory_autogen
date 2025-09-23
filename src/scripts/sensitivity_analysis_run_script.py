import sys
import os
from mpi4py import MPI
root_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir, 'src'))
from sensitivity_analysis.SA import CVS0D_SA
import traceback
import yaml
from parsers.PrimitiveParsers import YamlFileParser
from mpi4py import MPI

def run_SA(inp_data_dict=None):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    yaml_parser = YamlFileParser()
    inp_data_dict = yaml_parser.parse_user_inputs_file(inp_data_dict, obs_path_needed=True, do_generation_with_fit_parameters=False)

    DEBUG = inp_data_dict['DEBUG']
    model_path = inp_data_dict['model_path']
    model_type = inp_data_dict['model_type']
    # SA_method = inp_data_dict['SA_method']
    file_prefix = inp_data_dict['file_prefix']
    params_for_id_path = inp_data_dict['params_for_id_path']
    param_id_obs_path = inp_data_dict['param_id_obs_path']
    sim_time = inp_data_dict['sim_time']
    pre_time = inp_data_dict['pre_time']
    solver_info = inp_data_dict['solver_info']
    dt = inp_data_dict['dt']
    SA_sample_type = inp_data_dict['SA_sample_type']
    ga_options = inp_data_dict['ga_options']
    num_SA_samples = inp_data_dict['num_SA_samples']
    # resources_dir = inp_data_dict['resources_dir']
    SA_output_dir = inp_data_dict['param_id_output_dir']

    
    if rank == 0:
        print(f"SA output dir: {SA_output_dir}")

    # param_orig_vals = inp_data_dict['param_orig_vals']
    # num_samples = inp_data_dict['num_samples']
    # lower_bound_factor = inp_data_dict['lower_bound_factor']
    # upper_bound_factor = inp_data_dict['upper_bound_factor']
    model_out_names = inp_data_dict['model_out_names']

    protocol_info = {
        'pre_times': [pre_time],
        'sim_times': [[sim_time]],
        'dt': dt,
    }
    if rank == 0:
        print(protocol_info)

    SA_cfg = {
        "sample_type" : SA_sample_type,
        "num_samples": num_SA_samples,
    }

    if DEBUG and rank == 0:
        print('WARNING: DEBUG IS ON, TURN THIS OFF IF YOU WANT TO DO ANYTHING QUICKLY')

    SA_manager = CVS0D_SA(model_path, model_out_names, solver_info, SA_cfg, protocol_info, dt, 
                          SA_output_dir, param_id_path=param_id_obs_path, params_for_id_path=params_for_id_path,
                          verbose=False, use_MPI=True, ga_options=ga_options)
    S1_all, ST_all, S2_all = SA_manager.run()

    if rank == 0:
        print('Plotting results')
        print(f">>>>>>{S1_all}")
        SA_manager.plot_sobol_first_order_idx(S1_all, ST_all)
        SA_manager.plot_sobol_S2_idx(S2_all)

    MPI.Finalize()

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    try:
        run_SA()
    except:
        print(traceback.format_exc())
        comm.Abort()