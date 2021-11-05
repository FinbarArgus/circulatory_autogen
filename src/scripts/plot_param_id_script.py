'''
Created on 29/10/2021

@author: Finbar J. Argus
'''

import sys
import os
from mpi4py import MPI

root_dir_path = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir_path, 'src'))

resources_dir_path = os.path.join(root_dir_path, 'resources')
param_id_dir_path = os.path.join(root_dir_path, 'src/param_id')
generated_models_dir_path = os.path.join(root_dir_path, 'generated_models')

from param_id.paramID import CVS0DParamID
import traceback

if __name__ == '__main__':

    try:

        param_id_method = sys.argv[1]
        file_name_prefix = sys.argv[2]
        model_path = os.path.join(generated_models_dir_path, f'{file_name_prefix}.cellml')
        param_id_model_type = 'CVS0D' # TODO make this an input variable eventually

        param_id = CVS0DParamID(model_path, param_id_model_type, param_id_method, file_name_prefix)

        param_id.simulate_with_best_param_vals()
        param_id.plot_outputs()
        param_id.close_simulation()

    except:
        print(traceback.format_exc())
        print("Usage: file_name_prefix")
        print("e.g. simple_physiological")
        exit()
