import pandas as pd
import json
import numpy as np
from scipy import signal as sig

output_json_file_name = "/home/finbar/Documents/data/cardiohance_data/cardiohance_observables.json"

excel_paths = ['/home/finbar/Documents/data/cardiohance_data/CardiacFunction.xlsx']
mmHg_to_Pa = 133.332
ml_per_s_to_m3_per_s = 1e-6
dt = 0.01
T = 1.0
nSteps = int(T/dt)


variable_info = [[('Pressure_raw', 'aortic_root/u', 'alg', mmHg_to_Pa),
                  ('FlowRate_raw', 'aortic_root/v', 'state', ml_per_s_to_m3_per_s)]]
# (sheet_name, variable_name, state_or_alg, conversion_rate)
column_infos = [[[('Cardiohance_029', ['mean', 'min', 'max', 'series'])],
                 [('Cardiohance_029', ['mean', 'min', 'max', 'series'])]]]
# (column name, list of obs types to input into json)
entry_dict = {'data_item': []}
for file_idx, excel_path in enumerate(excel_paths):
    for sheet_idx, (sheet_name, variable_name, state_or_alg, conversion) in enumerate(variable_info[file_idx]):
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        for column_info in column_infos[file_idx][sheet_idx]:
            column_name = column_info[0]
            obs_types = column_info[1]
            series = df[column_name]
            t_resampled = np.linspace(0, 2*T, 2*nSteps + 1)
            series_resampled = sig.resample(series, nSteps + 1)
            series_rs_2period = np.concatenate([series_resampled, series_resampled[1:]])

            if 'mean' in obs_types:
                mean_val = np.mean(series_resampled)
                entry = {'variable': variable_name,
                         'data_type': 'constant',
                         'state_or_alg': state_or_alg,
                         'weight': 1.0,
                         'obs_type': 'mean',
                         'value': mean_val*conversion}
                entry_dict["data_item"].append(entry)
            if 'min' in obs_types:
                min_val = np.min(series_resampled)
                entry = {'variable': variable_name,
                         'data_type': 'constant',
                         'state_or_alg': state_or_alg,
                         'weight': 1.0,
                         'obs_type': 'min',
                         'value': min_val*conversion}
                entry_dict["data_item"].append(entry)
            if 'max' in obs_types:
                max_val = np.max(series_resampled)
                entry = {'variable': variable_name,
                         'data_type': 'constant',
                         'state_or_alg': state_or_alg,
                         'weight': 1.0,
                         'obs_type': 'max',
                         'value': max_val*conversion}
                entry_dict["data_item"].append(entry)
            if 'series' in obs_types:
                entry = {'variable': variable_name,
                         'data_type': 'series',
                         'state_or_alg': state_or_alg,
                         'weight': 0.0,
                         'obs_type': 'series',
                         'series': series_rs_2period*conversion,
                         't': t_resampled}
                entry_dict["data_item"].append(entry)

# Now create json file from entry_dict
json_df = pd.DataFrame(entry_dict)
# json_df.to_json(output_json_file_name)
result = json_df.to_json()
print(result)
parsed = json.loads(result)
with open(output_json_file_name, 'w') as wf:
    json.dump(parsed, wf, indent=2)
