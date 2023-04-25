import json
import pandas as pd
case_name = 'lobe_imped-prePEA'
data_file_path = f'/home/farg967/Documents/data/pulmonary/{case_name}.json'
vessel_array_path = f'/home/farg967/Documents/git_projects/circulatory_autogen/resources/lung_ROM_vessel_array.csv'
save_file_path = f'/home/farg967/Documents/git_projects/circulatory_autogen/resources/pulmonary_{case_name}_obs_data.json'


with open(data_file_path, 'r') as file:
    data = json.load(file)

vessel_array = pd.read_csv(vessel_array_path, index_col=0)
# dyne_s_per_cm5_to_J_s_per_m6 
conversion = 1e5

full_dict = {}
entry_list = []
for II in range(len(data["vessel_names"])):
    entry = {}
    entry["variable"] = data["vessel_names"][II]
    if entry["variable"].endswith("_V"):
        # Temporarily skip the venous vessel data
        continue
    entry["data_type"] = "frequency"
    entry["operation"] = "division"
    input_vessel = vessel_array["inp_vessels"][data["vessel_names"][II]].strip()
    BC_type = vessel_array["BC_type"][data["vessel_names"][II]].strip()
    if BC_type.startswith("p"):
        entry["operands"] = [f'{input_vessel}/u',
                             f'{data["vessel_names"][II]}/v']
    else:
        entry["operands"] = [f'{data["vessel_names"][II]}/u',
                             f'{input_vessel}/v']

    entry["unit"] = "Js/m^6" # data["impedance"]["unit"]
    entry["obs_type"] = "frequency"
    entry["value"] = [val*conversion for val in data["impedance"][data["vessel_names"][II]]]
    entry["std"] = [conversion*val/10 for val in data["impedance"][data["vessel_names"][II]]]
    entry["frequencies"] = data["frequency"]
    entry["phase"] = [val for val in data["phase"][data["vessel_names"][II]]]
    # temporarily hardcode the weights for the phase
    entry["weight"] = [1.0 for val in data["phase"][data["vessel_names"][II]]]
    entry["weight"][0] = 10
    entry["weight"][1] = 8
    entry["weight"][2] = 6
    entry["weight"][3] = 4
    entry["weight"][4] = 3
    entry["weight"][5] = 2
    entry["phase_weight"] = [200.0 for val in data["phase"][data["vessel_names"][II]]]
    entry["phase_weight"][0] = entry["phase_weight"][0]*3
    entry["phase_weight"][1] = entry["phase_weight"][1]*2
    for II in range(4, len(entry["phase_weight"])):
        entry["phase_weight"][II] = 0.0
    entry_list.append(entry)

full_dict["data_item"] = entry_list

with open(save_file_path, 'w') as wf: 
    json.dump(full_dict, wf, indent=2)


