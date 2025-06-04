import pandas as pd
import re
from sys import exit

param_cellml_file = 'parameters_autogen.cellml'
param_csv_file = 'parameters_autogen_new.csv'

data = {'variable_name':[], 'units':[], 'comp_env':[], 'value':[]}
section = ''
with open(param_cellml_file, 'r') as r:
    for line_number, line in enumerate(r, start=1):
        if 'parameters_pulmonary' in line:
            section = 'pulmonary'
            continue
        elif 'parameters_heart' in line:
            section = 'heart'
            continue
        elif 'parameters_systemic' in line:
            section = 'systemic'
            continue

        if section == 'pulmonary':
            if '<variable initial_value' in line:
                try:
                    variable_name = re.search('name="(\w+)"', line).group(1)
                    units = re.search('units="(\w+)"', line).group(1)
                    value = re.search('initial_value="(\d*.?\d*e?[-+]?\d*)"', line).group(1)
                except AttributeError:
                    print(f'couldn\'t find variable name, unit, or value in line number {line_number}')
                    exit()

                data['variable_name'].append(variable_name)
                data['units'].append(units)
                data['comp_env'].append('pulmonary')
                data['value'].append(value)
        elif section == 'heart':
            if '<variable initial_value' in line:
                try:
                    variable_name = re.search('name="(\w+)"', line).group(1)
                    units = re.search('units="(\w+)"', line).group(1)
                    value = re.search('initial_value="(\d*.?\d*e?[-+]?\d*)"', line).group(1)
                except AttributeError:
                    print(f'couldn\'t find variable name, unit, or value in line number {line_number}')
                    exit()

                data['variable_name'].append(variable_name)
                data['units'].append(units)
                data['comp_env'].append('heart')
                data['value'].append(value)

        elif section == 'systemic':
            if '<variable initial_value' in line:
                try:
                    variable_name = re.search('name="(\w+)"', line).group(1)
                    units = re.search('units="(\w+)"', line).group(1)
                    value = re.search('initial_value="(\d*.?\d*e?[-+]?\d*)"', line).group(1)
                except AttributeError:
                    print(f'couldn\'t find variable name, unit, or value in line number {line_number}')
                    exit()

                data['variable_name'].append(variable_name)
                data['units'].append(units)
                data['comp_env'].append('systemic')
                data['value'].append(value)

        else:
            continue

    df = pd.DataFrame(data)
    df.to_csv(param_csv_file, index=None, header=True)
    print(f'csv file created at "{param_csv_file}"')


