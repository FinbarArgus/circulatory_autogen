"""
Converts a CellML model to a module and generate files to run Circulatory Autogen.
Inputs: CellML model and the output directory where parameters.csv and vessel_array.csv files are created
"""
import argparse
import json
import os.path
import sys
import xml.etree.ElementTree as ET

import libcellml as lc

# Define namespaces
cellml_namespace = "http://www.cellml.org/cellml/1.1#"
mathml_namespace = "http://www.w3.org/1998/Math/MathML"

# Define units file paths
user_units_cellml = '../module_config_user/user_units.cellml'
units_cellml = '../src/generators/resources/units.cellml'

# Define file_prefix, vessel_name and data_reference for the model
file_prefix = "NKE_pump"
vessel_name = "NKE_system"
data_reference = "user_defined"

# Parse arguments
def _parse_args():
    parser = argparse.ArgumentParser(prog="convert_cellml_for_ca")
    parser.add_argument("-i", "--input-model", help="import CellML model to convert.", required=True)
    parser.add_argument("-o", "--output-dir", help="output directory for converted model data.", required=True)
    return parser.parse_args()

# Print errors
def _print_errors(l):
    print("Total errors: ", l.errorCount())
    for i in range(l.errorCount()):
        print(f"{i + 1}: {l.error(i).description()}")

# Extract variables, constants and state variables from the model
def _extract_variables_constants_states(analysed_model):
    variables = []
    constants = []
    states = []

    # Get variables and constants
    for i in range(analysed_model.variableCount()):
        analysed_model_variable = analysed_model.variable(i)
        variable = analysed_model_variable.variable()
        type_as_string = analysed_model_variable.typeAsString(analysed_model_variable.type())

        if type_as_string == "algebraic":
            variables.append([variable.name(), variable.units().name(), "variable"])
        elif type_as_string == "constant":
            constants.append([variable.name(), variable.units().name(), "constant", variable.initialValue()])

    # Get state variables and set initial values
    for i in range(analysed_model.stateCount()):
        analysed_model_state = analysed_model.state(i)
        state = analysed_model_state.variable()
        constants.append([f"{state.name()}_init", state.units().name(), "constant", state.initialValue()])
        states.append([state.name(), state.units().name(), "variable"])

    return variables, constants, states

# Generate module_config.json file
def _generate_module_config(variables, constants, states, file_prefix, component_name):
    config = {
        "vessel_type": file_prefix,
        "BC_type": "nn",
        "module_format": "cellml",
        "module_file": f"{file_prefix}_modules.cellml",
        "module_type": component_name,
        "entrance_ports": [],
        "exit_ports": [],
        "variables_and_units": []
    }

    # Add variables
    for variable in variables:
        info = [variable[0], variable[1], "access", variable[2]]
        config["variables_and_units"].append(info)

    # Add constants
    for constant in constants:
        info = [constant[0], constant[1], "access", constant[2]]
        config["variables_and_units"].append(info)

    # Add state variables
    for state in states:
        info = [state[0], state[1], "access", state[2]]
        config["variables_and_units"].append(info)

    module_config = [config]

    file_path = os.path.join('../module_config_user', f"{file_prefix}_module_config.json")

    with open(file_path, "w") as fh:
        json.dump(module_config, fh, indent=2)

# Generate vessel_array.csv file
def _generate_vessel_array_csv(output_dir, vessel_name, file_prefix):

    file_path = os.path.join(output_dir, f"{file_prefix}_vessel_array.csv")

    with open(file_path, "w") as fh:
        fh.writelines(["name,BC_type,vessel_type,inp_vessels,out_vessels\n"])
        fh.writelines([f"{vessel_name},nn,{file_prefix},,\n"])

# Generate parameters.csv file
def _generate_parameters_csv(output_dir, constants, vessel_name, file_prefix, data_reference):

    file_path = os.path.join(output_dir, f"{file_prefix}_parameters.csv")

    with open(file_path, "w") as fh:
        fh.writelines(["variable_name,units,value,data_reference\n"])

        for constant in constants:
            fh.writelines([f"{constant[0]}_{vessel_name},{constant[1]},{constant[3]},{data_reference}\n"])

# Generate CellML module
def _generate_cellml_module(input_model, states, file_prefix):

    # Parse the cellml file
    tree = ET.parse(input_model)
    root = tree.getroot()

    # Add xmlns:cellml to the root model element
    root.set("xmlns:cellml", cellml_namespace)

    # Register namespaces
    ET.register_namespace("", cellml_namespace) 
    ET.register_namespace("math", mathml_namespace)

    # Move units to user_units.cellml
    _update_units_file(root)

    # Modify variable elements
    _modify_variables(root)

    # Modify state variable elements
    _modify_state_variables(root, states)

    # Ensure math elements are using the proper namespace
    for math in root.findall(f".//{{{mathml_namespace}}}math"):
        math.attrib["xmlns"] = mathml_namespace

    # Modify the 'cn' elements inside math to have 'cellml:' for the 'units' attribute
    for cn in root.findall(f".//{{{mathml_namespace}}}cn"):
        if f"{{{cellml_namespace}}}units" in cn.attrib:
            cn.set("cellml:units", cn.attrib[f"{{{cellml_namespace}}}units"])
            del cn.attrib[f"{{{cellml_namespace}}}units"]

    # Remove math: prefix for all MathML elements
    cellml_str = ET.tostring(root, encoding="UTF-8", xml_declaration=True).decode("utf-8")
    cellml_str = cellml_str.replace('math:', '')

    # Write the modified cellml
    file_path = os.path.join('../module_config_user', f"{file_prefix}_modules.cellml")
    with open(file_path, "w", encoding="UTF-8") as f:
        f.write(cellml_str)

# Move units to user_units.cellml
def _update_units_file(root):

    # Extract units from user_units.cellml
    user_units_tree = ET.parse(user_units_cellml)
    user_units_root = user_units_tree.getroot()
    user_units = {unit.get("name"): unit for unit in user_units_root.findall(f".//{{{cellml_namespace}}}units")}

    # Extract units from units.cellml
    units_tree = ET.parse(units_cellml)
    units_root = units_tree.getroot()
    units = {unit.get("name"): unit for unit in units_root.findall(f".//{{{cellml_namespace}}}units")}

    # Extract units from original model
    units_dict = {unit.get("name"): unit for unit in root.findall(f".//{{{cellml_namespace}}}units")}


    # Add units to user_units.cellml
    for name, unit in units_dict.items():
        if name not in user_units and name not in units:
            user_units_root.append(unit)

    # Remove units from the module
    for unit in root.findall(f".//{{{cellml_namespace}}}units"):
        root.remove(unit)

    user_units_tree.write(user_units_cellml, xml_declaration=True, encoding="UTF-8")

# Modify variable elements in cellml module
def _modify_variables(root):
    for variable in root.findall(f".//{{{cellml_namespace}}}variable"):
        if variable.attrib["name"]=="t":
            variable.set("public_interface", "in")
            if "initial_value" in variable.attrib:
                del variable.attrib["initial_value"]
        elif "initial_value" in variable.attrib:
            del variable.attrib["initial_value"]
            variable.set("public_interface", "in")
        else:
            variable.set("public_interface", "out")

# Modify state variables in cellml module
def _modify_state_variables(root, states):
    for state in states:
        var_name = state[0]
        units = state[1]
        
        variable = root.find(f".//{{{cellml_namespace}}}variable[@name='{var_name}']")
        
        if variable is not None:
            new_name = f"{var_name}_init"
            variable.set("name", new_name)
            
            new_variable = ET.Element(f"{{{cellml_namespace}}}variable", name=var_name, units=units, public_interface="out", initial_value=new_name)
            
            root.find(f".//{{{cellml_namespace}}}component").append(new_variable)

def main():
    args = _parse_args()
    if not os.path.isfile(args.input_model):
        sys.exit(1)

    if not os.path.isdir(args.output_dir):
        sys.exit(2)

    with open(args.input_model) as fh:
        content = fh.read()

    # Parse model
    parser = lc.Parser(False)
    model = parser.parseModel(content)

    # Get component
    component = model.component(0)

    # Remove initial value of variable t
    variable_t = component.variable("t")
    variable_t.removeInitialValue()

    # Validate model
    validator = lc.Validator()
    validator.validateModel(model)

    if validator.errorCount() > 0:
        _print_errors(validator)
        sys.exit(3)

    # Analyse model
    analyser = lc.Analyser()
    analyser.analyseModel(model)

    if analyser.errorCount() > 0:
        _print_errors(analyser)
        sys.exit(4)

    analysed_model = analyser.model()

    variables, constants, states = _extract_variables_constants_states(analysed_model)

    _generate_module_config(variables, constants, states, file_prefix, component.name())

    _generate_parameters_csv(args.output_dir, constants, vessel_name, file_prefix, data_reference)

    _generate_vessel_array_csv(args.output_dir, vessel_name, file_prefix)

    _generate_cellml_module(args.input_model, states, file_prefix)

if __name__ == "__main__":
    main()
