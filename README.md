This project allows the generation and calibration of cellml (and soon to be more) circulatory system models from an array of module/vessel names and connections. 
This array is written in a csv file such as `test_vessel_array.csv` where the entries are detailed as...

```
[vessel_name,

BC_type             ('vv', 'vp', 'pv', 'pp', pp_wCont, pp_wLocal, nn),

vessel_type         ('heart', 'arterial', 'arterial_simple', 'venous', 'terminal', 'split_junction', 'merge_junction', 2in2out_junction, gas_transport_simple, pulomonary_GE, baroreceptor, chemoreceptor),

inp_vessels         (name of the input vessels.)

out_vessels         (name of the output vessels)
]
```

IMPORTANT: The order of input and output vessels is important for the heart module. The order must be

inp_vessels: 1:inferior vena cava, 2:superior vena cava, 3:pulmonary vein

out_vessels: 1:aorta, 2:pulmonary artery.

If the pulmonary vessels aren't included, a simple 2 vessel pulmonary system will be used.

NOTE: currently the terminal vessels should only have a BC starting with 'pp' 

## Process for generating a model and running the parameter identification

# Generating a model

First create a `{file_prefix}_vessel_array.csv` and `{file_prefix}_parameters.csv` file in `resources/`.
`file_prefix`'s of `simple_physiological` and `physiological` are good example files.

Then move to the `user_run_files` directory and ensure the `user_inputs.yaml` file and
opencor_pythonshell_path.sh file are filled out correctly. 
For model generation the following must be set

```yaml
file_prefix: {file_prefix} 
input_param_file: {file_prefix}_parameters.csv
```

```bash
opencor_pythonshell_path=path/to/opencor/pythonshell
```

then run the following to generate the model


```bash
./run_autogeneration.sh
```

If there were missing parameters in `{file_prefix}_parameters.csv` a new file named 
`{file_prefix}_parameters_unfinished.csv` will be created with the required parameters.
This must be filled in and renamed to `{file_prefix}_parameters.csv` to
generate a working model.

# Parameter identification

To then run the parameter identification you must fill in the `param_id` parameters in `user_inputs.sh`,
create a `{file_prefix}_params_for_id.csv` file and
create a json file with the ground truth data. See `resources/simple_physiological_params_for_id.csv` and `resources/simple_physiological_obs_data.json` for an example.

```bash
./run_param_id.sh
```

Following a successful parameter id process the model with updated parameters can be generated with
```bash
./run_autogeneration_with_id_params.sh
```

The generated models will be saved in `generated_models/`


## Creating your own modules.

This software is designed so the user can easily make their own modules and couple them with existing modules. The steps are as follows...

1: Either choose an existing '{module\_category}\_modules.cellml' file to write your module, or if it is a new category of module, create a '{module\_category}\_modules.cellml' file in 'src/generators/resources/'

2: Put you cellml model into the '{module\_category}\_modules.cellml' file.

3: create a corresponding module configuration entry into 'module\_config.json'. These module declarations detail the variables that can be accessed, the constants that must be defined and the available ports of the module.
    
4: include your new module into the vessel array file. IMPORTANT: modules that are connected as eachothers inputs and outputs will be coupled together with any ports with corresponding name. i.eif VesselOne has an entrance 'vessel\_port' and VesselTwo has in entrance 'vessel\_port', they will be coupled with the variables declared in their corresponding 'vessel\_port'. One must be careful, when making a new module, that the modules it couples to only has matching port types for the ones that are necessary for coupling. 

## CHANGES TO BE MADE

Currently "vessels" is used interchangeabley with "modules". This will be changes to use "modules" in all instances, as the project now allows all types of modules, not just vessels.

The connections between terminals and the venous system is hardcoded, as a terminal_venous_connection has to be made to sum up the flows and get an avergae concentration. This needs to be improved.

## Tests!! 

There is now a test for the autogeneration running. To run the test navigate to user_run_files and do ./run_test_autogeneration.sh

## requirements  

If the model being generated is a cellml model, OpenCOR must be downloaded 
and installed from [opencor](https://opencor.ws/downloads/index.html)

To install required python packages for this opencors version of python
you must do the following...  

```bash
/path/to/opencor/dir/python/bin/python -m pip install packagename
```

### Required packages for parameter identification
mpi4py
scikit-optimize
emcee
corner
schwimmbad
tqdm

### Required packages for autogeneration
pandas

### Potential Errors:
IMPORTANT if you get an SSL error you must do the following before the pip install

```bash
export LD_LIBRARY_PATH=[OpenCOR]/lib
```

so that libcrypto.so.3 can be
found to load the ssl module.

IMPORTANT intalling mpi4py requires mpi to be available. Therefore, the following lines
may be required to install the mpi software on your computer

```bash
sudo apt install libopenmpi-dev
sudo apt install libffi7
```

### Windows package instalment
Running on Windows is in development. It can work with some caveats...

1) To be able to importing mpi4py you may have to do the following:

1. download MS MPI, install both .mis and SDK.

2. set up environmental variables

control panel --> advanced system settings --> environmental variables --> add

C:\Program Files\Microsoft MPI\

C:\Program Files (x86)\Microsoft SDKs\MPI\

## Running in Windows
For running scripts you cannot use the .sh files, So you have to run the scripts in src/scripts explicitly.
e.g for running param id...

First move to C:\path\to\opencor\dir

Then run

```bash
.\pythonshell.bat C:\path\to\circulatory\autogen\src\scripts\param_id_run_script.py
```

# License
circulatory_autogen is fully open source and distributed under the very permissive Apache License 2.0. See LICENSE for more information.
