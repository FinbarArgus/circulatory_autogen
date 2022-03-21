This project allows the generation of cellml circulatory system models from an array of vessel names and connections. 
This array is written in a csv file such as `test_vessel_array.csv` where the entries are detailed as...

```
[vessel_name,

BC_type             ('vv', 'vp', 'pv', 'pp'),

vessel_type         ('heart', 'arterial', 'arterial_simple', 'venous', 'terminal', 'split_junction', 'merge_junction', 2in2out_junction),

inp_vessels         (name of the input vessels.)

out_vessels         (name of the output vessels)
]
```

IMPORTANT: The order of input and output vessels is important for the heart module. The order must be

inp_vessels: 1:inferior vena cava, 2:superior vena cava, 3:pulmonary vein

out_vessels: 1:aorta, 2:pulmonary artery.

If the pulmonary vessels aren't included, a simple 2 vessel pulmonary system will be used.

The aim is to combine the autogeneration of a circulatory system structure graph from an image with this code to
completely automate the creation of circulatory system models from images.

NOTE: currently the terminal vessels should only have 'pp' type boundary conditions

## Process for generating a model and running the parameter identification

First create a `{file_prefix}_vessel_array.csv` and `{file_prefix}_parameters.csv` file in `resources/`.
`file_prefix`'s of `simple_physiological` and `physiological` are good example files.

Then move to the `user_run_files` directory and ensure the `user_inputs.sh` file is filled out correctly. 
For model generation the following must be set

```bash
file_prefix={file_prefix} 
input_param_file={file_prefix}_parameters.csv
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

To then run the parameter identification you must fill in the `param_id` parameters in `user_inputs.sh`,
create a `{file_prefix}_params_for_id.csv` file and
create a json file with the ground truth data. See `resources/simple_physiological_params_for_id.csv` and `resources/simple_physiological_obs_data.json` for an example.

```bash
./run_param_id.sh
```

The above will output progress and errors to /param_id_output/log.txt

Following a succesfull parameter id process the model with updated parameters can be generated with
```bash
./run_autogeneration_with_id_params.sh
```

The generated models will be saved in `generated_models/`


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

### Required packages for autogeneration
pandas

IMPORTANT If installing on CENTOS, if you get an SSL error you must do the following before the pip install

```bash
export LD_LIBRARY_PATH=[OpenCOR]/lib
```

so that libcrypto.so.1.1 could be
found. I don't understand why this :is necessary, but it is

IMPORTANT intalling mpi4py may require the following line 
to install the mpi software on your computer

```bash
sudo apt install libopenmpi-dev
```

