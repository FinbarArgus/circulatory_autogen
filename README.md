This project allows the generation of cellml circulatory system models from an array of vessel names and connections. 
This array is written in a csv file such as test_vessel_array.csv where the entries are detailed as...


[vessel_name,
BC_type             ('vv', 'vp', 'pv', 'pp'),
vessel_type         ('heart', 'arterial', 'arterial_simple', 'venous', 'terminal', 'split_junction', 'merge_junction', 2in2out_junction),
inp_vessels         (name of the input vessels.)
out_vessels         (name of the output vessels)
]

IMPORTANT: The order of input and output vessels is important for the heart module. The order must be
inp_vessels: 1:inferior vena cava, 2:superior vena cava, 3:pulmonary vein
out_vessels: 1:aorta, 2:pulmonary artery

If the pulmonary vessels aren't included, a simple 2 vessel pulmonary system will be used.

The aim is to combine the autogeneration of a circulatory system structure graph from an image with this code to
completely automate the creation of circulatory system models from images.

NOTE: currently the terminal vessels should only have 'pp' type boundary conditions

## requirements  

If the model being generated is a cellml model, OpenCOR must be downloaded 
and installed from https://opencor.ws/downloads/index.html  

To install required python packages for this opencors version of python
you must do the following...  

/path/to/opencor/dir/python/bin/python -m pip install packagename

### Required packages for parameter identification
mpi4py
scikit-optimize

### Required packages for autogeneration
pandas

IMPORTANT If installing on CENTOS, if you get an SSL error you must do the following before the pip install

export LD_LIBRARY_PATH=[OpenCOR]/lib

so that libcrypto.so.1.1 could be
found. I don't understand why this :is necessary, but it is

IMPORTANT intalling mpi4py may require the following line 
to install the mpi software on your computer

sudo apt install libopenmpi-dev
