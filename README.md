This project allows the generation of cellml circulatory system models from an array of vessel names and connections. 
This array is written in a csv file such as test_vessel_array.csv where the entries are detailed as...


[vessel_name,
BC_type             ('vv', 'vp', 'pv', 'pp'),
vessel_type         ('arterial', 'venous', 'terminal', 'split_junction', 'merge_junction', 2in2out_junction),
inp_vessel_1        (name of the first input vessel. This doesn't need to be specified for venous type
inp_vessel_2        (name of the second input vessel if merge or 2in2out junction, '' otherwise)
out_vessel_1        (name of first output vessel)
out_vessel_2        (name of second output vessel if split or 2in2out junction, '' otherwise)
]

The aim is to combine the autogeneration of a circulatory system structure graph from an image with this code to
completely automate the creation of circulatory system models from images.

## requirements  

If the model being generated is a cellml model, OpenCOR must be downloaded 
and installed from https://opencor.ws/downloads/index.html  

To install required python packages for this opencors version of python
you must do the following...  

/path/to/opencor/dir/python/bin/python -m pip install packagename

### Required packages for parameter identification
mpi4py

### Required packages for autogeneration
pandas
