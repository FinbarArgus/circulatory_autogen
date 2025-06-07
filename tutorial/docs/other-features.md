## Other Useful Scripts

These scripts can be found in `src/scripts/`.
All files below are in development and the code may need to be changed for your own application.

# generate_modules_files.py

This script converts a generic cellml file into the modular format that can be used by Circulatory_Autogen.
- It currently only works for models made from a single module

# generate_obs_json.py

This script contains an example of automating the creation of an obs_data.json file from input data.
- Modify this for your own data if you don't want to modify the json file by hand

# read_and_insert_parameters.py

This script modifies your parameters.csv file with the parameters and values in a json file that is input to the script
when running it.