'''
Created on 23/05/2023

@author: Finbar Argus
'''

import numpy as np
import re
import pandas as pd
import os
from sys import exit
generators_dir_path = os.path.dirname(__file__)
from generators.CVSCellMLGenerator import CVS0DCellMLGenerator
from libcellml import Annotator
import utilities.libcellml_helper_funcs as cellml
from rdflib import Graph, Literal, RDF, URIRef
from rdflib.namespace import DCTERMS



class CVS0DCppGenerator(object):
    '''
    Generates Cpp files from CellML files and annotations for a 0D model
    '''


    def __init__(self, model, generated_model_subdir, filename_prefix, resources_dir=None, 
                 solver='CVODE'):
        '''
        Constructor
        '''
        self.model = model
        self.generated_model_subdir = generated_model_subdir
        if not os.path.exists(self.generated_model_subdir):
            os.mkdir(self.generated_model_subdir)
        self.filename_prefix = filename_prefix
        self.filename_prefix_with_ids = f'{self.filename_prefix}-with-ids'
        if resources_dir is None:
            self.resources_dir = os.path.join(generators_dir_path, '../../resources')
        else:
            self.resources_dir = resources_dir 
        self.generated_model_file_path = os.path.join(self.generated_model_subdir, 
                                                      self.filename_prefix + '.cellml')
        self.annotated_model_file_path = os.path.join(self.generated_model_subdir, 
                                                      self.filename_prefix_with_ids + '.cellml')
        
        self.cellml_model = cellml.parse_model(self.generated_model_file_path, False)
        self.solver = solver

        # these are for delayed variables
        self.variables_to_delay = []
        self.delayed_variables = []
        self.delay_amounts = []
        self.independent_variables = []
        # these are for two-way coupling
        self.port_input_variables = []
        self.port_output_variables = []
        self.connection_vessel_indices = []
        # these are for one-way coupling
        input_variables = []
        output_variables = []


        # TODO should the annotations stuff be in a different class? It's not crucial for 
        # running of the model but could be useful for making sure we have annotated models
        # with the delays, coupling variables etc.

        # define some URIs for things we need

        # use this URI to identify delayed variables - not the perfect URI, but will do for now
        #     This is actually "delayed differential equation model" from the MAMO ontology
        #delay_variable_uri = URIRef('http://identifiers.org/mamo/MAMO_0000089')
        self.delay_variable_uri = URIRef('https://github.com/nickerso/libcellml-python-utils/properties.rst#delay-variable')
        self.variable_to_delay_uri = URIRef('https://github.com/nickerso/libcellml-python-utils/properties.rst#variable-to-delay')
        self.independent_variable_uri = URIRef('https://github.com/nickerso/libcellml-python-utils/properties.rst#variable-to-delay')
        self.delay_amount_uri = URIRef('https://github.com/nickerso/libcellml-python-utils/properties.rst#delay-amount')

        # use this for some random thing that I want to define - http://example.com is a good base for things that will never resolve
        self.stuff_uri = URIRef('http://example.com/cool-thing#21')

        # a "readout" variable that we maybe want to connect to something external?
        self.timecourse_readout_uri = URIRef('http://identifiers.org/mamo/MAMO_0000031')
        self.output_variable_uri = URIRef('http://identifiers.org/mamo/MAMO_0000018')
        self.input_variable_uri = URIRef('http://identifiers.org/mamo/MAMO_0000017')

    def annotate_cellml(self):
        # TODO most of this function 
        # isn't needed for the generation anymore, but is it good to keep annotation in here? Probably.
        annotator = Annotator()
        annotator.setModel(self.cellml_model)

        if annotator.assignAllIds():
            print('Some entities have been assigned an ID, you should save the model!')
        else:
            print('Everything already had an ID.')

        duplicates = annotator.duplicateIds()
        if len(duplicates) > 0:
            print("There are some duplicate IDs, behaviour may be unreliable...")
            print(duplicates)

        # blow away all the IDs and reassign them
        annotator.clearAllIds()
        annotator.assignAllIds()

        model_string = cellml.print_model(self.cellml_model)
        print(model_string)
        
        # and save the updated model to a new file
        # - note, we need the model filename for making our annotations later
        with open(self.annotated_model_file_path, 'w') as f:
            f.write(model_string)
        
        # get the ID of the variables we want to annotate
        # The below for loops are the only part of annotate_cellml that is needed to run generate_cpp
        for vessel_tup in self.model.vessels_df.itertuples():
            if "delay_info" in vessel_tup._fields:
                if vessel_tup.delay_info is not "None":
                    for II in range(len(vessel_tup.delay_info["variables_to_delay"])):
                        variable_to_delay = vessel_tup.delay_info["variables_to_delay"][II]
                        delayed_variable = vessel_tup.delay_info["delayed_variables"][II]
                        delay_amount = vessel_tup.delay_info["delay_amounts"][II]
                        
                        self.variables_to_delay.append(self.cellml_model.component(vessel_tup.name \
                                                                        ).variable(variable_to_delay).id())
                        self.delayed_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                        ).variable(delayed_variable).id())
                        self.delay_amounts.append(self.cellml_model.component(vessel_tup.name \
                                                                        ).variable(delay_amount).id())
                        # TODO make functionality to delay from other variables?
                        self.independent_variables.append(self.cellml_model.component("environment" \
                                                                        ).variable('time').id())

        for vessel_tup in self.model.vessels_df.itertuples():
            for out_vessel in vessel_tup.out_vessels:
                out_vessel_row = self.model.vessels_df.loc[self.model.vessels_df["name"] == out_vessel]
                if out_vessel_row["vessel_type"] == "FV1D_vessel":
                    # get the index from the out_vessel name which is of format "FV1D_##" or FV1D_###"
                    fv1d_vessel_index = re.findall(r'\d+$', out_vessel)[0]
                    if len(fv1d_vessel_index) != 1:
                        print("ERROR: FV1D vessel idx is not written correctly",
                              "should be of format FV1D_# or FV1D_##, etc")
                        exit()
                    self.connection_vessel_indices.append(fv1d_vessel_index)
                    for exit_port in vessel_tup.exit_ports:
                        if exit_port["port_type"] == "vessel_port":
                            if vessel_tup.BC_type[1] == 'v':
                                self.port_input_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                ).variable(exit_port["variables"][0]).id())
                                self.port_output_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                ).variable(exit_port["variables"][1]).id())
                            elif vessel_tup.BC_type[1] == 'p': 
                                self.port_input_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                ).variable(exit_port["variables"][1]).id())
                                self.port_output_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                ).variable(exit_port["variables"][0]).id())
                            else:
                                print("unknown BC type of {vessel_tup.BC_type} connecting to a "
                                      "1D FV model")
                
            for inp_vessel in vessel_tup.inp_vessels:
                inp_vessel_row = self.model.vessels_df.loc[self.model.vessels_df["name"] == out_vessel]
                if inp_vessel_row["vessel_type"] == "FV1D_vessel":
                    fv1d_vessel_index = re.findall(r'\d+$', out_vessel)[0]
                    if len(fv1d_vessel_index) != 1:
                        print("ERROR: FV1D vessel idx is not written correctly",
                              "should be of format FV1D_# or FV1D_##, etc")
                        exit()
                    self.connection_vessel_indices.append(fv1d_vessel_index)
                    for entrance_port in vessel_tup.entrance_ports:
                        if entrance_port["port_type"] == "vessel_port":
                            if vessel_tup.BC_type[0] == 'v':
                                self.port_input_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                ).variable(exit_port["variables"][0]).id())
                                self.port_output_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                ).variable(exit_port["variables"][1]).id())
                            elif vessel_tup.BC_type[0] == 'p': 
                                self.port_input_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                ).variable(exit_port["variables"][1]).id())
                                self.port_output_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                ).variable(exit_port["variables"][0]).id())
                            else:
                                print("unknown BC type of {vessel_tup.BC_type} connecting to a "
                                      "1D FV model")

        # input_variables = [model.component(vessel_tup.name).variable('v_in').id()]
        # output_variables = [model.component(vessel_tup.name).variable('u').id()]

        g = Graph()


        # Create an RDF URI node for our variable to use as the subject for multiple triples
        # note: we are going to serialise the RDF graph into the same folder, so we need a URI that is relative to the intended file
        variable_to_delay_name_uri = [URIRef(f'{self.filename_prefix_with_ids}.cellml' + 
                                             '#' + variable) for variable in self.variables_to_delay]
        delay_variable_name_uri = [URIRef(f'{self.filename_prefix_with_ids}.cellml' + 
                                          '#' + variable) for variable in self.delayed_variables]
        delay_amount_name_uri = [URIRef(f'{self.filename_prefix_with_ids}.cellml' + 
                                        '#' + delay_amount) for delay_amount in self.delay_amounts]
        independent_variable_name_uri = [URIRef(f'{self.filename_prefix_with_ids}.cellml' + 
                                                '#' + variable) for variable in self.independent_variables]

        port_input_variable_name_uri = [URIRef(f'{self.filename_prefix_with_ids}.cellml' + 
                                               '#' + variable) for variable in port_input_variables]
        port_output_variable_name_uri = [URIRef(f'{self.filename_prefix_with_ids}.cellml' + 
                                                '#' + variable) for variable in port_output_variables]

        input_variable_name_uri = [URIRef(f'{self.filename_prefix_with_ids}.cellml' + 
                                          '#' + variable) for variable in input_variables]
        output_variable_name_uri = [URIRef(f'{self.filename_prefix_with_ids}.cellml' + 
                                           '#' + variable) for variable in output_variables]

        # Add triples using store's add() method.
        # We're using the Dublin Core term "type" to associate the variable with the delay...
        for II in range(len(variable_to_delay_name_uri)):
            g.add((variable_to_delay_name_uri[II], DCTERMS.type, self.variable_to_delay_uri))
            g.add((variable_to_delay_name_uri[II], self.delay_variable_uri, delay_variable_name_uri[II]))
            g.add((variable_to_delay_name_uri[II], self.independent_variable_uri, independent_variable_name_uri[II]))
            g.add((variable_to_delay_name_uri[II], self.delay_amount_uri, delay_amount_name_uri[II]))
        # Set coupling variables
        for II in range(len(port_input_variables)):
            # for this input variable, add the output variable as a coupling variable
            g.add((port_input_variable_name_uri[II], DCTERMS.type, self.input_variable_uri))
            g.add((port_input_variable_name_uri[II], self.output_variable_uri, port_output_variable_name_uri[II]))
        
        for II in range(len(output_variables)):
            g.add((output_variable_name_uri[II], DCTERMS.type, self.output_variable_uri))
        for II in range(len(input_variables)):
            g.add((input_variable_name_uri[II], DCTERMS.type, self.input_variable_uri))

        # print all the data in the turtle format
        print(g.serialize(format='ttl'))

        # and save to a file
        with open(os.path.join(self.generated_model_subdir, self.filename_prefix_with_ids + '--annotations.ttl'), 'w') as f:
            f.write(g.serialize(format='ttl'))


    def generate_cellml(self):
        print("generating CellML files, before Cpp generation")jkkkkkkkkkkkkkkkkk
        cellml_generator = CVS0DCellMLGenerator(self.model, self.generated_model_subdir, self.filename_prefix)
        cellml_generator.generate_files()

    def generate_cpp(self):
        print("now generating Cpp files")

        print('do something here, exiting for now')
        exit()

class CVS1DCppGenerator(object):
    '''
    Generates Cpp files for 1D model. This is a wrapper around Lucas Muller's code.

    WARNING: THIS CPP 1D CODE IS CURRENTLY NOT OPEN SOURCE, SO THIS CLASS WON'T WORK UNTIL IT IS MADE OPEN SOURCE.
    '''


    def __init__(self, model, output_path, filename_prefix):
        '''
        Constructor
        '''
        self.model = model
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        self.filename_prefix = filename_prefix
        self.user_resources_path = os.path.join(generators_dir_path, '../../resources')

    
    def generate_files(self):
        print("generating 1D Cpp files")

class CVSCoupledCppGenerator(object):
    '''
    Generates Cpp files for coupled 0D and 1D models.
    '''
    
    def __init__(self, model, output_path, filename_prefix):
        '''
        Constructor
        '''
        self.model = model
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        self.filename_prefix = filename_prefix
        self.user_resources_path = os.path.join(generators_dir_path, '../../resources')
    
    def generate_files(self):
        zeroD_generator = CVS0DCppGenerator(self.model, self.output_path, self.filename_prefix)
        zeroD_generator.generate_files()

        # currently we don't generate the 1D model, we assume it is already generated.
        # oneD_generator = CVS1DCppGenerator(self.model, self.output_path, self.filename_prefix)
        # oneD_generator.generate_files()
        print("now do the coupling")

