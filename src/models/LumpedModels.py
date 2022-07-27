'''
Created on 29/10/2021

@author: Gonzalo D. Maso Talou, Finbar Argus
'''

class CVS0DModel(object):
    '''
    Representation of a 0D cardiovascular model
    '''

    def __init__(self, vessels_df, parameters_array,
                 param_id_states=None, param_id_consts=None, param_id_date=None,
                 all_parameters_defined=False):
        '''
        Constructor
        '''
        self.vessels_df = vessels_df
        self.parameters_array = parameters_array
        self.all_parameters_defined = all_parameters_defined
        self.param_id_states = param_id_states
        self.param_id_consts = param_id_consts
        self.param_id_date = param_id_date

        # possible vessel and BC attributes of the model
        # TODO get these names from the modules_config.json file
        self.possible_vessel_types = list(set(self.vessels_df["vessel_type"].to_list()))
        self.possible_BC_types = list(set(self.vessels_df["BC_type"].to_list()))

    
    
        
