'''
Created on 01/11/2021

@author: Finbar Argus, Gonzalo D. Maso Talou
'''

class LumpedChecks():
    '''
    Class for checking lumped parameter models
    '''

    def __init__(self):
        """
        constructor
        """
        self.checks_list = []
        self.checks_list.append(self.__check_BC_types_and_vessel_types)

    def check_all(self, model_0D):
        for check in self.checks_list:
            check(model_0D)

    def __check_BC_types_and_vessel_types(self, model_0D):
        for vessel_vec in model_0D.vessels:
            if vessel_vec['BC_type'] not in model_0D.possible_BC_types:
                print(f'BC_type of {vessel_vec["BC_type"]} is not allowed for vessel {vessel_vec["name"]}')
            if vessel_vec['vessel_type'] not in model_0D.possible_vessel_types:
                print(f'vessel_type of {vessel_vec["BC_type"]} is not allowed for vessel {vessel_vec["name"]}')
