'''
Created on 01/11/2021
@summary: Defines check routines for a lumped parameter model.
@author: Finbar Argus, Gonzalo D. Maso Talou
'''

from abc import ABC, abstractmethod

class AbstractLumpedCheck(ABC):
    '''
    Abstract class of a check condition
    '''
    
    @abstractmethod
    def execute(self,model_0D):
        '''
        Runs the check activities
        :param model_0D: model to be checked.
        '''


class LumpedCompositeCheck(AbstractLumpedCheck):
    '''
    Class that performs multiple checks on a lumped parameter model
    '''

    def __init__(self, check_list=[]):
        """
        Constructor
        
        :param check_list: initial checking list.
        """
        self.checks_list = check_list

    def add_check(self,check):
        '''
        Adds an additional check to the checking list.
        :param check: check to be added.
        '''
        self.checks_list.append(check)
        
    def execute(self, model_0D):
        '''
        Executes all the checks in the checking list.
        :param model_0D:
        '''
        for current_check in self.checks_list:
            current_check.execute(model_0D)


class LumpedBCVesselCheck(AbstractLumpedCheck):
    '''
    Checks if the boundary conditions and vessel types are correct.
    '''

    def execute(self, model_0D):
        '''
        Executes all check activities.
        :param model_0D: model to be checked.
        '''
        model_0D.vessels_df.apply(self.execute_for_row, args=(model_0D, ), axis=1)

    def execute_for_row(self, vessel_row, model_0D):
        if (vessel_row["vessel_type"], vessel_row["BC_type"]) not in model_0D.possible_vessel_BC_types:
            print(f'vessel_type, BC_type combo of ({vessel_row["vessel_type"]}, {vessel_row["BC_type"]}) '
                  f'is not allowed for vessel {vessel_row["name"]}')
            exit()

class LumpedIDParamsCheck(AbstractLumpedCheck):
    '''
    Checks if the boundary conditions and vessel types are correct.
    '''

    def execute(self, model_0D):
        '''
        Executes all check activities.
        :param model_0D: model to be checked.
        '''
        for const_name, _ in model_0D.param_id_consts:
            if not const_name in model_0D.parameters_array['variable_name']:
                print(f'ERROR parameter id constant of {const_name} is not defined in the parameters file')
                exit()


#    Example of usage
#    Contructs a multiple check with only a LumpedBCVesselCheck 
#model_checks = LumpedCompositeCheck(check_list=[LumpedBCVesselCheck()])
#    Executes all checks in the model named model_0D
#model_checks.execute(model_0D)
