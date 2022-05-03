import numpy as np
from param_id.paramID import CVS0DParamID

class SequentialParamID:
    """
    This class contains a param_id object that can be run multiple times
    to reduce the parameter set to ensure identifiability.
    """

    def __init__(self, model_path, param_id_model_type, param_id_method, file_name_prefix,
                 input_params_path=None,  sensitivity_params_path=None, num_calls_to_function=1000,
                 param_id_obs_path=None, sim_time=2.0, pre_time=20.0, maximumStep=0.0001, dt=0.01,
                 DEBUG=False):

        self.model_path = model_path
        self.param_id_model_type = param_id_model_type
        self.param_id_method = param_id_method
        self.file_name_prefix = file_name_prefix
        self.input_params_path = input_params_path
        self.sensitivity_params_path = sensitivity_params_path
        self.num_calls_to_function = num_calls_to_function
        self.param_id_obs_path = param_id_obs_path
        self.sim_time = sim_time
        self.pre_time = pre_time
        self.maximumStep = maximumStep
        self.DEBUG =DEBUG


        self.param_id = CVS0DParamID(model_path, param_id_model_type, param_id_method, False, file_name_prefix,
                                input_params_path=input_params_path,
                                sensitivity_params_path=sensitivity_params_path,
                                param_id_obs_path=param_id_obs_path,
                                sim_time=sim_time, pre_time=pre_time, maximumStep=maximumStep, DEBUG=DEBUG)


        self.param_id.set_genetic_algorithm_parameters(num_calls_to_function)
        self.best_param_vals = None
        self.best_param_names = None

        # thresholds for identifiability TODO optimise these
        self.threshold_param_importance = 0.01
        self.keep_threshold_param_importance = 0.8
        self.threshold_collinearity = 20
        self.threshold_collinearity_pairs = 10
        self.second_deriv_threshold = -500

    def run(self):
        identifiable = False
        while not identifiable:
            self.param_id.run()
            self.param_id.run_sensitivity(None)

            self.best_param_vals = self.param_id.get_best_param_vals()
            self.param_names = self.param_id.get_param_names()

            param_importance = self.param_id.get_param_importance()
            collinearity_index = self.param_id.get_collinearity_index()
            collinearity_index_pairs = self.param_id.get_collinearity_index_pairs()

            if min(param_importance) > self.threshold_param_importance and \
                        max(collinearity_index) < self.threshold_collinearity:
                print(f'The model is identifiable with {len(self.param_names)} parameters:')
                print(self.param_names)
                identifiable = True
            else:
                # remove parameters that aren't identifiable
                # and update param_id object
                print(f'The model is NOT identifiable with {len(self.param_names)} parameters')
                print(f'determining which parameters to remove from identifying set')
                param_idxs_to_remove = []
                for II in range(len(self.param_names)):
                    param_name = self.param_names[II]
                    if param_importance[II] < self.threshold_param_importance:
                        param_idxs_to_remove.append(II)

                    for JJ in range(len(self.param_names)):
                        if collinearity_index_pairs[II, JJ] > self.threshold_collinearity_pairs:
                            if param_importance[II] < param_importance[JJ]:
                                param_idxs_to_remove.append(II)

                if len(param_idxs_to_remove) > 1:
                    # make sure we aren't removing important parameters
                    # it is better to remove too few than too many
                    for idx in param_idxs_to_remove:
                        # TODO this doesn't allow us to remove linearly related params if they are both important
                        #  Fix this!
                        if param_importance[idx] > self.keep_threshold_param_importance:
                            param_idxs_to_remove.remove(idx)

                # TODO future work: if we are reformulating the equations we will need to create and run a
                #  CVS0DCellMLGenerator object.
            # TODO remove this break
            break

        best_param_vals = self.param_id.get_best_param_vals()
        self.param_id.close_simulation()

        # Now run mcmc to check practical identifiability
        mcmc = CVS0DParamID(self.model_path, self.param_id_model_type, self.param_id_method, True,
                                 self.file_name_prefix,
                                 input_params_path=self.input_params_path,
                                 sensitivity_params_path=self.sensitivity_params_path,
                                 param_id_obs_path=self.param_id_obs_path,
                                 sim_time=self.sim_time, pre_time=self.pre_time, maximumStep=self.maximumStep,
                                 DEBUG=self.DEBUG)

        mcmc.set_best_param_vals(best_param_vals)
        mcmc.run_mcmc()
        mcmc.plot_mcmc()
        mcmc.calculate_mcmc_identifiability(second_deriv_threshold=self.second_deriv_threshold)

        # smooth mcmc

    def get_best_param_names(self):
        return self.best_param_names

    def get_best_param_vals(self):
        return self.best_param_vals


