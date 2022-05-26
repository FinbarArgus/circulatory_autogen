import numpy as np
from param_id.paramID import CVS0DParamID
from mpi4py import MPI

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
        self.threshold_param_importance = 0.1
        self.keep_threshold_param_importance = 0.8
        self.threshold_collinearity = 20
        self.threshold_collinearity_pairs = 10
        self.second_deriv_threshold = -1000

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.num_procs = self.comm.Get_size()

    def run(self):

        buff = np.array([False])
        identifiable = buff[0]
        num_params_to_remove_buff = np.array([0])
        while not identifiable:

            # self.param_id.temp_test()
            # self.param_id.temp_test2()

            self.param_id.run()
            if self.rank == 0:
                self.param_id.run_single_sensitivity(None)

                self.best_param_vals = self.param_id.get_best_param_vals().copy()
                self.param_names = self.param_id.get_param_names().copy()

                param_importance = self.param_id.get_param_importance().copy()
                collinearity_index = self.param_id.get_collinearity_index().copy()
                collinearity_index_pairs = self.param_id.get_collinearity_index_pairs().copy()

                if min(param_importance) > self.threshold_param_importance and \
                            max(collinearity_index_pairs) < self.threshold_collinearity_pairs:
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
                            identifiable = False
                        else:
                            for JJ in range(len(self.param_names)):
                                if collinearity_index_pairs[II, JJ] > self.threshold_collinearity_pairs:
                                    if param_importance[II] < param_importance[JJ]:
                                        param_idxs_to_remove.append(II)
                                        identifiable = False
                        if identifiable:
                            print('error, not identifiable, but no params to remove added.')
                            exit()

                    if len(param_idxs_to_remove) > 1:
                        # TODO make sure we aren't removing important parameters
                        # it is better to remove too few than too many
                        # for idx in param_idxs_to_remove:
                        #     # TODO this doesn't allow us to remove linearly related params if they are both important
                        #     #  Fix this!
                        #     if param_importance[idx] > self.keep_threshold_param_importance:
                        #         param_idxs_to_remove.remove(idx)
                        pass

                    # TODO future work: if we are reformulating the equations we will need to create and run a
                    #  CVS0DCellMLGenerator object.
            buff[0] = identifiable
            self.comm.Bcast(buff, root=0)
            identifiable = buff[0]
            self.comm.Bcast(num_params_to_remove_buff, root=0)
            if rank == 0:
                param_idxs_to_remove_array = np.array(param_idxs_to_remove)
            else:
                param_idxs_to_remove_array = np.empty(num_params_to_remove_buff[0], dtype=int)

            param_id.remove_params_from_idx(param_idxs_to_remove_array)



        self.best_param_vals = self.param_id.get_best_param_vals().copy()
        self.param_id.close_simulation()
        
        buff = np.array([False])
        identifiable = buff[0]
        while not identifiable:

            # Now run mcmc to check practical identifiability
            mcmc = CVS0DParamID(self.model_path, self.param_id_model_type, self.param_id_method, True,
                                     self.file_name_prefix,
                                     input_params_path=self.input_params_path,
                                     sensitivity_params_path=self.sensitivity_params_path,
                                     param_id_obs_path=self.param_id_obs_path,
                                     sim_time=self.sim_time, pre_time=self.pre_time, maximumStep=self.maximumStep,
                                     DEBUG=self.DEBUG)

            mcmc.set_best_param_vals(self.best_param_vals)
            mcmc.run_mcmc()
            mcmc.plot_mcmc()
            second_deriv, identifiable_param_array = mcmc.calculate_mcmc_identifiability(second_deriv_threshold=self.second_deriv_threshold)
            if np.any(identifiable_param_array):
                remove_idx = np.argmin(second_deriv)
                identifiable = False
            else:
                identifiable = True

            buff[0] = identifiable
            self.comm.Bcast(buff, root=0)
            identifiable = buff[0]

    print('parameter identification complete. Model parameters are identifiable')
    print('plotting core predictions distribution to check uncertainty on predictions')
    # TODO sample param distribution to get distribution of core
    # prediction. Use prediction_variables.csv for this

    def get_best_param_names(self):
        return self.best_param_names

    def get_best_param_vals(self):
        return self.best_param_vals


