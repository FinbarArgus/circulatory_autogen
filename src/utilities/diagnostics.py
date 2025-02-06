import numpy as np
from scipy.stats import normaltest

# This class was taken from https://github.com/Samreay/ChainConsumer and modified


class Diagnostics(object):
    def __init__(self):
        pass

    def gelman_rubin(self, chain, threshold=0.05):
        r""" Runs the Gelman Rubin diagnostic on the supplied chains.
        Parameters
        ----------
        chain : int|str, optional
            Which chain to run the diagnostic on. By default, this is `None`,
            which will run the diagnostic on all chains. You can also
            supply and integer (the chain index) or a string, for the chain
            name (if you set one).
        threshold : float, optional
            The maximum deviation permitted from 1 for the final value
            :math:`\hat{R}`
        Returns
        -------
        float
            whether or not the chains pass the test
        Notes
        -----
        I follow PyMC in calculating the Gelman-Rubin statistic, where,
        having :math:`m` chains of length :math:`n`, we compute
        .. math::
            B = \frac{n}{m-1} \sum_{j=1}^{m} \left(\bar{\theta}_{.j} - \bar{\theta}_{..}\right)^2
            W = \frac{1}{m} \sum_{j=1}^{m} \left[ \frac{1}{n-1} \sum_{i=1}^{n} \left( \theta_{ij} - \bar{\theta_{.j}}\right)^2 \right]
        where :math:`\theta` represents each model parameter. We then compute
        :math:`\hat{V} = \frac{n_1}{n}W + \frac{1}{n}B`, and have our convergence ratio
        :math:`\hat{R} = \sqrt{\frac{\hat{V}}{W}}`. We check that for all parameters,
        this ratio deviates from unity by less than the supplied threshold.
        """

        ## TODO test this function
        print('This Gelman Rubin function hasnt been tested yet, verify that the function is working properly'
              'before use.')

        num_steps = chain.shape[0]
        num_walkers = chain.shape[1]
        num_params = chain.shape[2]
        chains = np.split(chain, num_walkers)
        assert num_walkers > 1, "Cannot run Gelman-Rubin statistic with only one walker"
        m = 1.0 * len(chains)
        n = 1.0 * chains[0].shape[0]
        all_mean = np.mean(chain, axis=0)
        chain_means = np.array([np.mean(c, axis=0) for c in chains])
        chain_var = np.array([np.var(c, axis=0, ddof=1) for c in chains])
        b = n / (m - 1) * ((chain_means - all_mean) ** 2).sum(axis=0)
        w = (1 / m) * chain_var.sum(axis=0)
        var = (n - 1) * w / n + b / n
        v = var + b / (n * m)
        R = np.sqrt(v / w)

        passed = np.abs(R - 1) < threshold
        print("Gelman-Rubin Statistic values for chain s")
        # for p, v, pas in zip(parameters, R, passed):
        #     param = "Param %d" % p if isinstance(p, int) else p
        #     print("%s: %7.5f (%s)" % (param, v, "Passed" if pas else "Failed"))
        return np.all(passed)

    def geweke(self, chain, first=0.1, last=0.5, threshold=0.05):
        """ Runs the Geweke diagnostic on the supplied chains.
        Parameters
        ----------
        chain : int|str, optional
            Which chain to run the diagnostic on. By default, this is `None`,
            which will run the diagnostic on all chains. You can also
            supply and integer (the chain index) or a string, for the chain
            name (if you set one).
        first : float, optional
            The amount of the start of the chain to use
        last : float, optional
            The end amount of the chain to use
        threshold : float, optional
            The p-value to use when testing for normality.
        Returns
        -------
        float
            whether or not the chains pass the test
        """

        num_steps = chain.shape[0]
        num_walkers = chain.shape[1]
        num_params = chain.shape[2]
        assert num_walkers is not None and num_walkers > 0, "You need to specify the number of walkers to use the Geweke diagnostic."
        n_start = int(np.floor(first * num_steps))
        n_end = int(np.floor((1 - last) * num_steps))
        mean_start = np.array([np.mean(chain[:n_start, II, JJ]) for II in range(num_walkers) for JJ in range(num_params)])
        # TODO change back to the oneline
        var_start = []
        for II in range(num_walkers):
            for JJ in range(num_params):
                var_start.append(self._spec(chain[:n_start, II, JJ])/chain[:n_start, II, JJ].size)
        var_start = np.array(var_start)
        # var_start = np.array([self._spec(chain[:n_start, II, JJ]) / chain[:n_start, II, JJ].size for II in range(num_walkers)
        #                       for JJ in range(num_params)])
        mean_end = np.array([np.mean(chain[n_end:, II, JJ]) for II in range(num_walkers) for JJ in range(num_params)])
        var_end = np.array([self._spec(chain[n_end:, II, JJ]) / chain[n_end:, II, JJ].size for II in range(num_walkers)
                            for JJ in range(num_params)])
        zs = (mean_start - mean_end) / (np.sqrt(var_start + var_end))
        _, pvalue = normaltest(zs)
        print("Geweke Statistic for chain has p-value %e" % (pvalue))
        return pvalue > threshold

    # Method of estimating spectral density following PyMC.
    # See https://github.com/pymc-devs/pymc/blob/master/pymc/diagnostics.py
    def _spec(self, x, order=2):
        from statsmodels.regression.linear_model import yule_walker

        beta, sigma = yule_walker(x, order)
        return sigma ** 2 / (1.0 - np.sum(beta)) ** 2