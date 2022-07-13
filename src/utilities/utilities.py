import numpy as np
import math

class Normalise_class:
    def __init__(self, param_mins, param_maxs, mod_first_variables=0, modVal = 1.0):
        self.param_mins = param_mins
        self.param_maxs = param_maxs
        self.mod_first_variables=mod_first_variables
        self.modVal = modVal

    def normalise(self, x):
        xDim = len(x.shape)
        if xDim == 1:
            y = (x - self.param_mins)/(self.param_maxs - self.param_mins)
        elif xDim == 2:
            y = (x - self.param_mins.reshape(-1, 1))/(self.param_maxs.reshape(-1, 1) - self.param_mins.reshape(-1, 1))
        elif xDim == 3:
            y = ((x.reshape(x.shape[0], -1) - self.param_mins.reshape(-1, 1)) /
                 (self.param_maxs.reshape(-1, 1) - self.param_mins.reshape(-1, 1))).reshape(x.shape[0], x.shape[1],
                                                                                            x.shape[2])
        else:
            print('normalising not set up for xDim = {}, exiting'.format(xDim))
            exit()

        return y

    def unnormalise(self, x):
        xDim = len(x.shape)
        if xDim == 1:
            y = x * (self.param_maxs - self.param_mins) + self.param_mins
        elif xDim == 2:
            y = x * (self.param_maxs.reshape(-1, 1) - self.param_mins.reshape(-1, 1)) + self.param_mins.reshape(-1, 1)
        elif xDim == 3:
            y = (x.reshape(x.shape[0], -1)*(self.param_maxs.reshape(-1, 1) - self.param_mins.reshape(-1, 1)) +
                 self.param_mins.reshape(-1, 1)).reshape(x.shape[0], x.shape[1], x.shape[2])
        else:
            print('normalising not set up for xDim = {}, exiting'.format(xDim))
            exit()
        return y


def obj_to_string(obj, extra='    '):
    return str(obj.__class__) + '\n' + '\n'.join(
        (extra + (str(item) + ' = ' +
                  (obj_to_string(obj.__dict__[item], extra + '    ') if hasattr(obj.__dict__[item], '__dict__') else str(
                      obj.__dict__[item])))
         for item in sorted(obj.__dict__)))

# The below are for mcmc convergence obtained from https://emcee.readthedocs.io/en/stable/tutorials/autocorr/
def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf

# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

# Following the suggestion from Goodman & Weare (2010)
def autocorr_gw2010(y, c=5.0):
    f = autocorr_func_1d(np.mean(y, axis=1))
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]


def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[0])
    for II in range(y.shape[1]):
        f += autocorr_func_1d(y[:,II])
    f /= y.shape[1]
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]









