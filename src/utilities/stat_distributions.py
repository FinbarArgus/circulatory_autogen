import numpy as np

def gaussian(x,a,mu,sigma):
    return a*np.exp(-(x-mu)**2/(2*sigma**2))

def gaussian_jacobian(x, a, mu, sigma):
    d_da = np.exp(-(x-mu)**2/(2*sigma**2))
    d_dmu = a*((x-mu)/sigma**2)*np.exp(-(x-mu)**2/(2*sigma**2))
    d_dsigma = a*((x-mu)/sigma**3)*np.exp(-(x-mu)**2/(2*sigma**2))
    return np.array([d_da, d_dmu, d_dsigma])

def gaussian_d2_dx2(x, a, mu, sigma):
    return (-a/sigma**2 + a*(x-mu)**2/sigma**4)*np.exp(-(x-mu)**2/(2*sigma**2))
