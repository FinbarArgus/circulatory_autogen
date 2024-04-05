import numpy as np
import os
import sys
import sympy
from scipy.signal import find_peaks

# decorator for functions that turn a series into a constant
# Needed if you want to plot the series ontop of estimated constants
def series_to_constant(func):
    func.series_to_constant = True
    return func

# example function
def ml_to_m3(x):
    return x*1e-6

def RICRI_get_pole_freq_1_Hz(C_T, I_1, I_2, R_T, frac_R_T_1_of_R_T):
    assert isinstance(C_T, float)
    assert isinstance(I_1, float)
    assert isinstance(I_2, float)
    assert isinstance(R_T, float)
    assert isinstance(frac_R_T_1_of_R_T, float)
    R_1 = R_T*frac_R_T_1_of_R_T
    R_2 = R_T*(1.0-frac_R_T_1_of_R_T)
    # this frequency was calculated using sympy from the differential equation for an RICRI terminal
    freq = np.float(0.159154943091895*sympy.re(sympy.Abs(R_2/(2*I_2) + sympy.sqrt(C_T*(C_T*R_2**2 - 4*I_2))/(2*C_T*I_2)).evalf()))
    assert isinstance(freq, float)
    return freq

def RICRI_get_pole_freq_2_Hz(C_T, I_1, I_2, R_T, frac_R_T_1_of_R_T):
    assert isinstance(C_T, float)
    assert isinstance(I_1, float)
    assert isinstance(I_2, float)
    assert isinstance(R_T, float)
    assert isinstance(frac_R_T_1_of_R_T, float)
    R_1 = R_T*frac_R_T_1_of_R_T
    R_2 = R_T*(1.0-frac_R_T_1_of_R_T)
    # this frequency was calculated using sympy from the differential equation for an RICRI terminal
    freq = np.float(0.159154943091895*sympy.re(sympy.Abs(R_2/(2*I_2) - sympy.sqrt(C_T*(C_T*R_2**2 - 4*I_2))/(2*C_T*I_2)).evalf()))
    assert isinstance(freq, float)
    return freq

def RICRI_get_zero_freq_1_Hz(C_T, I_1, I_2, R_T, frac_R_T_1_of_R_T):
    assert isinstance(C_T, float)
    assert isinstance(I_1, float)
    assert isinstance(I_2, float)
    assert isinstance(R_T, float)
    assert isinstance(frac_R_T_1_of_R_T, float)
    R_1 = R_T*frac_R_T_1_of_R_T
    R_2 = R_T*(1.0-frac_R_T_1_of_R_T)
    # this frequency was calculated using sympy from the differential equation for an RICRI terminal
    freq = np.float(0.159154943091895*sympy.re(sympy.Abs(((I_1*R_2 + I_2*R_1)**2/(I_1**2*I_2**2) - 3*(C_T*R_1*R_2 + I_1 + I_2)/(C_T*I_1*I_2))/(3*(sympy.sqrt(-4*((I_1*R_2 + I_2*R_1)**2/(I_1**2*I_2**2) - 3*(C_T*R_1*R_2 + I_1 + I_2)/(C_T*I_1*I_2))**3 + (2*(I_1*R_2 + I_2*R_1)**3/(I_1**3*I_2**3) + 27*(R_1 + R_2)/(C_T*I_1*I_2) - 9*(I_1*R_2 + I_2*R_1)*(C_T*R_1*R_2 + I_1 + I_2)/(C_T*I_1**2*I_2**2))**2)/2 + (I_1*R_2 + I_2*R_1)**3/(I_1**3*I_2**3) + 27*(R_1 + R_2)/(2*C_T*I_1*I_2) - 9*(I_1*R_2 + I_2*R_1)*(C_T*R_1*R_2 + I_1 + I_2)/(2*C_T*I_1**2*I_2**2))**(1/3)) + (sympy.sqrt(-4*((I_1*R_2 + I_2*R_1)**2/(I_1**2*I_2**2) - 3*(C_T*R_1*R_2 + I_1 + I_2)/(C_T*I_1*I_2))**3 + (2*(I_1*R_2 + I_2*R_1)**3/(I_1**3*I_2**3) + 27*(R_1 + R_2)/(C_T*I_1*I_2) - 9*(I_1*R_2 + I_2*R_1)*(C_T*R_1*R_2 + I_1 + I_2)/(C_T*I_1**2*I_2**2))**2)/2 + (I_1*R_2 + I_2*R_1)**3/(I_1**3*I_2**3) + 27*(R_1 + R_2)/(2*C_T*I_1*I_2) - 9*(I_1*R_2 + I_2*R_1)*(C_T*R_1*R_2 + I_1 + I_2)/(2*C_T*I_1**2*I_2**2))**(1/3)/3 + (I_1*R_2 + I_2*R_1)/(3*I_1*I_2)).evalf()))
    assert isinstance(freq, float)
    return freq

def RICRI_get_zero_freq_2_Hz(C_T, I_1, I_2, R_T, frac_R_T_1_of_R_T):
    assert isinstance(C_T, float)
    assert isinstance(I_1, float)
    assert isinstance(I_2, float)
    assert isinstance(R_T, float)
    assert isinstance(frac_R_T_1_of_R_T, float)
    R_1 = R_T*frac_R_T_1_of_R_T
    R_2 = R_T*(1.0-frac_R_T_1_of_R_T)
    # this frequency was calculated using sympy from the differential equation for an RICRI terminal
    freq = np.float(0.159154943091895*sympy.re(sympy.Abs(-((I_1*R_2 + I_2*R_1)**2/(I_1**2*I_2**2) - 3*(C_T*R_1*R_2 + I_1 + I_2)/(C_T*I_1*I_2))/(3*(-1/2 + sympy.sqrt(3)*sympy.I/2)*(sympy.sqrt(-4*((I_1*R_2 + I_2*R_1)**2/(I_1**2*I_2**2) - 3*(C_T*R_1*R_2 + I_1 + I_2)/(C_T*I_1*I_2))**3 + (2*(I_1*R_2 + I_2*R_1)**3/(I_1**3*I_2**3) + 27*(R_1 + R_2)/(C_T*I_1*I_2) - 9*(I_1*R_2 + I_2*R_1)*(C_T*R_1*R_2 + I_1 + I_2)/(C_T*I_1**2*I_2**2))**2)/2 + (I_1*R_2 + I_2*R_1)**3/(I_1**3*I_2**3) + 27*(R_1 + R_2)/(2*C_T*I_1*I_2) - 9*(I_1*R_2 + I_2*R_1)*(C_T*R_1*R_2 + I_1 + I_2)/(2*C_T*I_1**2*I_2**2))**(1/3)) - (-1/2 + sympy.sqrt(3)*sympy.I/2)*(sympy.sqrt(-4*((I_1*R_2 + I_2*R_1)**2/(I_1**2*I_2**2) - 3*(C_T*R_1*R_2 + I_1 + I_2)/(C_T*I_1*I_2))**3 + (2*(I_1*R_2 + I_2*R_1)**3/(I_1**3*I_2**3) + 27*(R_1 + R_2)/(C_T*I_1*I_2) - 9*(I_1*R_2 + I_2*R_1)*(C_T*R_1*R_2 + I_1 + I_2)/(C_T*I_1**2*I_2**2))**2)/2 + (I_1*R_2 + I_2*R_1)**3/(I_1**3*I_2**3) + 27*(R_1 + R_2)/(2*C_T*I_1*I_2) - 9*(I_1*R_2 + I_2*R_1)*(C_T*R_1*R_2 + I_1 + I_2)/(2*C_T*I_1**2*I_2**2))**(1/3)/3 - (I_1*R_2 + I_2*R_1)/(3*I_1*I_2)).evalf()))
    assert isinstance(freq, float)
    return freq

def RICRI_get_zero_freq_3_Hz(C_T, I_1, I_2, R_T, frac_R_T_1_of_R_T):
    assert isinstance(C_T, float)
    assert isinstance(I_1, float)
    assert isinstance(I_2, float)
    assert isinstance(R_T, float)
    assert isinstance(frac_R_T_1_of_R_T, float)
    R_1 = R_T*frac_R_T_1_of_R_T
    R_2 = R_T*(1.0-frac_R_T_1_of_R_T)
    # this frequency was calculated using sympy from the differential equation for an RICRI terminal
    freq = np.float(0.159154943091895*sympy.re(sympy.Abs(-((I_1*R_2 + I_2*R_1)**2/(I_1**2*I_2**2) - 3*(C_T*R_1*R_2 + I_1 + I_2)/(C_T*I_1*I_2))/(3*(-1/2 - sympy.sqrt(3)*sympy.I/2)*(sympy.sqrt(-4*((I_1*R_2 + I_2*R_1)**2/(I_1**2*I_2**2) - 3*(C_T*R_1*R_2 + I_1 + I_2)/(C_T*I_1*I_2))**3 + (2*(I_1*R_2 + I_2*R_1)**3/(I_1**3*I_2**3) + 27*(R_1 + R_2)/(C_T*I_1*I_2) - 9*(I_1*R_2 + I_2*R_1)*(C_T*R_1*R_2 + I_1 + I_2)/(C_T*I_1**2*I_2**2))**2)/2 + (I_1*R_2 + I_2*R_1)**3/(I_1**3*I_2**3) + 27*(R_1 + R_2)/(2*C_T*I_1*I_2) - 9*(I_1*R_2 + I_2*R_1)*(C_T*R_1*R_2 + I_1 + I_2)/(2*C_T*I_1**2*I_2**2))**(1/3)) - (-1/2 - sympy.sqrt(3)*sympy.I/2)*(sympy.sqrt(-4*((I_1*R_2 + I_2*R_1)**2/(I_1**2*I_2**2) - 3*(C_T*R_1*R_2 + I_1 + I_2)/(C_T*I_1*I_2))**3 + (2*(I_1*R_2 + I_2*R_1)**3/(I_1**3*I_2**3) + 27*(R_1 + R_2)/(C_T*I_1*I_2) - 9*(I_1*R_2 + I_2*R_1)*(C_T*R_1*R_2 + I_1 + I_2)/(C_T*I_1**2*I_2**2))**2)/2 + (I_1*R_2 + I_2*R_1)**3/(I_1**3*I_2**3) + 27*(R_1 + R_2)/(2*C_T*I_1*I_2) - 9*(I_1*R_2 + I_2*R_1)*(C_T*R_1*R_2 + I_1 + I_2)/(2*C_T*I_1**2*I_2**2))**(1/3)/3 - (I_1*R_2 + I_2*R_1)/(3*I_1*I_2)).evalf()))
    assert isinstance(freq, float)
    return freq

@series_to_constant
def calc_spike_period(t, V, series_output=False):
    if series_output:
        return V
    peak_idxs, peak_properties = find_peaks(V)
    # TODO maybe check peak properties here
    if len(peak_idxs) < 2:
        # there aren't enough peaks to calculate a period
        # so set the period to the max time of the simulation
        period = t[-1] - t[0]
    else:
        # calculate the average period between peaks
        period = np.sum([t[peak_idxs[II+1]] - t[peak_idxs[II]] for II in range(len(peak_idxs)-1)])/(len(peak_idxs) - 1)
    return period

@series_to_constant
def calc_spike_frequency_windowed(t, V, series_output=False):
    """
    this calculates the number of spikes per 
    second in the given window. Not an accurate actual 
    frequency, but useful for some applications.
    """
    if series_output:
        return V
    peak_idxs, peak_properties = find_peaks(V)
    # TODO maybe check peak properties here
    spikes_per_s = len(peak_idxs)/(t[-1] - t[0])
    return spikes_per_s

@series_to_constant
def calc_spike_frequency_windowed_V(t, V, series_output=False):
    """
    this calculates the number of spikes per 
    second in the given window. Not an accurate actual 
    frequency, but useful for some applications.

    This includes a minimum threshold for peaks of 0
    """
    if series_output:
        return V
    peak_idxs, peak_properties = find_peaks(V, height=0.0)

    # TODO maybe check peak properties here
    spikes_per_s = len(peak_idxs)/(t[-1] - t[0])
    return spikes_per_s

@series_to_constant
def first_peak_time(t, V, series_output=False):
    """ 
    returns the time value (time from start of pre_time, NOT the start of 
    experiment or subexperiment) that the first peak occurs

    It is the time from the start, but it only checks in the subexperiment defined in obs_data.
    """
    if series_output:
        return V
    peak_idxs, peak_properties = find_peaks(V)
    
    if len(peak_idxs) == 0:
        # there are no peaks, return a big number but not np.inf, because it causes errors in mcmc
        return 99999999
    # t_first_peak = t[peak_idxs[0]] - t[0] # this would calc from start of subexperiment but there are plotting issues
    t_first_peak = t[peak_idxs[0]] # this is from the start of the pre_time, not the start of experiment.
    return t_first_peak
