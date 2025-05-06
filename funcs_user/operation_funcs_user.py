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

# TODO we should find a way to only find_peaks once per subexperiment
# ATM if multiple of the below functions are called, it does find_peaks multiple times
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
def calc_spike_frequency_windowed(t, V, series_output=False, spike_min_thresh=-10, start_frac=0.0, end_frac=1.0):
    """
    this calculates the number of spikes per 
    second in the given window. Not an accurate actual 
    frequency, but useful for some applications.

    This includes a minimum threshold for peaks of spike_min_thresh
    """
    if series_output:
        return V
    # get the start and end of the window
    start_idx = int(start_frac*(len(t)-1))
    end_idx = int(end_frac*(len(t)-1))
    peak_idxs, peak_properties = find_peaks(V[start_idx:end_idx], height=spike_min_thresh)

    # TODO maybe check peak properties here
    spikes_per_s = len(peak_idxs)/(t[end_idx] - t[start_idx])
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

@series_to_constant
def steady_state_min(x, series_output=False):
    """
    finds the min of the second half of this subexperiment. 
    The aim of this is to allow the dynamics to reach steady state
    or periodic steady state before getting the minimum
    """
    if series_output:
        return x
    else:
        return np.min(x[len(x)//2:])

@series_to_constant
def calc_min_to_max_period_diff(t, V, series_output=False, spike_min_thresh=None):
    if series_output:
        return V
    peak_idxs, peak_properties = find_peaks(V, height=spike_min_thresh)
    # TODO maybe check peak properties here
    if len(peak_idxs) < 2:
        # there aren't enough peaks to calculate a period
        # so set the period_diff to the max time of the simulation
        period_diff = t[-1] - t[0]
    else:
        # calculate the periods
        periods = [t[peak_idxs[II+1]] - t[peak_idxs[II]] for II in range(len(peak_idxs)-1)]
        # calculate the difference in time between max period and min period
        period_diff = max(periods) - min(periods)

    return period_diff

@series_to_constant
def calc_min_peak(t, V, series_output=False, spike_min_thresh=None):
    if series_output:
        return V
    peak_idxs, peak_properties = find_peaks(V, height=spike_min_thresh)
    # TODO maybe check peak properties here
    if len(peak_idxs) < 1:
        # if there aren't spikes set the min peak to the max of the voltage the max/sudo-peak)
        min_peak = max(V)
    else:
        min_peak = min(V[peak_idxs])

    return min_peak

@series_to_constant
def min_period(t, V, series_output=False, spike_min_thresh=None, distance=None):
    if series_output:
        return V
    # set distance = 5 to make sure it doesn't count a peak as two
    peak_idxs, peak_properties = find_peaks(V, height=spike_min_thresh, distance=distance)
    # TODO maybe check peak properties here
    if len(peak_idxs) < 2:
        # there aren't enough peaks to calculate a period
        # so set the period_diff to the max time of the simulation
        period_min = t[-1] - t[0]
    else:
        # calculate the periods
        periods = [t[peak_idxs[II+1]] - t[peak_idxs[II]] for II in range(len(peak_idxs)-1)]
        # calculate the difference in time between max period and min period
        period_min = min(periods)

    return period_min

@series_to_constant
def first_period(t, V, series_output=False, spike_min_thresh=None, distance=None):
    if series_output:
        return V
    # set distance = 5 to make sure it doesn't count a peak as two
    peak_idxs, peak_properties = find_peaks(V, height=spike_min_thresh, distance=distance)
    # TODO maybe check peak properties here
    if len(peak_idxs) < 1:
        # there aren't enough peaks to calculate a period
        # so set the period_diff to the max time of the simulation
        first_period = t[-1] - t[0]
    elif len(peak_idxs) < 2:
        # there aren't enough peaks to calculate a first period
        # so set the period_diff to the time to the first peak
        first_period = t[-1] - t[0]
    else:
        # calculate peaks without a threshold after first peak
        V_2 = V[peak_idxs[0]-2:] # first peak will be the same peak
        t_2 = t[peak_idxs[0]-2:] # first peak will be the same peak
        threshold_for_spike = min(V[peak_idxs[0]:peak_idxs[1]]) + 10 # setting this to try to ignore some noise
        peak_idxs_2, peak_properties_2 = find_peaks(V_2, height=threshold_for_spike, distance=distance)

        if len(peak_idxs_2) < 2:
            # there should have been another peak but for some reason it wasn't detected...
            first_period = t[-1] - t[0]
        else:
            # calculate the period
            first_period = t_2[peak_idxs_2[1]] - t[peak_idxs[0]]

    return first_period

@series_to_constant
def second_period(t, V, series_output=False, spike_min_thresh=None, distance=None):
    if series_output:
        return V
    # set distance = 5 to make sure it doesn't count a peak as two
    peak_idxs, peak_properties = find_peaks(V, height=spike_min_thresh, distance=distance)
    # TODO maybe check peak properties here
    if len(peak_idxs) < 3:
        # there aren't enough peaks to calculate a period
        # so set the period_diff to the max time of the simulation
        second_period = t[-1] - t[0]
    else:
        # calculate peaks without a threshold after first peak
        V_2 = V[peak_idxs[1]-2:] # first peak of the next peak calc will be the same peak
        t_2 = t[peak_idxs[1]-2:] 
        threshold_for_spike = min(V[peak_idxs[1]:peak_idxs[2]]) + 10 # setting this to try to ignore some noise
        peak_idxs_2, peak_properties_2 = find_peaks(V_2, height=threshold_for_spike, distance=distance)

        if len(peak_idxs_2) < 2:
            # there should have been another peak but for some reason it wasn't detected...
            second_period = t[-1] - t[0]
        else:
            # calculate the period
            second_period = t_2[peak_idxs_2[1]] - t[peak_idxs[1]]

    return second_period

@series_to_constant
def E_A_ratio(t, x, T, series_output=False):
    if series_output:
        return x
    peak_idxs, peak_properties = find_peaks(x)

    if len(peak_idxs) < 1:
        # no peeak idxs found, return big value to make it a big cost
        return 100
    elif len(peak_idxs) < 2:
        # there is only one peak. E and A ontop of eachother. return large cost.
        return 10
    if (t[peak_idxs[1]] - t[peak_idxs[0]] > 0.7* T) :
        # the peaks are too far apart. Probably because there is only one peak per heart beat.
        # return large value
        return 10

    # calculate with the first two peaks. This assumes that the E peak comes first
    E_A_ratio = x[peak_idxs[0]]/x[peak_idxs[1]]

    return E_A_ratio

# included by David Shaw
@series_to_constant
def peak_times(t, V, series_output=False):
    """
    returns all peak times
    """
    if series_output:
        return V
    peak_idxs, peak_properties = find_peaks(V)
    if len(peak_idxs) == 0:
        return 99999999
    peaks = t[peak_idxs]
    return peaks

@series_to_constant
def mean_last_half(x, series_output=False):
    if series_output:
        return x
    else:
        half_len = len(x) // 2
        last_half_values = x[half_len:]
        return np.mean(last_half_values)

@series_to_constant
def mean_last_quarter(x, series_output=False):
    if series_output:
        return x
    else:
        quarter_len = len(x) // 4
        last_quarter_values = x[-quarter_len:]
        return np.mean(last_quarter_values)

@series_to_constant
def max_first_half(x, series_output=False):
    if series_output:
        return x
    else:
        half_len = len(x) // 2
        first_half_values = x[:half_len]
        return np.max(first_half_values)

@series_to_constant
def max_first_quarter(x, series_output=False):
    if series_output:
        return x
    else:
        quarter_len = len(x) // 4
        first_quarter_values = x[:quarter_len]
        return np.max(first_quarter_values)

@series_to_constant
def max_second_quarter(x, series_output=False):
    if series_output:
        return x
    else:
        quarter_len = len(x) // 4
        second_quarter_values = x[quarter_len:2 * quarter_len]
        return np.max(second_quarter_values)

@series_to_constant
def max_last_quarter(x, series_output=False):
    if series_output:
        return x
    else:
        quarter_len = len(x) // 4
        last_quarter_values = x[-quarter_len:]
        return np.max(last_quarter_values)

@series_to_constant
def min_first_half(x, series_output=False):
    if series_output:
        return x
    else:
        half_len = len(x) // 2
        first_half_values = x[:half_len]
        return np.min(first_half_values)

@series_to_constant
def min_first_quarter(x, series_output=False):
    if series_output:
        return x
    else:
        quarter_len = len(x) // 4
        first_quarter_values = x[:quarter_len]
        return np.min(first_quarter_values)

@series_to_constant
def mean_AP_threshold(t, V, series_output=False, spike_min_thresh=None, distance=None, dV_dt_thresh=10e3):
    """
    This function calculates the mean action potential threshold
    using the peak detection algorithm from scipy.
    It finds the peaks in the voltage signal and then 
    moves back to pre AP (approximately) It then moves foreward until
    dV/dt is greater than dV_dt_thresh, default is 10 mV/ms (10e3 mV/s) from platkiewicz2010Threshold.

    # TODO this won't work with noise
    """
            
    if series_output:
        return V
    # set distance = 5 to make sure it doesn't count a peak as two
    peak_idxs, peak_properties = find_peaks(V, height=spike_min_thresh, distance=distance)
    # TODO maybe check peak properties here
    if len(peak_idxs) < 1:
        # there are no peaks, so set value to mean of the voltage
        threshold = np.mean(V)
    else:
        prev_idx = 0
        thresholds = []
        for peak_idx in peak_idxs:
            t_peak = t[peak_idx]
            current_idx = int((peak_idx + prev_idx) *3/ 4)
            dV_dt = 0
            dV_dt_prev = 0
            while dV_dt < dV_dt_thresh and current_idx < len(t) - 1:
                dV_dt_prev = dV_dt
                dV_dt = (V[current_idx + 1] - V[current_idx]) / (t[current_idx + 1] - t[current_idx])
                current_idx += 1
            if current_idx < len(t) - 1:
                interp_thresh = np.interp(dV_dt_thresh, [dV_dt_prev, dV_dt], [V[current_idx-1], V[current_idx]])
                thresholds.append(interp_thresh) 
                prev_idx = peak_idx
            else:
                # threshold not found for this peak, ignore it.
                pass
            
        if len(thresholds) == 0:
            # no thresholds found, exit
            print("no thresholds found, setting cost to large")
            threshold = 9999
        else:
            threshold = np.mean(thresholds)

    return threshold

@series_to_constant
def mean_peak_to_trough_time(t, V, series_output=False, spike_min_thresh=None, distance=None):
    """
    This function calculates the time between the peak and trough of each action potential
    then takes the mean of them all
    """
            
    if series_output:
        return V
    # set distance = 5 to make sure it doesn't count a peak as two
    peak_idxs, peak_properties = find_peaks(V, height=spike_min_thresh, distance=distance)
    # TODO maybe check peak properties here
    if len(peak_idxs) < 1:
        # there are no peaks, so set value to zero
        t_diff = 0
    else:
        t_diff_times = []
        for II in range(len(peak_idxs)):
            t_peak = t[peak_idxs[II]]

            next_peak_idx = peak_idxs[II + 1] if II + 1 < len(peak_idxs) else len(t) - 1
            trough_idx = np.argmin(V[peak_idxs[II]:next_peak_idx]) + peak_idxs[II]
            t_diff_times.append(t[trough_idx] - t_peak)
            
        t_diff = np.mean(t_diff_times)

    return t_diff






