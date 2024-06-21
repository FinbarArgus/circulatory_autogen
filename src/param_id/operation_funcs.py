import numpy as np

def series_to_constant(func):
    func.series_to_constant = True
    return func

@series_to_constant
def max(x, series_output=False):
    if series_output:
        return x
    else:
        return np.max(x)

@series_to_constant
def min(x, series_output=False):
    if series_output:
        return x
    else:
        return np.min(x)

@series_to_constant
def mean(x, series_output=False):
    if series_output:
        return x
    else:
        return np.mean(x)

@series_to_constant
def max_minus_min(x, series_output=False):
    if series_output:
        return x
    else:
        return np.max(x) - np.min(x)

def addition(x1, x2):
    return x1 + x2

def subtraction(x1, x2):
    return x1 - x2

def multiplication(x1, x2):
    return x1 * x2

def division(x1, x2):
    return x1 / x2
