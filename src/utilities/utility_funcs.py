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

def bin_resample(data, freq_1, freq_ds):

    new_len = len(freq_ds)
    new_data = np.zeros((new_len))
    new_count = 0 
    this_count = 0 
    addup = 0 
    for II in range(0, len(freq_1)):
        
        dist_behind = np.abs(freq_1[II] - freq_ds[new_count])
        dist_infront = np.abs(freq_1[II] - freq_ds[new_count+1])
        if dist_behind < dist_infront:
            addup += data[II]
            this_count += 1
        else:
            if new_count == 0:
                # overwrite with 0th entry of data
                # this ignores some data points directly after 0 frequency
                new_data[0] = data[0]
            else:
                new_data[new_count] = addup / this_count
            addup = data[II]
            this_count = 1 
            new_count += 1

        if new_count == len(freq_ds) - 1:
            # add all remaining data points to this new datapoint and average
            new_data[new_count] = np.sum(data[II+1:]) / len(data[II+1:])
            break

    return new_data










