# -*- coding: utf-8 -*-
"""Utilities for assignment 4
"""
import re
import numpy as np


def load_stats(stat_file):
    """Load the statistics from the text file."""
    # load the raw data
    raw = np.loadtxt(stat_file, skiprows=1, dtype=object)
    idxs = raw[0, 1:].astype(int)
    files = raw[1:, 0].astype(str)
    data = raw[1:, 1:].astype(float)
    # sort files by wind speed, then by file name. Have to use fancy regex and Python
    #   sorting tricks. Sorry that this is hard to understand what's going on...
    outsort = sorted(zip(files, data),
                     key=lambda f: (float(re.findall('[0-9]{1,2}.[0-9]', f[0])[0]),  # wind speed first
                                    f[0]  # then filename seconds
                                    )
                     )  # returns a tuple of tuples, (file, datarow)
    files, data = zip(*outsort)  # separate each tuple back into two separate lists
    data = np.array(data)  # convert data back to a numpy array instead of a tuple of arrays
    # return the result
    return files, idxs, data


def load_hawc2s(path):
    """load channels from a pwr or opt file"""
    pwr = True if path.endswith('.pwr') else False
    data = np.loadtxt(path, skiprows=1)
    if pwr:
        u = data[:, 0]  # wind speed [m/s]
        paero = data[:, 1] * 1e3  # aerodynamic power [W]
        thrust = data[:, 2] * 1e3  # thrust [N]
        pitch = data[:, 8]  # pitch [deg]
        rotspd = data[:, 9] * np.pi/30  # rotational speed [rad/s]
        aerotrq = paero / rotspd  # aerodynamic torque [N]
    else:
        u = data[:, 0]  # wind speed [m/s]
        pitch = data[:, 1]  # pitch [deg]
        rotspd = data[:, 2] * np.pi/30  # rotational speed [rad/s]
        paero = data[:, 3] * 1e3  # aerodynamic power [W]
        thrust = data[:, 4] * 1e3  # thrust [N]
        aerotrq = paero / rotspd  # aerodynamic torque [N]
    return u, pitch, rotspd, paero, thrust, aerotrq
        