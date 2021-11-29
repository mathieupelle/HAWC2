# -*- coding: utf-8 -*-
"""Calculate extreme loads.

Use method (a) in IEC 61400-1 (2019), which is mean extreme times 1.25*1.35. Note that
we are incorrectly applying this to all loads, when it is only technically allowed for
blade root moments and deflections.
"""
import re
import matplotlib.pyplot as plt
import numpy as np
from _loads_utils import load_stats


stat_dir = './res_turb/'  # results directory with statistics files  !!! END WITH SLASH !!!
i_plot = [17, 18, 20, 21, 25, 26, 27, 108]  # channel indices in .sel file that you want to process
i_wind = 15  # channel number with the wind speed

# dictionary to map .sel index to ylabel for the plot
ylabels = {4: 'Pitch angle [deg]',
           10: 'Rotor speed [rad/s]',
           13: 'Thrust [kN]',
           15: 'Wind speed [m/s]',
           17: 'Tower-base FA [kNm]',
           18: 'Tower-base SS [kNm]',
           20: 'Yaw-bearing pitch [kNm]',
           21: 'Yaw-bearing roll [kNm]',
           25: 'Shaft torsion [kNm]',
           26: 'OoP BRM [kNm]',
           27: 'IP BRM [kNm]',
           70: 'Generator torque [Nm]',
           100: 'Electrical power [W]',
           108: 'Tower clearance [m]'}

# load the min statistics
stat_file = stat_dir + 'stats_min.txt'
files, idxs, data_min = load_stats(stat_file)

# load the mean statistics
stat_file = stat_dir + 'stats_mean.txt'
files, idxs, data_mean = load_stats(stat_file)
wind = data_mean[:, idxs == i_wind].squeeze()

# load the max statistics
stat_file = stat_dir + 'stats_max.txt'
files, idxs, data_max = load_stats(stat_file)

# extract the set wind speed value from the filename using regex tricks
wsps = [float(re.findall('[0-9]{1,2}[.][0-9]', f)[0]) for f in files]

# loop through the channels
for i, chan_idx in enumerate(i_plot):

    # define ylabel
    ylabel = ylabels[chan_idx]

    # isolate the channels to plot
    minval = data_min[:, idxs == chan_idx]
    maxval = data_max[:, idxs == chan_idx]

    # get mean of the extremes for each wind speed bin
    wsp_unique = np.unique(wsps)
    mean_min = np.empty(wsp_unique.size)
    mean_max = np.empty(wsp_unique.size)
    for j, vj in enumerate(wsp_unique):
        mean_min[j] = minval[np.isclose(wsps, vj)].mean()
        mean_max[j] = maxval[np.isclose(wsps, vj)].mean()

    # calculate design extreme loads
    if 'clearance' in ylabel.lower():  # tower clearance is weird: no safety factors and min of min
        maxval = np.full_like(maxval, np.nan)  # set maxes to nans, because those are irrelevant
        meanval = np.full_like(maxval, np.nan)  # set maxes to nans, because those are irrelevant
        mean_max[:] = np.nan
        extr_design_load = np.min(minval)  # transform back to tower clearance
    else:
        extremes = np.hstack((mean_min, mean_max)).squeeze()  # combine min and max extremes
        i_ext = np.nanargmax(np.abs(extremes))  # find index of value with max abs value
        fc = 1.35 * extremes[i_ext]  # characteristic load
        extr_design_load = 1.25 * fc  # extreme design load
    print(ylabel, f'{extr_design_load:.6e}')

    # make the plot
    fig = plt.figure(1 + i, figsize=(7, 3), clear=True)
    plt.plot(wind, minval, 'o')  # minimum values
    plt.plot(wsp_unique, mean_min, 'or', mec='0.2', ms=7, alpha=0.8)  # plot mean extremes for fun
    plt.plot(wind, maxval, 'o')  # maximum values
    plt.plot(wsp_unique, mean_max, 'or', mec='0.2', ms=7, alpha=0.8)  # plot mean extremes for fun
    plt.plot(wind, extr_design_load * np.ones_like(wind), lw=2, c='0.2')  # extreme design load
    plt.grid('on')
    plt.xlabel('Wind speed [m/s]')
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.legend(['min', 'max'])

plt.show()