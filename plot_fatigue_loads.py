# -*- coding: utf-8 -*-
"""Calculate extreme loads.

Use method (a) in IEC 61400-1 (2019), which is extreme times 1.35. Note that we are
incorrectly applying this to all loads, when it is only technically allowed for blade
root moments and deflections.
"""
import re
import matplotlib.pyplot as plt
import numpy as np
from _loads_utils import load_stats


stat_dir1 = './dtu10mw_res/dtu10mw_tca/'
stat_dir2 = './res_turb/'  # results directory with statistics files  !!! END WITH SLASH !!!
stat_dirs = [stat_dir1, stat_dir2]
labels = ['DTU 10 MW', 'DTU 10 MW (mean)', 'DTU 10 MW Redesign', 'DTU 10 MW Redesign (mean)']

i_plot = [17, 18, 20, 21, 25, 26, 27, 108]  # channel indices in .sel file that you want to process
i_wind = 15  # channel number with the wind speed
# undef_tclear = 18.25  # undeflected-blade tower clearance [m]
v_ref = [50, 37.5]  # reference wind speed based on wind class (I=50, 2=42.5, 3=37.5)


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

marker = ['dg', 'om']
marker2 = ['db', 'or']
# loop through the channels
for i, chan_idx in enumerate(i_plot):

    # define ylabel
    ylabel = ylabels[chan_idx]

    fig = plt.figure(1 + i, figsize=(7, 3), clear=True)
    plt.grid('on')
    plt.xlabel('Wind speed [m/s]')
    plt.ylabel(ylabel)
    plt.tight_layout()


    for t in range(2):

        stat_dir = stat_dirs[t]
        # load the del 4 statistics
        stat_file = stat_dir + 'stats_del4.txt'
        files, idxs, data_4 = load_stats(stat_file)

        # load the del 10 statistics
        stat_file = stat_dir + 'stats_del10.txt'
        files, idxs, data_10 = load_stats(stat_file)

        # get the mean wind for plotting
        stat_file = stat_dir + 'stats_mean.txt'
        files, idxs, data_mean = load_stats(stat_file)
        wind = data_mean[:, idxs == i_wind].squeeze()


        # extract the set wind speed value from the filename using regex tricks
        wsps = [float(re.findall('[0-9]{1,2}[.][0-9]', f)[0]) for f in files]


        if t==0:
            chan_idx_new = chan_idx + 2
        else:
            chan_idx_new = chan_idx

        # determine which DEL to select
        if 'BRM' in ylabel:
            data = data_10[:, idxs == chan_idx_new].squeeze()
            m = 10  # 10 exponent for composites
        else:
            data = data_4[:, idxs == chan_idx_new].squeeze()
            m = 4  # 4 exponent for metals

        # combine short-term dels in given wind speed bin to single value for that bin
        wsp_uniqe = np.unique(wsps)
        st_dels = np.empty(wsp_uniqe.size)
        for j, vj in enumerate(wsp_uniqe):
            wsp_dels = data[np.isclose(wsps, vj)]  # short-term dels
            p = 1 / wsp_dels.size  # probability of each wsp in this bin
            st_dels[j] = sum(p * wsp_dels**m)**(1/m)

        # plot short-term dels versus wind speed

        plt.plot(wind, data, marker[t])
        # for fun, plot the wind-speed-averaged DELs on top
        plt.plot(wsp_uniqe, st_dels, marker2[t], mec='0.2', ms=7, alpha=0.9)

        # calculate the lifetime damage equivalent load
        v_ave = 0.2 * v_ref[t]  # average wind speed per IEC 61400-1
        dvj = wsp_uniqe[1] - wsp_uniqe[0]  # bin width. assumes even bins!
        probs = (np.exp(-np.pi*((wsp_uniqe - dvj/2) / (2*v_ave))**2)
                 - np.exp(-np.pi*((wsp_uniqe + dvj/2) / (2*v_ave))**2))  # prob of wind in each bin
        del_life = sum(probs * st_dels**m)**(1/m)  # sum DELs ignoring reference number of cycles

        if t==0:
            print('DTU 10MW '+ylabel, f'{del_life:.6e}')
        else:
            print('Redesign '+ylabel, f'{del_life:.6e}')
    plt.legend(labels)
