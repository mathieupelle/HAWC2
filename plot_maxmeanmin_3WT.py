# -*- coding: utf-8 -*-
"""Plot a statistic versus wind speed.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from _loads_utils import load_stats


stat_dir1 = './dtu10mw_res/dtu10mw_tca/'  # stats file, turbine 1  !!! END WITH SLASH !!! DTU 10MW!!!!
stat_dir2 = './res_turb/'  # stats file, turbine 2  !!! END WITH SLASH !!!
labels = ['DTU 10 MW', 'DTU 10 MW Redesign']  # legend labels

# values to plot and indices of that value for the two turbines
wind_idxs = [15, 15]  # indices of vy wind for turbine 1 and turbine 2
#               Label                  Idx_t1  Idx_t2
plot_vals = [['Pitch angle [deg]',       4,    4],
             ['Rotor speed [rad/s]',     10,   10],
             ['Thrust [kN]',             13,   13],
             ['AoA @ 2/3R m',            None, 61],
             ['Cl @ 2/3R m',             None, 64],
             ['Tower-base FA [kNm]',     19,   17],
             ['Tower-base SS [kNm]',     20,   18],
             ['Yaw-bearing pitch [kNm]', 22,   20],
             ['Yaw-bearing roll [kNm]',  23,   21],
             ['Shaft torsion [kNm]',     27,   25],
             ['OoP BRM [kNm]',           28,   26],
             ['IP BRM [kNm]',            29,   27],
             ['Generator torque [Nm]',   72,   70],
             ['Electrical power [W]',    102,  100],
             ['Tower clearance [m]',     110,  108],
             ]

# =======================================================================================
# you shouldn't need to change below this line :)

axs = []  # initialize list of axes
# colors = ['#1f77b4', '#ff7f0e']  # colors for turbines 1 and 2
colors = ['purple', 'r']  # colors for turbines 1 and 2
colors1 = ['g', 'b']  # colors for turbines 1 and 2
markers = ['x', '.']  # markers for turbines 1 and 2

# load the stats data for turbines 1 and 2 (it's ugly i'm sorry)
_, idxs_1, means_1 = load_stats(stat_dir1 + 'stats_mean.txt')  # means
_, _, mins_1 = load_stats(stat_dir1 + 'stats_min.txt')  # mins
_, _, maxs_1 = load_stats(stat_dir1 + 'stats_max.txt')  # maxs
wind_1 = means_1[:, idxs_1 == wind_idxs[0]]
_, idxs_2, means_2 = load_stats(stat_dir2 + 'stats_mean.txt')  # means
_, _, mins_2 = load_stats(stat_dir2 + 'stats_min.txt')  # mins
_, _, maxs_2 = load_stats(stat_dir2 + 'stats_max.txt')  # maxs
wind_2 = means_2[:, idxs_2 == wind_idxs[1]]

pa1 = Patch(facecolor='green', edgecolor='black')
pa2 = Patch(facecolor='purple', edgecolor='black')
#
pb1 = Patch(facecolor='b', edgecolor='black')
pb2 = Patch(facecolor='r', edgecolor='black')
# plotting settings for that turbine
# m = markers[iturb]; mec = colors[iturb]

# loop over the channels to plot
for i, (ylabel, idx_t1, idx_t2) in enumerate(plot_vals):

    # make the plot and initialize some plot settings
    fig, ax = plt.subplots(num=1 + i, figsize=(7, 3), clear=True)
    ax.grid('on')
    ax.set(xlabel='Wind speed [m/s]', ylabel=ylabel)

    # loop through the turbines
    handles = []
    for iturb, (idxs, means, mins, maxs, wind) in enumerate(zip(
            [idxs_1, idxs_2], [means_1, means_2], [mins_1, mins_2],
            [maxs_1, maxs_2], [wind_1, wind_2])):

        # get the channel idx, skip if None
        chan_idx = [idx_t1, idx_t2][iturb]
        if chan_idx is None:
            continue

        # isolate the channel to plot
        meanval = means[:, idxs == chan_idx]
        minval = mins[:, idxs == chan_idx]
        maxval = maxs[:, idxs == chan_idx]

        # get plotting marker and color
        m = markers[iturb]
        mec = colors[iturb]
        mec1 = colors1[iturb]

        # only plot minimum value if tower clearance
        if 'clearance' in ylabel:
            l, = ax.plot(wind, minval, marker='.', mec=mec, mfc='none', alpha=0.8, linestyle='none')
        # otherwise plot mean, max and min
        else:
            l, = ax.plot(wind, meanval, marker='.', mec=mec1, mfc='none', alpha=0.8, linestyle='none')
            ax.plot(wind, minval, marker='.', mec=mec, mfc='none', alpha=0.8, linestyle='none')
            ax.plot(wind, maxval, marker='.', mec=mec, mfc='none',  alpha=0.8, linestyle='none')
        handles.append(l)

    if 'AoA' in ylabel:
        ax.legend(handles=[pb1, pb2],
                  labels=['', 'DTU 10 MW Redesign'], ncol=2, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5,
          loc='best', fontsize=10)
    elif 'Cl' in ylabel:
        ax.legend(handles=[pb1, pb2],
                  labels=['', 'DTU 10 MW Redesign'], ncol=2, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5, loc='best', fontsize=10)
    elif 'clearance' in ylabel:
        ax.legend(handles=[pa2, pb2],
                  labels=['DTU 10MW', 'DTU 10 MW Redesign'], ncol=1, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5, loc='best', fontsize=10)
    else:
        ax.legend(handles=[pa1, pb1, pa2, pb2],
                  labels=['', '', 'DTU 10MW', 'DTU 10 MW Redesign'], ncol=2, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5,
          loc='best', fontsize=10)
    fig.tight_layout()

    plt.show()
