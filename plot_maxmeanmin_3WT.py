# -*- coding: utf-8 -*-
"""Plot a statistic versus wind speed.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from _loads_utils import load_stats


stat_dir1 = './DTU10MW/results/hawc2/dtu10mw_tca/'  # stats file, turbine 1  !!! END WITH SLASH !!! DTU 10MW!!!!
stat_dir2 = './V1/results/hawc2/res_turb/'  # stats file, turbine 2  !!! END WITH SLASH !!!
stat_dir3= './V2/results/hawc2/res_turb/'  # stats file, turbine 3  !!! END WITH SLASH !!!
labels = ['DTU 10MW', 'Redesign V1', 'Redesign V2']  # legend labels

# values to plot and indices of that value for the two turbines
wind_idxs = [15, 15, 15]  # indices of vy wind for turbine 1 and turbine 2
#               Label                  Idx_t1  Idx_t2
plot_vals = [['Pitch angle [deg]',       4,    4,   4],
             ['Rotor speed [rad/s]',     10,   10,  10],
             ['Thrust [kN]',             13,   13,  13],
             ['AoA @ 2/3R m',            None, 61,  61],
             ['Cl @ 2/3R m',             None, 64,  64],
             ['Tower-base FA [kNm]',     19,   17,  17],
             ['Tower-base SS [kNm]',     20,   18,  18],
             ['Yaw-bearing pitch [kNm]', 22,   20,  20],
             ['Yaw-bearing roll [kNm]',  23,   21,  21],
             ['Shaft torsion [kNm]',     27,   25,  25],
             ['OoP BRM [kNm]',           28,   26,  26],
             ['IP BRM [kNm]',            29,   27,  27],
             ['Generator torque [Nm]',   72,   70,  70],
             ['Electrical power [W]',    102,  100, 100],
             ['Tower clearance [m]',     110,  108, 108],
             ]

# =======================================================================================
# you shouldn't need to change below this line :)

axs = []  # initialize list of axes
# colors = ['#1f77b4', '#ff7f0e']  # colors for turbines 1 and 2
colors_maxmin = ['#8A8A8A', '#EE6363', '#4876FF']  # colors for max and min
colors_mean = ['#363636', '#CD0000', '#000080']  # colors mean
markers = ['.', '+', '2']  # markers for turbines 1 and 2

# load the stats data for turbines 1 and 2 (it's ugly i'm sorry)
_, idxs_1, means_1 = load_stats(stat_dir1 + 'stats_mean.txt')  # means
_, _, mins_1 = load_stats(stat_dir1 + 'stats_min.txt')  # mins
_, _, maxs_1 = load_stats(stat_dir1 + 'stats_max.txt')  # maxs
wind_1 = means_1[:, idxs_1 == wind_idxs[0]]

_, idxs_2, means_2 = load_stats(stat_dir2 + 'stats_mean.txt')  # means
_, _, mins_2 = load_stats(stat_dir2 + 'stats_min.txt')  # mins
_, _, maxs_2 = load_stats(stat_dir2 + 'stats_max.txt')  # maxs
wind_2 = means_2[:, idxs_2 == wind_idxs[1]]

_, idxs_3, means_3 = load_stats(stat_dir3 + 'stats_mean.txt')  # means
_, _, mins_3 = load_stats(stat_dir3 + 'stats_min.txt')  # mins
_, _, maxs_3 = load_stats(stat_dir3 + 'stats_max.txt')  # maxs
wind_3 = means_3[:, idxs_3== wind_idxs[1]]

pa1 = Patch(facecolor='#363636', edgecolor='black')
pa2 = Patch(facecolor='#8A8A8A', edgecolor='black')

#
pb1 = Patch(facecolor='#CD0000', edgecolor='black')
pb2 = Patch(facecolor='#EE6363', edgecolor='black')

pc1 = Patch(facecolor='#000080', edgecolor='black')
pc2 = Patch(facecolor='#4876FF', edgecolor='black')
# plotting settings for that turbine
# m = markers[iturb]; mec = colors[iturb]

# loop over the channels to plot
for i, (ylabel, idx_t1, idx_t2, idx_t3) in enumerate(plot_vals):

    # make the plot and initialize some plot settings
    fig, ax = plt.subplots(num=1 + i, figsize=(7, 3), clear=True)
    ax.grid('on')
    ax.set(xlabel='Wind speed [m/s]', ylabel=ylabel)

    # loop through the turbines
    handles = []
    for iturb, (idxs, means, mins, maxs, wind) in enumerate(zip(
            [idxs_1, idxs_2, idxs_3], [means_1, means_2, means_3], [mins_1, mins_2, mins_3],
            [maxs_1, maxs_2, maxs_3], [wind_1, wind_2, wind_3])):

        # get the channel idx, skip if None
        chan_idx = [idx_t1, idx_t2, idx_t3][iturb]
        if chan_idx is None:
            continue

        # isolate the channel to plot
        meanval = means[:, idxs == chan_idx]
        minval = mins[:, idxs == chan_idx]
        maxval = maxs[:, idxs == chan_idx]

        # get plotting marker and color
        m = markers[iturb]
        mec = colors_maxmin[iturb]
        mec1 = colors_mean[iturb]

        # only plot minimum value if tower clearance
        if 'clearance' in ylabel:
            l, = ax.plot(wind, minval, marker='.', mec=mec, mfc='none', alpha=0.8, linestyle='none')
        # otherwise plot mean, max and min
        else:
            l, = ax.plot(wind, meanval, marker=m, mec=mec1, mfc='none', alpha=0.8, linestyle='none')
            ax.plot(wind, minval, marker=m, markersize=4, mec=mec, mfc='none', alpha=0.8, linestyle='none')
            ax.plot(wind, maxval, marker=m, markersize=4, mec=mec, mfc='none',  alpha=0.8, linestyle='none')
        handles.append(l)

    if 'AoA' in ylabel:
        ax.legend(handles=[pb1, pc1, pb2, pc2],
                  labels=['', '', 'Redesign V1' , 'Redesign V2'], ncol=2, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5,
          loc='best', fontsize=10)
    elif 'Cl' in ylabel:
        ax.legend(handles=[pb1, pc1, pb2, pc2],
                  labels=['', '', 'Redesign V1' , 'Redesign V2'], ncol=2, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5, loc='best', fontsize=10)
    elif 'clearance' in ylabel:
        ax.legend(handles=[pa2, pb2, pc2],
                  labels=['DTU 10MW', 'Redesign V1' , 'Redesign V2'], ncol=1, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5, loc='best', fontsize=10)
    else:
        ax.legend(handles=[pa1, pb1, pc1, pa2, pb2, pc2],
                  labels=['', '', '', 'DTU 10MW', 'Redesign V1', 'Redesign V2'], ncol=2, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5,
          loc='best', fontsize=10)
    fig.tight_layout()

    plt.show()
