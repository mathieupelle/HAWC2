# -*- coding: utf-8 -*-
"""
Load a structural and aeroelastic Campbell diagram and plot it.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


path = './V2/data/V2_hs2.opt'

struc_path = './V2/results/hawcstab2/structural_it2.cmb'  # either None or path to structural .cmb file
aero_path = './V2/results/hawcstab2/aeroelastic_it2.cmb'  # either None or path to aeroelastic .cmb file
min_wsp = 4  # minimum wind speed to plot
max_modes = 12  # maximum number of modes to plot
opt_path = path  # opt path for 1P, 3P, 6P lines
save_fig = False  # save the figures to png?

# identification of modes for structural and aeroelastic campbell diagram
modes_struc = ['1. 1st Twr FA', '2. 1st Twr SS', '3. 1st BW flap', '4. 1st SYM flap',
              '5. 1st FW flap', '6. 1st BW edge', '7. 1st FW edge', '8. 2nd BW flap',
              '9. 2nd FW flap', '10. 2nd SYM flap', '11. 3rd SYM flap ', '12. 1st DT ']
modes_aero = ['1. 1st Twr FA', '2. 1st Twr SS', '3. 1st BW flap', '4. 1st FW flap',
              '5. 1st SYM flap', '6. 1st BW edge', '7. 1st FW edge', '8. 2nd BW flap',
              '9. 2nd FW flap', '10. 2nd SYM flap', '11. 1st SYM edge', '12. 1st DT']


#%% Define useful functions

def load_campbell(path, min_wsp=4, max_modes=11, filter_modes=True):
    """Load a Campbell diagram, filtering out modes who are all critically damped.
    Option to provide a minimum wind speed and max number of modes to load.
    """
    damp_thresh = 98  # set threshold to 98%
    # open the file, isolate useful columns
    with open(path, 'r') as f:
        nmodes = int(f.readline().split()[-1])
    all_data = np.loadtxt(path, skiprows=1)
    wind = all_data[:, 0]
    freqs = all_data[:, 1:nmodes + 1]
    damps = all_data[:, nmodes + 1:2*nmodes + 1]
    # if desired, filter out very-damped modes
    if filter_modes:
        mask = np.abs(damps) > damp_thresh  # mask the values whose damping is too high
        freqs[mask] = np.nan  # set overdamped freqs to nan
        out_cols = ~np.all(mask, axis=0)  # only return cols who have at least 1 non-nan
        freqs = freqs[:, out_cols]
        damps = damps[:, out_cols]
    # filter out wind speeds below minimum and by max number of modes
    freqs = freqs[wind > min_wsp, :][:, :max_modes]
    damps = damps[wind > min_wsp, :][:, :max_modes]
    wind = wind[wind > min_wsp]
    # sort columns by increasing first frequency
    idx_sort = np.argsort(freqs[0, :])
    freqs = freqs[:, idx_sort]
    damps = damps[:, idx_sort]
    return wind, freqs, damps


def switch_modes(wind, freqs, damps, col_mode1, col_mode2, wsp_start):
    """Switch modes in freqs, damps due to misidentification"""
    # check that the requested starting wind speed is in the provided wind speeds
    if wsp_start not in wind:
        raise ValueError(f'Wind speed {wsp_start} not in available wind speeds!')
    # get the row index for the starting wind speed
    idx_start = np.argmax(wind == wsp_start)
    # perform the mode switching
    f_mode1 = freqs[idx_start:, col_mode1].copy()
    d_mode1 = damps[idx_start:, col_mode1].copy()
    freqs[idx_start:, col_mode1] = freqs[idx_start:, col_mode2]
    damps[idx_start:, col_mode1] = damps[idx_start:, col_mode2]
    freqs[idx_start:, col_mode2] = f_mode1
    damps[idx_start:, col_mode2] = d_mode1
    return freqs, damps


def plot_freqs_vs_wsp(wsps, freqs, fig_num=1, labels=None, opt_path=None):
    """Make a Campbell diagram figure. Option to plot the NP harmonics if opt_path given."""
    # set plotting parameters
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', plt.cm.tab20.colors[::2] + plt.cm.tab20.colors[1::2])
    # initialize the figure
    fig, ax = plt.subplots(1, num=fig_num, clear=True, figsize=(9, 4))
    # plot the data
    handles = ax.plot(wsps, freqs, '-o', mfc='w')
    # plot 1P, 3P and 6P if operation.dat file path is given
    if opt_path is not None:
        opt_wsp, opt_rotspd = np.loadtxt(opt_path, skiprows=1, usecols=[0, 2]).T
        opt_rotspd = opt_rotspd[opt_wsp >= wsps.min()] / 60  # in Hz
        opt_wsp = opt_wsp[opt_wsp >= wsps.min()]
        lines = [ax.plot(opt_wsp, opt_rotspd*np.max([3*i, 1]),
                         '0.2', alpha=0.8, linestyle=['--', '-.', ':'][i])[0]
                 for i in range(3)]
        handles += lines
        labels = labels + ['1P', '3P', '6P']
    # make the plot pretty
    ax.grid('on')
    ax.set(xlabel='Wind speed [m/s]', ylabel='Natural frequency [Hz]')
    if labels is not None:
        ax.legend(handles=handles, labels=labels,
                  bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    fig.tight_layout()
    return fig, ax


def plot_damp_vs_wsp(wsps, damp, fig_num=1, labels=None,):
    """Plot damping versus wind speed."""
    # set plotting parameters
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', plt.cm.tab20.colors[::2] + plt.cm.tab20.colors[1::2])
    # initialize the figure
    fig, ax = plt.subplots(1, num=fig_num, clear=True, figsize=(9, 4))
    # plot the data
    handles = ax.plot(wsps, damp, '-o', mfc='w')
    # make the plot pretty
    ax.grid('on')
    ax.set(xlabel='Wind speed [m/s]', ylabel='Damping [% critical]')
    if labels is not None:
        ax.legend(handles=handles, labels=labels,
                  bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    fig.tight_layout()
    return fig, ax


#%% Plot the figures

# =================== structural campbell diagram ===================
if struc_path is not None:

    # freqs and damps for no-aero
    u_struc, f_struc, d_struc = load_campbell(struc_path, min_wsp=min_wsp, max_modes=max_modes)

    # plot the campbell diagram
    fig, axs = plot_freqs_vs_wsp(u_struc, f_struc, fig_num=1, labels=modes_struc)
    if save_fig: fig.savefig('campbell_structural.png', dpi=150)

    # plot the damping
    fig, axs = plot_damp_vs_wsp(u_struc, d_struc, fig_num=2, labels=modes_aero)
    if save_fig: fig.savefig('damping_structural.png', dpi=150)


# =================== aeroelastic campbell diagram ===================

if aero_path is not None:

    # freqs and damps for aero
    u_aero, f_aero, d_aero = load_campbell(aero_path, min_wsp=min_wsp, max_modes=max_modes)

    # plot the campbell diagram
    fig, axs = plot_freqs_vs_wsp(u_aero, f_aero, fig_num=3, opt_path=opt_path, labels=modes_aero)
    if save_fig: fig.savefig('campbell_aeroelastic.png', dpi=150)

    # plot the damping
    fig, axs = plot_damp_vs_wsp(u_aero, d_aero, fig_num=4, labels=modes_aero)
    if save_fig: fig.savefig('damping_aeroelastic.png', dpi=150)

#%% Tower (NR) and blade (R) frequencies

import pandas as pd

wsp = 15
omega = 7.74 #RPM
omega = omega*np.pi/30/2/np.pi
wsp_idx = np.asarray(np.where(u_aero==wsp))[0][0]

f = f_aero[wsp_idx,:]
modes_aero = np.array(['1st Twr FA', '1st Twr SS', '1st BW flap', '1st FW flap',
              '1st SYM flap', '1st BW edge', '1st FW edge', '2nd BW flap',
              '2nd FW flap', '2nd SYM flap', '1st SYM edge', '1st DT'])

idx_reorder = np.array([1, 2, 3, 5, 4, 6, 11, 7])-1
f_reordered = f[idx_reorder]

idx_nonsym = np.array([1, 2, 3, 5, 6, 7])-1
freqs = np.zeros((len(f_reordered),3))
freqs[:,0] = f_reordered
freqs[:,1] = f_reordered - omega
freqs[:,2] = f_reordered + omega
freqs[3, 1:3] = freqs[3,0]
freqs[6, 1:3] = freqs[6,0]

df = pd.DataFrame(freqs, columns = ['Tower [Hz]', 'Blade signal FW [Hz]', 'Blade signal BW [Hz]'])
df.index = modes_aero[idx_reorder]
df = df.round(3)

print(df.to_latex())