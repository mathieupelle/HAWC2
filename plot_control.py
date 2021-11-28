# -*- coding: utf-8 -*-
"""How to load various HAWC2 formats to a NumPy array.
Also does a plot thing because why not.
"""
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os


def load_hawc_ascii(dat_path):
    """Load HAWC ascii data to a NumPy array.
    Input path to .dat file.
    """
    data = np.loadtxt(dat_path)
    return data


def load_hawc_binary(dat_path):
    """Load HAWC binary file to a NumPy array.
    Input path to .dat file.
    """
    sel_path = dat_path.replace('.dat', '.sel')
    with open(sel_path, 'r') as f:
        content = f.readlines()
        # get number lines and channels
        nr_lines, nr_chan = [int(s) for s in content[8].split()[:2]]
        # get the scale factors
        i_scale = [i+1 for i in range(len(content)) if 'scale factors' in content[i].lower()][0]
        scale_factors = np.loadtxt(sel_path, skiprows=i_scale)
    with open(dat_path, 'rb') as fid:
        data = np.zeros((nr_lines, nr_chan))
        j = 0
        for i in range(nr_chan):
            fid.seek(i * nr_lines * 2, 0)
            data[:, j] = np.fromfile(fid, 'int16', nr_lines) * scale_factors[i]
            j += 1
    return data

#%% Importing datasets

cases = ['./V2/results/hawc2/control/f0.5_eta0.7.dat']

data_lst = []
for c in range(len(cases)):
    path = cases[c]
    data = load_hawc_binary(path)
    data_lst.append(data)

#%% Plotting

itime = 0  # column index of time
irotspd = 9  # column index of rotor speed
iwspd = 14 # column index of wind speed
ipitch = 70 # column index of pitch angle
iepower = 99 # column index of electrical power
itorque = 95 # column index of generator torque

data_all = data_lst
time_shift = 0
# ====================== plot things for fun ======================
idx = [irotspd, iwspd, ipitch, iepower, itorque]
name = ['rpm', 'wspd', 'pitch', 'power', 'torque']
leg = ['$\omega$ [rad/s]', '$V_{\infty}$ [m/s]', r'$\theta$ [deg]',
       '$P_{electric}$ [MW]', '$Q_{gen}$ [Nm]']
#lab = ['C1: $f = 0.05$, $\zeta = 0.7$', 'C2: $f = 0.01$, $\zeta = 0.7$', 'C3: $f = 0.10$, $\zeta = 0.7$']
lab = ['C4: $f = 0.05$, $\zeta = 0.7$', 'C5: $f = 0.01$, $\zeta = 0.7$','C6: $f = 0.10$, $\zeta = 0.7$']


for i in range(len(idx)):
    plt.figure()
    for j in range(len(data_all)):
        data = data_all[j]
        time = data[time_shift*100:, itime]
        y = data[time_shift*100:, idx[i]]
        if idx[i] == ipitch:
            y = np.rad2deg(y)
        elif idx[i] == iepower:
            y = y/10**6
        plt.plot(time, y, label=lab[j])
    plt.ylabel(leg[i])
    plt.xlabel('t [s]')
    plt.grid()
    plt.legend()
    #plt.savefig('figures/cstP_'+name[i]+'.png')
    #plt.savefig('figures/cstT_'+name[i]+'.png')




