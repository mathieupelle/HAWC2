# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 13:52:29 2021

@author: Mathieu Pellé
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% For varying wsp

f1 = './DTU10MW/results/hawc2s/DTU_10MW_flexible_hawc2s.pwr'
f2 = './V1/results/hawc2s/DTU_10MW_redesign_flexible_hawc2s.pwr'
f3 = './V2/results/hawc2s/ind/flex/V2_hs2.pwr'

paths =  [f1, f2, f3]
leg_names = ['DTU 10MW', 'Redesign V1', 'Redesign V2']
colours = ['black', 'red', 'blue']
col_names = ['V', 'P', 'T', 'CP', 'CT', 'Q', 'Flap M', 'Edge M',\
             'pitch', 'speed', 'tip x', 'typ y', 'tip z', 'J_rot', 'J_dt']
ylab = ['V', 'P [kW]', 'T [kN]', '$C_P$ [-]', '$C_T$ [-]', 'Q [kNm]', 'Flap M [kNm]', 'Edge M [kNm]',\
             r'$\theta$ [deg]', '$\omega$ [rpm]', 'tip x', 'typ y', 'tip z', 'J_rot', 'J_dt']
idx_plot = [1, 2, 3, 4, 6, 7, 8, 9]

for i in range(len(idx_plot)):

    plt.figure()
    for j in range(len(paths)):

        data = pd.read_csv(paths[j], sep='\s+', skiprows = 1, names = col_names)
        x = data['V']
        y = data[col_names[idx_plot[i]]]
        plt.plot(x, y, '-', label=leg_names[j], color=colours[j])

    plt.xlabel('$V_{\infty}$ [m/s]')
    plt.ylabel(ylab[idx_plot[i]])
    plt.legend()
    plt.grid()

#%% For varying tsr

path = './V2/results/hawc2s/tsr/V2_hs2.pwr'

R = 95.32742340218736

leg_names = ['DTU 10MW', 'Redesign 1']
col_names = ['V', 'P', 'T', 'CP', 'CT', 'Q', 'Flap M', 'Edge M',\
             'pitch', 'speed', 'tip x', 'typ y', 'tip z', 'J_rot', 'J_dt']
ylab = ['V', 'P [kW]', 'T [kN]', '$C_P$ [-]', '$C_T$ [-]', 'Q', 'Flap M', 'Edge M',\
             r'$\theta$ [deg]', '$\omega$ [rpm]', 'tip x', 'typ y', 'tip z', 'J_rot', 'J_dt']
idx_plot = []

for i in range(len(idx_plot)):

    plt.figure()

    data = pd.read_csv(path, sep='\s+', skiprows = 1, names = col_names)
    x = R*data['speed']*np.pi/30/data['V']
    y = data[col_names[idx_plot[i]]]
    plt.plot(x, y, '-')

    plt.xlabel(r'$\lambda$ [-]')
    plt.ylabel(ylab[idx_plot[i]])
    plt.grid()

data = pd.read_csv(path, sep='\s+', skiprows = 1, names = col_names)
fig,ax = plt.subplots()
ax.plot( R*data['speed']*np.pi/30/data['V'], data['P'], color="red", marker="o")
ax.set_xlabel(r'$\lambda$ [-]')
ax.set_ylabel('$P$ [kW]',color="red")
# ax.set_ylim([min(data['P']), max(data['P'])])
ax2=ax.twinx()
ax2.plot(R*data['speed']*np.pi/30/data['V'], data['CP'],color="blue")
ax2.set_ylabel('$C_P$ [-]', color="blue", fontsize=14)
# ax2.set_ylim([min(data['CP']), max(data['CP'])])
ax.grid()


fig,ax = plt.subplots()
ax.plot( R*data['speed']*np.pi/30/data['V'], data['T'], color="red", marker="o")
ax.set_xlabel(r'$\lambda$ [-]')
ax.set_ylabel('$T$ [kN]',color="red")
ax.set_ylim([min(data['T']), max(data['T'])])
ax2=ax.twinx()
ax2.plot(R*data['speed']*np.pi/30/data['V'], data['CT'],color="blue",marker="o")
ax2.set_ylabel('$C_T$ [-]', color="blue", fontsize=14)
#ax2.set_ylim([min(data['CT']), max(data['CT'])])
ax.grid()