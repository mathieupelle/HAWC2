# -*- coding: utf-8 -*-
"""Calculate the annual energy production
"""
import re
import matplotlib.pyplot as plt
import numpy as np
from _loads_utils import load_stats


stat_dir = 'C:/Users/Mathieu Pellé/Documents/GitHub/LAC_RotorDesign/Loads/res_turb/'  # results directory with statistics files  !!! END WITH SLASH !!!
v_ref = 37.5  # reference wind speed based on wind class (I=50, 2=42.5, 3=37.5)
i_wind = 15  # channel number with the wind speed
i_pow = 100  # channel number for electrical power

# dictionary to map .sel index to ylabel for the plot
ylabels = {4: 'Pitch angle [deg]',
           10: 'Rotor speed [rad/s]',
           13: 'Thrust [kN]',
           15: 'Wind speed [m/s]',
           17: 'Tower-base FA [kNm]',
           18: 'Tower-base SS [kNm]',
           20: 'Yaw-bearing pitch [kNm]',
           22: 'Yaw-bearing roll [kNm]',
           25: 'Shaft torsion [kNm]',
           26: 'OoP BRM [kNm]',
           27: 'IP BRM [kNm]',
           70: 'Generator torque [Nm]',
           100: 'Electrical power [W]',
           108: 'Tower clearance [m]'}

# load the mean statistics for wind speed and power
stat_file = stat_dir + 'stats_mean.txt'
files, idxs, data = load_stats(stat_file)
wind = data[:, idxs == i_wind].squeeze()
power = data[:, idxs == i_pow].squeeze()

# extract the set wind speed value from the filename using regex tricks
wsps = [float(re.findall('[0-9]{1,2}[.][0-9]', f)[0]) for f in files]

# calculate the average power in a wind speed bin
wsp_unique = np.unique(wsps)
delta_v = wsp_unique[1] - wsp_unique[0]
pows = np.empty(wsp_unique.size)  # mean power at each wind speed
for j, vj in enumerate(wsp_unique):
    # isolate the dels from each simulation
    wsp_pows = power[np.isclose(wsps, vj)]  # powers for that wind speed
    p = 1/wsp_pows.size  # probability of each simulation in the wsp bin is equal 1/nsim
    pows[j] = sum(p * wsp_pows)  # this is actually just a mean, really


# calculate the annual energy production
v_ave = 0.2*v_ref  # v_ave=0.2*vref
hrs_per_year = 365 * 24  # hours per year
dvj = wsp_unique[1] - wsp_unique[0]  # assuming even bins!
probs = (np.exp(-np.pi*((wsp_unique - dvj/2) / (2*v_ave))**2)
          - np.exp(-np.pi*((wsp_unique + dvj/2) / (2*v_ave))**2))  # prob of wind in each bin
aep = hrs_per_year * sum(probs * pows)  # sum weighted power and convert to AEP (Wh)
print(f'The AEP is: {aep/(1e6):.1f} MWh')

# make the plot
fig, ax1 = plt.subplots(1, 1, num=1, figsize=(7, 3), clear=True)
plt.plot(wind, power, 'o', zorder=10)  # 10-min means
plt.plot(wsp_unique, pows, 'or', mec='0.2', ms=7, alpha=0.9, zorder=11)  # bin-average
plt.grid('on')
plt.xlabel('Wind speed [m/s]')
plt.ylabel(ylabels[i_pow])
# bar plot with probabilities
ax2 = ax1.twinx()  # new axis with shared x
ax2.bar(wsp_unique, probs, facecolor='0.8', edgecolor='0.4', alpha=0.7, zorder=-2)
ax2.set_yticks([])
ax1.set_zorder(1)  # magic to put bars under power
ax1.patch.set_visible(False)  # prevent ax1 from hiding ax2
plt.tight_layout()
