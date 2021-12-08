# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 14:04:32 2021

@author: MathieuPelle
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

rootV1 = './V1/results/hawc2/airfoil/dtu_10mw_rwt_05.'
rootV2 = './V2/results/hawc2/airfoil/v2_turb_08.'


seeds1 = ['0_21716', '0_40322', '0_43619', '0_47165', '0_47952', '0_50292']
seeds2 = ['0_15536', '0_22441', '0_32293', '0_41120', '0_46290', '0_60415']
paths = [rootV2+seeds2[4]+'.dat']#, rootV1+seeds1[5]+'.dat']

plt.figure(figsize=(7, 3))
plt.grid()
plt.xlabel(r'$\alpha$ [deg]')
plt.ylabel('$C_l$ [-]')
for i in range(len(paths)):
    data = np.loadtxt(paths[i])

    alpha = data[:,60]
    cl = data[:,63]
    plt.plot(alpha,cl, label = 'Unsteady polar',  color='purple')


aero_data_path = './Aerodynamics/Airfoil data/FFA-W3-241.txt'
data_af = np.loadtxt(aero_data_path)
idx1 = 52
idx2 = 65
plt.plot(data_af[idx1:idx2,0], data_af[idx1:idx2,1],'--k', label='Steady polar')
plt.legend()
