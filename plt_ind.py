# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 13:52:29 2021

@author: Mathieu Pellé
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


f1 = './DTU10MW/results/steady/ind/DTU_10MW_RWT_hs2_u8000.ind'
f2 = './New_design/New_design_hs2_u8000.ind'

paths =  [f1, f2]
leg_names = ['DTU 10MW', 'Redesign 1']
col_names = ['s', 'a', 'ap', 'phi', 'alpha', 'U0', 'FX0', 'FY0',\
             'M0', 'UX0', 'UY0' , 'UZ0', 'twist', 'X_AC0', 'Y_AC0', 'Z_AC0',\
             'Cl', 'Cd', 'Cm', 'Clp0', 'Cdp0', 'Cmp0', 'F', 'Fp', 'CL_FS',\
             'Cl_FSp', 'Va', 'Vt', 'Tors', 'vx', 'vy', 'chord', 'CT', 'CP', 'angle', 'v_1',\
            'v_2', 'v_3']
ylab = ['s', 'a [-]', 'ap [-]', r'$phi_0$ [deg]', r'$\alpha_0$ [deg]', 'U0', 'FX0', 'FY0',\
        'M0', 'UX0', 'UY0' , 'UZ0', r'$\beta$', 'X_AC0', 'Y_AC0', 'Z_AC0',\
        '$C_l$ [-]', '$C_d$ [-]', 'Cm', 'Clp0', 'Cdp0', 'Cmp0', 'F', 'Fp', 'CL_FS',\
        'CFLps', 'Va', 'Vt', 'Tors', 'vx', 'vy', 'chord', '$C_T$ [-]', '$C_P$ [-]', 'angle', 'v_1',\
        'v_2', 'v_3']
idx_plot = [1, 2, 3, 4, 16, 17, 32, 33]
for i in range(len(idx_plot)):

    plt.figure()
    for j in range(len(paths)):
        data = pd.read_csv(paths[j], sep='\s+', skiprows = 1, names = col_names)
        x = data['s']
        y = data[col_names[idx_plot[i]]]
        plt.plot(x, y, '-', label=leg_names[j])

    plt.xlabel('r/R [-]')
    plt.ylabel(ylab[idx_plot[i]])
    plt.legend()
    plt.grid()

