# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 11:09:04 2021

@author: Mathieu Pellé
"""

#%% A E R O  D E S I G N

# Loading class
from design_functions import HAWC_design_functions
import numpy as np
import matplotlib.pyplot as plt
WT = HAWC_design_functions()

R_ref = 178.3/2 # Reference radius
V_rated_ref = 11.4 # Reference rated speed

# Radius scaling
WT.Radius_Scaling(R_ref, V_rated_ref, 'A', 'B')


# Airfoil design point selection
WT.Airfoil_Tuning([0.28, 0.30, 0.47, 0.2], remove = ['cylinder.txt', 'FFA-W3-600.txt'], polars=True)

# Aerodynamic design polynomials
WT.Fit_Polynomials([3, 3, 2, 6], plotting=True)

# Looping over TSRs
# tsr_lst = np.arange(6.5, 10, 0.25)
# CP_lst = []
# CT_lst = []
# for i in range(len(tsr_lst)):
#     WT.Chord_Optimisation(B=3, TSR=tsr_lst[i], plotting=False)
#     CP_lst.append(WT.CP)
#     CT_lst.append(WT.CT)

# fig, axs = plt.subplots(2)
# axs[0].plot(tsr_lst, CT_lst)
# axs[0].set(xlabel =r'$\lambda$', ylabel='$C_T$')
# axs[0].grid()
# axs[1].plot(tsr_lst, CP_lst)
# axs[1].set(xlabel =r'$\lambda$', ylabel='$C_P$')
# axs[1].grid()

# Chord optimisation and fixing
WT.Chord_Optimisation(B=3, TSR=7.75, plotting=False)
WT.Limits_and_Smoothing(plotting=True, spline_plot=True)

# Generating ae, htc and structural files
# htc modes: generate_opt, controller_tuning
WT.Make_ae_file('V2')
WT.Make_htc_steady(9.0156, 0.07, tsr = 7.75, omega_min=5.6097)
WT.Make_st_file()

# For tsr case...
# WT.Generate_tsr_opt(np.arange(6,10,0.25),8)
# WT.Define_htc_steady_mode('standard', blade_distributions=False, properties=False, path_opt='./data/V2_hs2_tsr.opt')

# For opt file...
# WT.Define_htc_steady_mode('generate_opt', blade_distributions=False, properties=False)

#Apply peak shaving ONLY RUN ONCE!!!!
#WT.Apply_peak_shaving('data/V2_hs2.opt')

# For pwr and ind files without regenerating opt...
#WT.Define_htc_steady_mode('standard', blade_distributions=True, properties=False, path_opt='./data/V2_hs2.opt')

#%% S T R U C T U R E S

# For generating rigid opts and rhtc...
#WT.Define_htc_steady_mode('generate_opt', blade_distributions=True, properties=False, rigid=True)
#WT.Define_htc_steady_mode('standard', blade_distributions=True, properties=False, rigid=True, path_opt='./data/V2_hs2_rigid.opt')


#%% C O N T R O L

# freq1, damp1, freq2, damp2, gain_scheduling, control_type

# Generates htc for controller tuning using opt file
WT.Define_htc_steady_mode('controller_tuning', blade_distributions=False, \
                            control_lst = [0.05, 0.7, 0.05, 0.7, 2, 1], properties=False)

# # Generates htc fo unsteady control simulation
# WT.Make_htc_unsteady('./V2/ctrl_tuning/V2_hs2_ctrl_tuning.txt')

#%%


from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from  scipy.optimize  import  least_squares
from scipy.interpolate import CubicSpline
import shutil
import pandas as pd
import os
import re




#%%


