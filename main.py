# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 11:09:04 2021

@author: Mathieu Pellé
"""

#%% A E R O  D E S I G N

from design_functions import HAWC_design_functions

WT = HAWC_design_functions()

R_ref = 178.3/2
V_rated_ref = 11.4

WT.Radius_Scaling(R_ref, V_rated_ref, 'A', 'B')
WT.Airfoil_Tuning([0.38, 0.32, 0.47, 0.2], remove = ['cylinder.txt', 'FFA-W3-600.txt'], polars=False)
WT.Fit_Polynomials([3, 3, 2, 6], R_ref, plotting=False)
WT.Chord_Optimisation(B=3, TSR=6.75, plotting=False)
WT.Limits_and_Smoothing(R_ref)
WT.Make_ae_file('New_design')
WT.Make_htc(7.74, 0.07, R_ref, tsr = 7.25, omega_min=0)



WT.Define_htc_mode('controller_tuning', blade_dist=False, control_lst = [0.5, 0.7, 0.5, 0.7, 1, 1])



#%%

from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from  scipy.optimize  import  least_squares
from scipy.interpolate import CubicSpline
import shutil
import os
import re


#TODO look at structural part
#TODO look at modal analysis part
#TODO look at control part
#TODO look at loads part


