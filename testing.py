# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 11:09:04 2021

@author: Mathieu Pellé
"""

from design_functions import Aero_design
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline




Aero_design = Aero_design()

R_ref = 178.3/2
V_rated_ref = 11.4

Aero_design.Radius_Scaling(R_ref, V_rated_ref, 'A', 'B')
Aero_design.Airfoil_Tuning([0.38, 0.32, 0.47, 0.2], remove = ['cylinder.txt', 'FFA-W3-600.txt'], polars=False)
Aero_design.Fit_Polynomials([3, 3, 2, 6], R_ref, plotting=False)
Aero_design.Chord_Optimisation(B=3, TSR=6.75)
Aero_design.Limits_and_Smoothing(R_ref)

