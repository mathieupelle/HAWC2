# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 18:00:17 2021

@author: MathieuPelle
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = '.\V2\ctrl_tuning\\f0.05_eta0.7.txt'
df = pd.read_csv(path, delimiter='\s+')
pitch = df['PI'].iloc[16:].astype(float)
true = df['generator'].iloc[16:].astype(float)
fit = df['torque'].iloc[16:].astype(float)

#df['DataFrame Column'] = df['DataFrame Column']

plt.figure()
plt.plot(pitch, true, 'x', label='True values')
plt.plot(pitch, fit, '--', label='Fit')
plt.legend()
plt.grid()

