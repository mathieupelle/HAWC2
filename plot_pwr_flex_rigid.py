#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 11:12:32 2021

@author: ozgo

Plot pwr files without wetb
"""
# ============================================================================
import numpy as np
from matplotlib import pyplot as plt

# ============================================================================
# INPUTS
# ============================================================================
f_name_1 = './V2/results/hawc2s/ind/rigid/V2_hs2.pwr'
f_name_2 = './V2/results/hawc2s/ind/flex/V2_hs2.pwr'
#
p_lab = ['Rigid','Flex']
ch_i = [1,2,8,10,11,12] #[10,11,8,6] #
# ============================================================================
df_1 = np.loadtxt(f_name_1, skiprows=1)
df_2 = np.loadtxt(f_name_2, skiprows=1)
with open(f_name_1) as f:
    a = f.readline()
b = list(filter(None,a.split('#')[1].split('   ')))
col_name = [' '.join(i.split()[:-1]) for i in b]
# ============================================================================
# PLOTS
#============================================================================
px = np.zeros((df_1.shape[0],len(ch_i),2))
py = np.zeros((df_2.shape[0],len(ch_i),2))
#
for i in range(len(ch_i)):
    px[:,i,0] = df_1[:,0].copy()
    px[:,i,1] = df_2[:,0].copy()
    py[:,i,0] = df_1[:,ch_i[i]].copy() #- a[i]
    py[:,i,1] = df_2[:,ch_i[i]].copy() #- b[i]
#
ylab = []
xlab = []
for i in range(len(ch_i)):
    ylab += ['%s' %col_name[ch_i[i]]]
    xlab += ['Wind speed [m/s]']
p_clr = ['#C4000D','#000000','#1F3DFF','#C4000D','#C4000D','#1F3DFF',]
p_mrk   = ['o','s','v','o','s','v']
p_mrk_f = ['full','full','full','none','none','none']
p_line = ['solid','solid','solid',(0, (5, 5)),(0, (5, 5)),(0, (5, 5))] #(0, (1, 10)),
p_mrk_s = [3.0, 3.0, 3.0, 4.0, 4.0, 4.0]
p_mrk_e = [1.0, 1.0, 1.0, 1.5, 1.5, 1.5]
p_mrk_every_t = [50, 50, 50, 50, 50, 50]
p_mrk_every_s = [1, 1, 1, 1, 1, 1]
p_line_t = [2, 2, 2, 2, 2, 2]
#

font_size = 10
p_size = [16,8]
p_adjust=[0.07,0.13,0.98,0.98]
#
fig, axs = plt.subplots(2,int(len(ch_i)/2))
fig.set_size_inches(p_size[0],p_size[1])
fig.subplots_adjust(left=p_adjust[0], bottom=p_adjust[1],
                    right=p_adjust[2], top=p_adjust[3],wspace = 0.17)
#
i_c = 0
for j_i in range(2):
    for j in range(int(len(ch_i)/2)):
        i = 0
        axs[j_i,j].plot(px[:,i_c,0], py[:,i_c,0],marker=p_mrk[i],label=p_lab[i],
                         markersize=p_mrk_s[i],linestyle=p_line[i],
                         fillstyle=p_mrk_f[i],color=p_clr[i],
                         markeredgewidth=p_mrk_e[i],linewidth=p_line_t[i])
        i = 1
        axs[j_i,j].plot(px[:,i_c,1], py[:,i_c,1],marker=p_mrk[i],label=p_lab[i],
                         markersize=p_mrk_s[i],linestyle=p_line[i],
                         fillstyle=p_mrk_f[i],color=p_clr[i],
                         markeredgewidth=p_mrk_e[i],linewidth=p_line_t[i])
        axs[j_i,j].set_ylabel(ylab[i_c], fontsize=font_size)
        axs[j_i,j].legend(bbox_to_anchor=(0.0, 0.99),fontsize=font_size
           ,loc=2, ncol=1)
        i_c = i_c +1
#
#
for i in range(int(len(ch_i)/2)):
    axs[1,i].set_xlabel(xlab[i], fontsize=font_size)
#
for ax in fig.get_axes():
    ax.tick_params(axis='both', labelsize=font_size)
    ax.grid(which='major',axis='both', linestyle=':', linewidth=1)

