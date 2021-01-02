# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:07:06 2020

@author: qulab
"""

import os
from fpga_lib import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

## load results of old experiment
#exp_dir = r'D:\DATA\exp\gkp_exp.CD_gate.out_and_back_amp_phase_sweep\archive'
#fname = '20210101.h5'
#grp_name = '13'
#
#file_name = os.path.join(exp_dir, fname)
#res = Results.create_from_file(file_name, grp_name)
#
#time = 160 * 1e-9 
## 160 ns is actual delay, 2*12 is due to padding of the fast displacements 
## to match minimum pulse length of 24 ns. there is half padding on forward and
## backward displacements.
#freq_e = res['sz_e_postselected_e_fit_gaussian_results'] / 360.0 / time # Hz
#freq_g = res['sz_g_postselected_g_fit_gaussian_results'] / 360.0 / time # Hz
#
#fig, ax = plt.subplots(1,1)
#ax.set_xlabel('nbar')
#ax.set_ylabel('Frequency (kHz)')
#plt.plot(freq_e.ax_data[0], freq_e.data*1e-3, marker='.')
#plt.plot(freq_g.ax_data[0], freq_g.data*1e-3, marker='.')



exp_dir = r'D:\DATA\exp\gkp_exp.CD_gate.out_and_back_amp_phase_sweep\archive'
fname = '20210101.h5'
file_name = os.path.join(exp_dir, fname)


qubit_states = ['g', 'e']
groups = [6, 5, 7, 4, 8, 9, 10, 11, 12, 13]
time = np.array([24, 48, 72, 100, 120, 144, 168, 192, 216, 240]) # [ns]
nbar = np.linspace(1, 361, 46)

time_points = len(time)
nbar_points = len(nbar)

phase = {s:[] for s in qubit_states}
for i in range(time_points):
    grp_name = str(groups[i])
    res = Results.create_from_file(file_name, grp_name)
    phase['e'].append(res['sz_e_postselected_e_fit_gaussian_results'].data)
    phase['g'].append(res['sz_g_postselected_g_fit_gaussian_results'].data)
    
phase = {s:np.array(phase[s]) for s in qubit_states}


# fit rotation frequency
def linear(t, freq, offset):
    return 360*freq*t*1e-9 + offset

freq_fits = {s:[] for s in qubit_states}
offset_fits = {s:[] for s in qubit_states}
for s in qubit_states:
    for i in range(nbar_points):
        popt, pcov = curve_fit(linear, time, phase[s][:,i])
        freq_fits[s].append(popt[0])
        offset_fits[s].append(popt[1])

# Plot phase vs time for each alpha
fig, ax = plt.subplots(1,1, figsize=(9,5))
ax.set_xlabel('Time (ns)')
ax.set_ylabel('Return phase (deg)')
cmap = plt.get_cmap('Spectral_r')
norm = mpl.colors.Normalize(vmin = min(nbar), vmax = max(nbar))

for s in qubit_states:
    for i in range(nbar_points):
        mean_phase = phase[s][:,i]
        color = cmap(norm(nbar[i]))
        ax.plot(time, mean_phase, color=color, linestyle='none', marker='.')
        ax.plot(time, linear(time, freq_fits[s][i], offset_fits[s][i]), 
                linestyle='-', marker=None, color=color)


# Plot rotation frequency vs nbar
fig, ax = plt.subplots(1,1)
ax.set_xlabel('nbar')
ax.set_ylabel('Rotation frequency (kHz)')
freq_fits = {s:np.array(freq_fits[s]) for s in qubit_states}
for s in qubit_states:
    ax.plot(nbar, freq_fits[s]*1e-3, marker='o')


# Plot linear fit for all nbars in separate panels
fig, axes = plt.subplots(7,7, sharex=True, sharey=True, figsize=(25,14))
axes = axes.ravel()
for j in range(46):
    for s in ['g']:
        for i in [j]:
            mean_phase = phase[s][:,i]
            axes[j].plot(time, mean_phase, color='black', linestyle='none', marker='o')
            axes[j].plot(time, linear(time, freq_fits[s][i], offset_fits[s][i]), 
                    linestyle='-', marker=None, color='red')
plt.tight_layout()

