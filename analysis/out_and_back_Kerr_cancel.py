# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 15:47:06 2021

@author: qulab
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from fpga_lib import *
from fpga_lib.dsl.result import Results
import plot_config


from fpga_lib.scripting import wait_complete, get_last_results, connect_to_gui, get_gui, get_experiment
connect_to_gui()
get_gui()
#time = np.array([48, 72, 96, 120, 144, 168, 192, 216, 240])
time = np.array([100,200,300,400,500,600])
nbar_range = (10, 40, 11)
nbar_points = nbar_range[-1]
nbar = np.linspace(*nbar_range)
phase = []

s = 'g'

for tau in time:       
    exp = get_experiment('gkp_exp.fpga_exp.out_and_back_Kerr_cancel', from_gui=True)
    exp.set_params({'do_e' : False, 'do_g' : False})
    exp.set_params({'Kerr_pump_time_ns' : tau,
                    'nbar_range' : nbar_range,
                    'do_'+s : True})
    exp.run()
    wait_complete(exp)
    res = get_last_results(exp)
    phase.append(res['sz_'+s+'_postselected_'+s+'_fit_gaussian_results'].data)

phase = np.array(phase)



# fit rotation frequency
def linear(t, freq, offset):
    return 360*freq*t*1e-9 + offset

freq_fits = []
offset_fits = []
for i in range(nbar_points):
    popt, pcov = curve_fit(linear, time, phase[:,i])
    freq_fits.append(popt[0])
    offset_fits.append(popt[1])

freq_fits = np.array(freq_fits)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------



# Plot "Phase vs nbar" for all times in one panel.
fig, ax = plt.subplots(1,1, figsize=(3.375,2), dpi=200)
ax.set_xlabel('nbar')
ax.set_ylabel('Return phase (deg)')
cmap = plt.get_cmap('Spectral_r')
norm = mpl.colors.Normalize(vmin = min(time), vmax = max(time))

for i in range(len(time)):
    mean_phase = phase[i,:]
    color = cmap(norm(time[i]))
    ax.plot(nbar, mean_phase, color=color, linestyle='none', marker='.',
            label=str(time[i]))
ax.legend()
plt.grid()
plt.tight_layout()


# Plot "rotation frequency vs nbar" and linear fit with extracted Kerr
def linear_fit(n, c0, c1):
    return c0 + c1*n
popt, _ = curve_fit(linear_fit, nbar, freq_fits)

fig, ax = plt.subplots(1,1,figsize=(3.375,2), dpi=200)
ax.set_xlabel('nbar')
ax.set_ylabel('Rotation freq (kHz)')
ax.plot(nbar, freq_fits*1e-3, marker='.', linestyle='none')
ax.plot(nbar, linear_fit(nbar, *popt)*1e-3, label=r'$K_g$=%.0f Hz' %popt[1])
ax.legend(loc='lower right')
plt.tight_layout()



# Plot "phase vs time" for all nbars in one panel. This shows that rotation
# is indeed linear in time, so it can be characterized by a angular frequency
fig, ax = plt.subplots(1,1, figsize=(3.375,2), dpi=200)
ax.set_xlabel('Time (ns)')
ax.set_ylabel('Return phase (deg)')
cmap = plt.get_cmap('Spectral_r')
norm = mpl.colors.Normalize(vmin = min(nbar), vmax = max(nbar))

for i in range(nbar_points):
    mean_phase = phase[:,i]
    color = cmap(norm(nbar[i]))
    ax.plot(time, mean_phase, color=color, linestyle='none', marker='.',
            markersize=10)
    ax.plot(time, linear(time, freq_fits[i], offset_fits[i]), 
            linestyle='-', marker=None, color=color)


# Plot "phase vs time" for all nbars in separate panels
fig, axes = plt.subplots(3,4, sharex=True, sharey=True, figsize=(25,14))
axes = axes.ravel()
for j in range(11):
    for s in ['g']:
        for i in [j]:
            mean_phase = phase[:,i]
            axes[j].plot(time, mean_phase, color='black', linestyle='none', marker='o')
            axes[j].plot(time, linear(time, freq_fits[i], offset_fits[i]), 
                    linestyle='-', marker=None, color='red')
plt.tight_layout()
