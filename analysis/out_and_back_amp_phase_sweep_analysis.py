# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:07:06 2020

@author: qulab
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from fpga_lib import *
from fpga_lib.dsl.result import Results

# Run a series of experiments for different values of the delay time 'tau'.
# In each experiment, cavity rotation angle accumulated during this time
# will be extracted as a function of nbar. 
if 1:
    from fpga_lib.scripting import wait_complete, get_last_results, connect_to_gui, get_gui, get_experiment
    connect_to_gui()
    get_gui()
    qubit_states = ['g', 'e']
#    time = np.array([48, 72, 96, 120, 144, 168, 192, 216, 240])
    time = np.array([48, 96,144, 192, 240])
#    time = np.array([48, 144, 240])
    nbar_range = (4, 400, 45)
    nbar_points = nbar_range[-1]
    nbar = np.linspace(*nbar_range)
    phase = {s:[] for s in qubit_states}
    
    for tau in time:       
        exp = get_experiment('gkp_exp.CD_gate.out_and_back_amp_phase_sweep', from_gui=True)
        exp.set_params({'static_tau' : tau,
                        'nbar_range' : nbar_range})
        exp.run()
        wait_complete(exp)
        res = get_last_results(exp)
        phase['e'].append(res['sz_e_postselected_e_fit_gaussian_results'].data)
        phase['g'].append(res['sz_g_postselected_g_fit_gaussian_results'].data)
    
    phase = {s:np.array(phase[s]) for s in qubit_states}
    

# If the experiments were done manually from GUI, this will load them from archive
if 0:
    exp_dir = r'D:\DATA\exp\gkp_exp.CD_gate.out_and_back_amp_phase_sweep\archive'
    fname = '20210423.h5'
    file_name = os.path.join(exp_dir, fname)
    
    qubit_states = ['g', 'e']
    groups = [1, 3, 5, 8, 7, 6, 4, 2, 0]
    time = np.array([48, 72, 100, 120, 144, 168, 192, 216, 240]) # [ns]
    nbar = np.linspace(4, 400, 45)
    
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

np.save(r'Y:\tmp\for Vlad\from_vlad\nbar.npy', nbar)
np.save(r'Y:\tmp\for Vlad\from_vlad\freq_e.npy', freq_fits['e'])
np.save(r'Y:\tmp\for Vlad\from_vlad\freq_g.npy', freq_fits['g'])

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


# Plot linear fit for all nbars in separate panels
fig, axes = plt.subplots(7,7, sharex=True, sharey=True, figsize=(25,14))
axes = axes.ravel()
for j in range(45):
    for s in ['g']:
        for i in [j]:
            mean_phase = phase[s][:,i]
            axes[j].plot(time, mean_phase, color='black', linestyle='none', marker='o')
            axes[j].plot(time, linear(time, freq_fits[s][i], offset_fits[s][i]), 
                    linestyle='-', marker=None, color='red')
plt.tight_layout()




#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

freq_g, freq_e = np.array(freq_fits['g']), np.array(freq_fits['e'])
# Plot cavity rotation frequency as a function of nbar
fig, ax = plt.subplots(1,1,figsize=(3.375,2), dpi=200)
ax.set_xlabel('nbar')
ax.set_ylabel('Rotation freq (kHz)')
ax.plot(nbar, freq_g*1e-3, marker='.', linestyle='none', label='g')
ax.plot(nbar, freq_e*1e-3, marker='.', linestyle='none', label='e')
ax.legend(loc='lower right')
plt.tight_layout()


# Plot average rotation frequency and fit Kerr and detuning
avg_freq = (freq_g + freq_e)/2.0
def avg_freq_fit_func(n, Kerr, Delta):
    return  Delta + Kerr * n
fit_pts = 45
popt, pcov = curve_fit(avg_freq_fit_func, nbar[:fit_pts], avg_freq[:fit_pts])
Kerr, Delta = popt

fig, ax = plt.subplots(1,1,figsize=(3.375,2.2), dpi=200)
ax.set_xlabel('nbar')
ax.set_ylabel('Avg rotation freq (kHz)')
ax.plot(nbar, avg_freq*1e-3, marker='.',linestyle='none')
ax.plot(nbar[:fit_pts], avg_freq_fit_func(nbar[:fit_pts], *popt)*1e-3,
        label=r'K=%.0f Hz, $\Delta$= %.1f kHz' %(Kerr,Delta*1e-3))
ax.legend()
plt.tight_layout()


# Plot difference rotation frequency and fit chi and chi_prime
diff_freq = freq_g - freq_e
def diff_freq_fit_func(n, chi, chi_prime):
    return  chi + chi_prime * n
popt, pcov = curve_fit(diff_freq_fit_func, nbar[:fit_pts], diff_freq[:fit_pts])
chi, chi_prime = popt

fig, ax = plt.subplots(1,1,figsize=(3.375,2.2), dpi=200)
ax.set_xlabel('nbar')
ax.set_ylabel('Diff rotation freq (kHz)')
ax.plot(nbar, diff_freq*1e-3, marker='.',linestyle='none')
ax.plot(nbar[:fit_pts], diff_freq_fit_func(nbar[:fit_pts], *popt)*1e-3,
        label='chi=%.0f kHz, chi_prime= %.0f Hz' %(chi*1e-3,chi_prime))
ax.legend()
plt.tight_layout()


# Plot quartic fits to 'e' and 'g' rotation frequency
def quartic_fit(n, c0, c1, c2, c3, c4):
    return c0 + c1*n + c2*n**2 + c3*n**3 + c4*n**4
popt_g, _ = curve_fit(quartic_fit, nbar, freq_g)
popt_e, _ = curve_fit(quartic_fit, nbar, freq_e)

fig, ax = plt.subplots(1,1,figsize=(3.375,2), dpi=200)
ax.set_xlabel('nbar')
ax.set_ylabel('Rotation freq (kHz)')
ax.plot(nbar, freq_g*1e-3, marker='.', linestyle='none', label='g')
ax.plot(nbar, freq_e*1e-3, marker='.', linestyle='none', label='e')
ax.plot(nbar, quartic_fit(nbar, *popt_g)*1e-3)
ax.plot(nbar, quartic_fit(nbar, *popt_e)*1e-3)
ax.legend(loc='lower right')
plt.tight_layout()