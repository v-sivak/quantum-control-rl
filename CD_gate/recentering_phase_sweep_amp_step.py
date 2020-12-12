# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 11:09:57 2020

@author: qulab

Fit data to this Hamiltonian:
H = Delta * n + K/2 * n^2 + chi/2 * sigma_z * n + chi_prime/4 * sigma_z * n^2

"""


import numpy as np
from fpga_lib.scripting import get_experiment, wait_complete, get_last_results
from init_script import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from time import time
from scipy.optimize import curve_fit
import pickle
import os

from gkp_exp.CD_gate.recentering_phase_sweep import linear

exp_name = 'gkp_exp.CD_gate.recentering_phase_sweep'

points = 21
alphas = np.linspace(5, 15, points)

qubit_states = ['g', 'e']
phase_range_deg = dict(g=(-25,25,21), e=(-25,25,21))

results, fits = {}, {}
for s in qubit_states:
    results[s], fits[s] = [], []
    for alpha in alphas:
        exp = get_experiment(exp_name)
        exp.flip_qubit = True if s=='e' else False
        exp.phase_range_deg = phase_range_deg[s]
        exp.alpha = alpha
        exp.run()
        wait_complete(exp)
        results[s].append(exp.results)
        fits[s].append(exp.fit_params)


# Plot phase vs time for each alpha
fig, ax = plt.subplots(1,1, figsize=(9,5))
ax.set_xlabel('Time (ns)')
ax.set_ylabel('Return phase (deg)')
cmap = plt.get_cmap('Spectral_r')
norm = mpl.colors.Normalize(vmin = min(alphas), vmax = max(alphas))

for s in qubit_states:
    for i in range(points):
        mean_phase = results[s][i]['mean_phase'].data
        times = results[s][i]['mean_phase'].ax_data[0]
        color = cmap(norm(alphas[i]))
        ax.plot(times, mean_phase, color=color, linestyle='none', marker='.')
        ax.plot(times, linear(times, fits[s][i]['f0'], fits[s][i]['offset']), 
                linestyle='-', marker=None, color=color)


# Plot rotation frequency vs alpha
fig, ax = plt.subplots(1,1)
ax.set_xlabel('nbar')
ax.set_ylabel('Rotation frequency (kHz)')
freq = {}
nbar = np.array(alphas)**2
for s in qubit_states:
    freq[s] = np.array([fits[s][i]['f0'] for i in range(points)])
    ax.plot(nbar, freq[s]*1e-3)


# Plot average rotation frequency and fit Kerr and detuning
avg_freq = (freq['g'] + freq['e'])/2.0
def avg_freq_fit_func(n, Kerr, Delta):
    return  Delta + Kerr * n
popt, pcov = curve_fit(avg_freq_fit_func, nbar, avg_freq)
Kerr, Delta = popt

fig, ax = plt.subplots(1,1)
ax.set_xlabel('nbar')
ax.set_ylabel('Average rotation frequency (kHz)')
ax.plot(nbar, avg_freq*1e-3, marker='.',linestyle='none')
ax.plot(nbar, avg_freq_fit_func(nbar, *popt)*1e-3,
        label='K=%.1f Hz, Delta= %.1f Hz' %(Kerr,Delta))
ax.legend()


# Plot difference rotation frequency and fit chi and chi_prime
diff_freq = freq['g'] - freq['e']
def diff_freq_fit_func(n, chi, chi_prime):
    return  chi + chi_prime * n
popt, pcov = curve_fit(diff_freq_fit_func, nbar, diff_freq)
chi, chi_prime = popt

fig, ax = plt.subplots(1,1)
ax.set_xlabel('nbar')
ax.set_ylabel('Difference rotation frequency (kHz)')
ax.plot(nbar, diff_freq*1e-3, marker='.',linestyle='none')
ax.plot(nbar, diff_freq_fit_func(nbar, *popt)*1e-3,
        label='chi=%.1f kHz, chi_prime= %.1f Hz' %(chi*1e-3,chi_prime))
ax.legend()