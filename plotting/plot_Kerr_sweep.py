# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 16:43:29 2020

@author: qulab
"""
import matplotlib.pyplot as plt
import numpy as np
import os

datadir = r'E:\VladGoogleDrive\Qulab\GKP\sims\Kerr\hexagonal_sweep\new_scaling\250us_cavity'

T1_osc = 245
T1_qb = 50
t_rest = 1.2e-6 # readout + feedback delay 

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
datasets = ['no_rotation_50us_qubit', 'no_rotation_perfect_qubit', 
            'with_rotation_50us_qubit', 'with_rotation_perfect_qubit']
linestyles = ['--', '-.', '-', ':']
palette = plt.get_cmap('tab10')
colors = [palette(2), palette(9), palette(1), palette(3)]

fig, ax = plt.subplots(1,1, figsize=(7,4))
ax.set_title(r'Hexagonal code , $t_{gate}\propto 1\,/\,\sqrt[4]{Kerr}$')
ax.set_ylabel(r'Logical  lifetime ($\,\mu s\, $)')
ax.set_xlabel('Kerr (Hz)')
# ax.set_yscale('log')
# ax.set_ylim(0,2500)

Kerr = np.load(os.path.join(datadir, 'Kerr.npy'))
t_gate = np.load(os.path.join(datadir, 't_gate.npy'))

# Load and plot sweep results
for i, dataname in enumerate(datasets):
    for state in ['X+', 'Y+', 'Z+']:
        data = np.load(os.path.join(datadir, dataname, 'T_'+state[0]+'.npy'))
        label = datasets[i] if state=='Z+' else None
        ax.plot(Kerr, data, linestyle='-', color=colors[i],
                label=label)

# Plot various rates and bounds
rate_qubit_duty_cycle = 1/T1_qb * t_gate/(t_rest + t_gate) * 1/2 * 1/2
rate_perfect_qubit = 1/np.load(os.path.join(datadir, 'with_rotation_perfect_qubit', 'T_Z.npy'))
rate_combined =  rate_qubit_duty_cycle + rate_perfect_qubit

ax.plot(Kerr, [T1_osc]*len(Kerr), color='black', label=r'$1\,/\,T_1$ oscillator')
ax.plot(Kerr, 1/rate_qubit_duty_cycle, color='black', label=r'qubit duty cycle', linestyle=':')
ax.plot(Kerr, 1/rate_combined, color='black', label=r'combined', linestyle='--')
ax.plot([1],[210], color='purple', marker='^', label='experiment',linestyle='none')

# Shrink current axis by 35%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.65, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))