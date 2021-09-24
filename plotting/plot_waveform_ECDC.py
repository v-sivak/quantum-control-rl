# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 14:50:42 2021

@author: qulab
"""
import numpy as np
import matplotlib.pyplot as plt
import plot_config

filename = r'Z:\tmp\for Vlad\from_vlad\plus_Z_2.npz'
data = np.load(filename, allow_pickle=True)
c_pulse, q_pulse = data['c_pulse'], data['q_pulse']

time = np.arange(len(c_pulse))

fig, ax = plt.subplots(1, 1, figsize=(3.375, 1), sharex=True, dpi=200)
colors = plt.get_cmap('Paired')
# ax.grid(True)
ax.set_ylabel('DAC amplitude')
ax.plot(time, c_pulse.real, label='I', color=colors(1))
ax.plot(time, c_pulse.imag, label='Q', color=colors(0))
ax.set_xlabel('Time (ns)')
ax.plot(time, q_pulse.real, label='I', color=colors(5))
ax.plot(time, q_pulse.imag, label='Q', color=colors(4))

ax.legend()
savename = r'E:\VladGoogleDrive\Qulab\GKP\experiment\figures\gkp_Z_pulse.pdf'
fig.savefig(savename)