# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 11:52:05 2021
"""
import numpy as np
import matplotlib.pyplot as plt
import plot_config

RL = np.array(
    [0.9995864033699036,
     0.9989489316940308,
     0.9974895119667053,
     0.9974268078804016,
     0.9979210495948792,
     0.9966591596603394,
     0.9968177080154419,
     0.994166374206543,
     0.99238121509552,
     0.9928078055381775])

NM = np.array(
    [0.9978455305099487,
     0.9895886778831482,
     0.9725096225738525,
     0.8997492790222168,
     0.8213904500007629,
     0.7750567197799683,
     0.6290889382362366,
     0.5851404666900635,
     0.18651925027370453,
     0.012788497842848301])

SA = np.array(
    [0.9879633784294128,
     0.9398494958877563,
     0.8676950335502625,
     0.8264684677124023,
     0.6265802383422852,
     0.6686856746673584,
     0.7009228467941284,
     0.5255513191223145,
     0.5650225281715393,
     0.5919240713119507])



barWidth = 0.25
range_RL = np.arange(1,11,1)-barWidth
range_NM = range_RL + barWidth
range_SA = range_NM + barWidth


fig, ax = plt.subplots(1,1, figsize=(3.5, 2), dpi=200)
plt.grid(True, axis='y')
ax.set_yscale('log')
ax.set_ylim(3e-4,1.3)
ax.set_xticks(range(1,11))
ax.set_ylabel('Fidelity')
ax.set_xlabel('Fock state')
palette = plt.get_cmap('tab10')


for i in range(1,11):
    
    ax.fill_between([i-3/2*barWidth, i-1/2*barWidth], 1-RL[i-1], y2=1, 
                    color=palette(0), label='RL' if i==1 else None, linewidth=0, zorder=4)
    ax.fill_between([i-1/2*barWidth, i+1/2*barWidth], 1-NM[i-1], y2=1, 
                    color=palette(1), label='NM' if i==1 else None, linewidth=0, zorder=4)
    ax.fill_between([i+1/2*barWidth, i+3/2*barWidth], 1-SA[i-1], y2=1, 
                    color=palette(2), label='SA' if i==1 else None, linewidth=0, zorder=4)

ax.legend()

savename = r'E:\VladGoogleDrive\Qulab\GKP\paper\figs\bar_plot.pdf'
fig.savefig(savename)