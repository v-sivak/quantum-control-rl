# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 17:55:40 2021

@author: qulab
"""

import matplotlib.pyplot as plt
import numpy as np
import plot_config

fig, ax = plt.subplots(1,1, figsize=(2,3), dpi=300)
figname = r'E:\VladGoogleDrive\Qulab\GKP\paper\figs\pulse_spectrum.pdf'
x = np.arange(800)/100-4
for i in range(-4,5):
    ax.vlines(i, -0.4, 1.2, color='grey')
ax.plot(x, np.sinc(x*100),label='1')
ax.plot(x, np.sinc(x*3.4),label='2')
ax.plot(x, np.sinc(x*0.4),label='3')
ax.axis('off')
ax.legend()
fig.savefig(figname)