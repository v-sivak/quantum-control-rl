# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:58:24 2021

@author: qulab
"""
import os
import plot_config
import numpy as np 
import matplotlib.pyplot as plt

CF_files = [
    r'Z:\tmp\for Vlad\from_vlad\20210913.npz',
    r'Z:\tmp\for Vlad\from_vlad\20210914.npz',
    r'Z:\tmp\for Vlad\from_vlad\20210914.npz',
    ]

names = [
    r'$|+Z\rangle$',
    r'$|-Y\rangle$',
    r'$|-X\rangle$',
    ]

# Plot 2D Wigner
fig, axes = plt.subplots(1, 3, sharey=True, figsize=(3.375, 3), dpi=200)
for i in range(3):
    data = np.load(CF_files[i])
    CF, xs, ys = data['CF'], data['xs'], data['ys']
    
    if i == 2: CF = CF*0

    ax = axes[i] 
    if i == 1: ax.set_xlabel('Real')
    if i == 0: ax.set_ylabel('Imag')
    ax.set_aspect('equal')
    ax.set_title(names[i])
    ax.set_xticks([-3,0,3])
    ax.set_yticks([-3,0,3])
    
    plot_kwargs = dict(cmap='RdBu_r', vmin=-1, vmax=1)
    p = ax.pcolormesh(xs, ys, np.transpose(CF), **plot_kwargs)

savedir = r'E:\VladGoogleDrive\Qulab\GKP\experiment\figures'
savename = 'CF_tomo_square_code'
fig.savefig(os.path.join(savedir, savename), fmt='pdf')


# fig, ax = plt.subplots(1, 1, figsize=(4,3.375), dpi=600)
# plt.colorbar(p)
# ax.remove()
# savename = 'CF_tomo_square_code_colorbar'
# fig.savefig(os.path.join(savedir, savename), fmt='pdf')