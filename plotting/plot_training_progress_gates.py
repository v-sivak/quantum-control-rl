# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 20:55:17 2020
@author: Vladimir Sivak
"""

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# append parent 'gkp-rl' directory to path 
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import numpy as np
from numpy import sqrt, pi
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import plot_config



### Plot training progress
figname = r'E:\VladGoogleDrive\Qulab\GKP\paper\figs\gates_figure.pdf' # where to save

fig, ax = plt.subplots(1,1, figsize=(3.5, 2.0), dpi=300)
plt.grid(True)
ax.set_ylabel(r'$1-\cal F$')
ax.set_xlabel('Epoch')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim(80,10000)
ax.set_ylim(1.5e-3,1)
# ax.set_ylim(3e-4,1.3)
palette = plt.get_cmap('tab10')



all_root_dir = {
    r'$H$ (fock)' : r'E:\data\gkp_sims\PPO\paper_data\H_gate_fock',
    r'$X$ (fock)' : r'E:\data\gkp_sims\PPO\paper_data\X_gate_fock',
    r'$\sqrt{H}$ (gkp)' : r'E:\data\gkp_sims\PPO\paper_data\gates\sqrtH_gate_gkp_T1S80'
    }

colors = {
    r'$H$ (fock)' : palette(1),
    r'$X$ (fock)' : palette(2),
    r'$\sqrt{H}$ (gkp)' : palette(3)
    }

last_epoch_all = {
    r'$H$ (fock)' : 2000,
    r'$X$ (fock)' : 4000,
    r'$\sqrt{H}$ (gkp)' : 8000
    }

for gate, root_dir in all_root_dir.items():

    # load data from the log
    rewards, epochs = {}, {}
    
    for seed in os.listdir(root_dir):
        fname = os.path.join(root_dir, seed, 'log.hdf5')
        h5file = h5py.File(fname,'r+')
        grpname = list(h5file.keys())[-1] # pick the latest log
        try:
            grp = h5file[grpname]
            epochs[seed] = np.array(grp.get('epochs'))
            rewards[seed] = np.array(grp.get('returns'))
        finally:
            h5file.close()
    
    
    

    
    last_epoch = last_epoch_all[gate]
    train_epochs = np.array(epochs['seed0'])
    ind_last = np.where(train_epochs == last_epoch)[0][0]
    train_epochs = np.array(epochs['seed0'])[:ind_last]
    
    for seed in rewards.keys():
        ax.plot(train_epochs, 1-np.array(rewards[seed])[:ind_last], 
                linestyle='--', alpha=0.25, color=colors[gate])    
    
    all_seeds = np.array([rews for seed, rews in rewards.items()])[:,:ind_last]
    
    # calculate the best infidelity
    ind = np.argmax(all_seeds[:,-1])
    infidelity = 1 - all_seeds[ind,:]
    
    median_reward = np.median(all_seeds, axis=0)
    
    ax.plot(train_epochs, infidelity, color=colors[gate], linewidth=1.0)
    ax.plot(train_epochs, infidelity, color=colors[gate], linewidth=1.0,
            label=gate, linestyle='none', marker='.')




ax.legend(prop={'size': 8})
# fig.savefig(figname)