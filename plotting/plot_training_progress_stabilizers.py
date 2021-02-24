# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 13:43:26 2020

@author: Vladimir Sivak
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import plot_config

root_dir = r'E:\data\gkp_sims\PPO\paper_data\gkp_stabilizers_small_nn'

log = {}
for sim_name in os.listdir(root_dir):
    fname = os.path.join(root_dir, sim_name, 'log.hdf5')
    h5file = h5py.File(fname,'r+')
    grpname = list(h5file.keys())[-1] # pick the latest log
    log[sim_name] = {}
    try:
        grp = h5file[grpname]
        log[sim_name]['epochs'] = np.array(grp.get('epochs'))
        log[sim_name]['returns'] = np.array(grp.get('returns'))
    finally:
        h5file.close()

# Plot return vs epoch number
figname = r'E:\VladGoogleDrive\Qulab\GKP\paper\figs\gkp_from_stabilizers.pdf' # where to save
fig, ax = plt.subplots(1,1, figsize=(3.375, 2.1), dpi=300)
ax.set_ylabel(r'Avg. stabilizer')
ax.set_xlabel('Epoch')
ax.set_ylim(-0.2,1)
plt.grid(True)
cmap = plt.get_cmap('tab10')
color = cmap(5)

# in the background, plot all random seeds
for sim_name in log.keys():
    epochs = log[sim_name]['epochs']
    ind = np.arange(0,len(epochs),4)
    ax.plot(epochs[ind], log[sim_name]['returns'][ind],
            linestyle='--', color=color, alpha=0.3)
    
# calculate the best return
all_seeds = np.array(list(log[i]['returns'] for i in log.keys()))
ind = np.argmax(all_seeds[:,-1])
returns = all_seeds[ind,:]

ind = np.arange(0,len(epochs),4)
ax.plot(epochs[ind], returns[ind], color=color, linewidth=1.0)
fig.tight_layout()
fig.savefig(figname)
