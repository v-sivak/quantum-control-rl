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

# naming convention is (num points, num averages)


root_dir = {
    'overlap' : r'E:\data\gkp_sims\PPO\paper_data\Wigner_bin1_small_v2_nn_eps0.2\overlap',
    '100' : r'E:\data\gkp_sims\PPO\paper_data\Wigner_bin1_small_v2_nn_eps0.2\sample100_v2',
    '1' : r'E:\data\gkp_sims\PPO\paper_data\Wigner_bin1_small_v2_nn_eps0.2\sample1',
    '10' : r'E:\data\gkp_sims\PPO\paper_data\Wigner_bin1_small_v2_nn_eps0.2\sample10',
    }


# root_dir = {
#     'overlap' : r'E:\data\gkp_sims\PPO\paper_data\Wigner_cat2_small_nn\overlap',
#     '100' : r'E:\data\gkp_sims\PPO\paper_data\Wigner_cat2_small_nn\sample100',
#     '1' : r'E:\data\gkp_sims\PPO\paper_data\Wigner_cat2_small_nn\sample1',
#     '10' : r'E:\data\gkp_sims\PPO\paper_data\Wigner_cat2_small_nn\sample10',
#     }

batch_size = 1000

samples_per_protocol = {
    'overlap' : 1,
    '1' : 1,
    '10' : 10,
    '100' : 100,
    }

cmap1 = plt.get_cmap('Dark2')
cmap2 = plt.get_cmap('tab10')

colors = {
    'overlap' : cmap2(0),
    '1' : cmap1(0), #plt.get_cmap('tab20c')(19),
    '10' : cmap1(5), #plt.get_cmap('tab20c')(17),
    '100' : cmap2(3) #plt.get_cmap('tab20c')(16),
    }

log = {k:{} for k in root_dir.keys()}

for protocol in root_dir.keys():
    
    for sim_name in os.listdir(root_dir[protocol]):
        fname = os.path.join(root_dir[protocol], sim_name, 'log.hdf5')
        h5file = h5py.File(fname,'r+')
        grpname = list(h5file.keys())[-1] # pick the latest log
        log[protocol][sim_name] = {}
        try:
            grp = h5file[grpname]
            log[protocol][sim_name]['epochs'] = np.array(grp.get('epochs'))
            log[protocol][sim_name]['returns'] = np.array(grp.get('returns'))
        finally:
            h5file.close()

# Plot infidelity vs epoch number
            
figname = r'E:\VladGoogleDrive\Qulab\GKP\paper\figs\Wigner_sweep_bin.pdf' # where to save
fig, ax = plt.subplots(1,1, figsize=(3.375, 2.1), dpi=300)
ax.set_ylabel(r'$1-\cal F$')
ax.set_xlabel('Epoch')
ax.set_ylim(8e-4,1.5)
# ax.set_xlim(1e2-2e1,1e5+2e4)
ax.set_xlim(8e1,2e4+0.25e4)
ax.set_yscale('log')
ax.set_xscale('log')
plt.grid(True)
palette = plt.get_cmap('tab10')

for i, protocol in enumerate(log.keys()):
    
    # in the background, plot all random seeds
    for sim_name in log[protocol].keys():
        epochs = log[protocol][sim_name]['epochs']

        if protocol != 'overlap':
            ind = np.arange(len(epochs)).astype(int)
        else:
            ind1 = np.arange(len(epochs)/2-1).astype(int)
            ind2 = np.arange(len(epochs)/2-1, len(epochs), 3).astype(int)
            ind = np.concatenate([ind1,ind2])

        ax.plot(epochs[ind], 1-log[protocol][sim_name]['returns'][ind],
                linestyle='--', color=colors[protocol], alpha=0.3)

    # calculate mean log infidelity
    all_seeds = np.array(list(log[protocol][i]['returns'] for i in log[protocol].keys()))
    # log_infidelity = np.mean(np.log10(1-all_seeds), axis=0)
    # infidelity = 10 ** log_infidelity

    # calculate the best infidelity
    inds = np.argmax(all_seeds[:,-1])
    infidelity = 1 - all_seeds[inds,:]
    

    
    ax.plot(epochs[ind], infidelity[ind], color=colors[protocol], linewidth=1.0)
    # ax.plot(epochs, infidelity, color=colors[protocol], linewidth=1.0,
    #         label=protocol, linestyle='none', marker='.')
# ax.legend(loc='best', prop={'size': 7})
fig.tight_layout()
fig.savefig(figname)

# Plot infidelity vs number of training episodes

fig, ax = plt.subplots(1,1, figsize=(3.375, 2), dpi=300)
ax.set_ylabel(r'$1-\cal F$')
ax.set_xlabel('Trajectory')
ax.set_yscale('log')
ax.set_xscale('log')
plt.grid(True, color='grey', linestyle='--', linewidth=0.4)
palette = plt.get_cmap('tab10')

for i, protocol in enumerate(log.keys()-['overlap']):
    # in the background, plot all random seeds
    for sim_name in log[protocol].keys():
        samples = log[protocol][sim_name]['epochs'] * samples_per_protocol[protocol] *batch_size
        ax.plot(samples, 1-log[protocol][sim_name]['returns'],
                linestyle='--', color=colors[protocol], alpha=0.3)

    # calculate mean log infidelity
    all_seeds = np.array(list(log[protocol][i]['returns'] for i in log[protocol].keys()))
    # log_infidelity = np.mean(np.log10(1-all_seeds), axis=0)
    # infidelity = 10 ** log_infidelity
    
    # calculate the best infidelity
    ind = np.argmax(all_seeds[:,-1])
    infidelity = 1 - all_seeds[ind,:]
    
    epochs = log[protocol][sim_name]['epochs']
    ax.plot(samples, infidelity, color=colors[protocol], linewidth=1.0, label=protocol)

ax.legend(loc='best')

fig.tight_layout()
