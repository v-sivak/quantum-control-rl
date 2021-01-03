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
    'overlap' : r'E:\data\gkp_sims\PPO\examples\Wigner_reward_sweep\overlap',
    '(100, 1)' : r'E:\data\gkp_sims\PPO\examples\Wigner_reward_sweep\sample_avg1_point100',
    '(1, 1)' : r'E:\data\gkp_sims\PPO\examples\Wigner_reward_sweep\sample_avg1_point1',
    '(10, 1)' : r'E:\data\gkp_sims\PPO\examples\Wigner_reward_sweep\sample_avg1_point10',
    # '(1, 10)' : r'E:\data\gkp_sims\PPO\examples\Wigner_reward_sweep\sample_avg10_point1',
    # '(10, 10)' : r'E:\data\gkp_sims\PPO\examples\Wigner_reward_sweep\sample_avg10_point10'
    }

batch_size = 1000

samples_per_protocol = {
    'overlap' : 1,
    '(1, 1)' : 1,
    '(10, 1)' : 10,
    '(100, 1)' : 100,
    # '(1, 10)' : 10,
    # '(10, 10)' : 100
    }

colors = {
    'overlap' : 'black',
    '(1, 1)' : plt.get_cmap('tab20c')(19),
    '(10, 1)' : plt.get_cmap('tab20c')(17),
    '(100, 1)' : plt.get_cmap('tab20c')(16),
    # '(1, 10)' : plt.get_cmap('Paired')(6),
    # '(10, 10)' : plt.get_cmap('Paired')(7)
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
            
# figname = r'somewhere' # where to save
fig, ax = plt.subplots(1,1, figsize=(3.375, 2), dpi=300)
ax.set_ylabel(r'$1-\cal F$')
ax.set_xlabel('Epoch')
ax.set_ylim(3e-4,1.5)
ax.set_xlim(1e2-2e1,1e5+2e4)
ax.set_yscale('log')
ax.set_xscale('log')
plt.grid(True, color='grey', linestyle='--', linewidth=0.4)
palette = plt.get_cmap('tab10')

for i, protocol in enumerate(log.keys()):
    # in the background, plot all random seeds
    for sim_name in log[protocol].keys():
        epochs = log[protocol][sim_name]['epochs']
        ax.plot(epochs, 1-log[protocol][sim_name]['returns'],
                linestyle='--', color=colors[protocol], alpha=0.3)

    # calculate mean log infidelity
    all_seeds = np.array(list(log[protocol][i]['returns'] for i in log[protocol].keys()))
    log_infidelity = np.mean(np.log10(1-all_seeds), axis=0)
    infidelity = 10 ** log_infidelity
    
    ax.plot(epochs, infidelity, color=colors[protocol], linewidth=1.0, label=protocol)

fig.tight_layout()


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
    log_infidelity = np.mean(np.log10(1-all_seeds), axis=0)
    infidelity = 10 ** log_infidelity
    epochs = log[protocol][sim_name]['epochs']
    ax.plot(samples, infidelity, color=colors[protocol], linewidth=1.0, label=protocol)

# ax.legend(loc='best')

fig.tight_layout()
# fig.savefig(figname)