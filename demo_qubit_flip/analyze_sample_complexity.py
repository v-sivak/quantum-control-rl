# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 20:49:24 2021

@author: Vladimir Sivak
"""

# append parent 'gkp-rl' directory to path 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))


import numpy as np
import matplotlib.pyplot as plt
from plotting import plot_config
import os

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Create a figure showing 1) sample complexity required for fidelity
# estimation on a static protocol, and 2) sample complexity of RL training 
# required to achieve the same fidelity. It demonstrates that the agent can
# reach close to the shot-noise limit on fidelity estimation

folder_name = r'E:\data\gkp_sims\PPO\simple_demo\sweep17'
savename = r'E:\VladGoogleDrive\Qulab\Talks\21_05_21_Julich_control_club_Model-Free Qauntum Control with Reinforcement Learning\sample_complexity.pdf'

### Load trajectories data
all_trajectories = []
N_trajectories = len(os.listdir(folder_name))-1
for i in range(N_trajectories):
    log = np.load(os.path.join(folder_name, 'data'+str(i)+'.npz'), allow_pickle=True)
    sigma_z = -log['eval_rewards'].squeeze()
    fidelity = (1-sigma_z)/2
    all_trajectories.append(fidelity)
eval_epochs = log['eval_epochs']
batch = len(log['train_rewards'][0])

if 'train_samples' in log.keys():
    train_samples = np.array([0]+list(log['train_samples']))
else: train_samples = eval_epochs * batch


### Plot individual trajectories
fig, ax = plt.subplots(1,1, figsize=(3.375,2.3), dpi=200)

ymin, ymax = (5e-4,1.2)
ax.set_ylim(ymin, ymax)
# ax.set_xlim(0,60)
ax.set_yscale('log')
# ax.set_xscale('log')
ax.set_xlabel('Epoch')
ax.set_ylabel('Infidelity')
# for i in range(N_trajectories):
#     ax.plot(eval_epochs, 1-all_trajectories[i], alpha=0.25, linestyle='--')


### Plot 1) logmean fidelity 2) shot noise limit
median = np.median(all_trajectories, axis=0)

log10_infidelity = np.log10(1-np.array(all_trajectories))
log10_infidelity = np.ma.array(log10_infidelity, mask=np.where(log10_infidelity==-np.inf, True, False))
logmean = 1-10**(np.mean(log10_infidelity, axis=0))

ax.plot(eval_epochs, 1-logmean, color='k', linestyle='--', label='logmean')
ax.plot(eval_epochs, 1/(1+train_samples), color='r', label='shot noise bound')
# ax.plot(eval_epochs, 1-median, color='r', linestyle='--', label='median')
ax.legend(loc='upper right')
# ax.legend(loc='lower left')

msmt = lambda x: x*batch
ax2 = ax.secondary_xaxis('top', functions=(msmt, msmt))
ax2.set_xlabel('Sample size, M')
ax2.set_xticks([0,250,500,750,1000])
ax.set_xticks([0,25,50,75,100])


### Histogram the fidelities and make a colormap
hist_range = (-5, 0, 41)
all_hists = []
for i in eval_epochs:
    this_hist = np.histogram(log10_infidelity[:,i], bins=hist_range[-1], range=hist_range[:-1])[0]
    all_hists.append(this_hist)
all_hists = np.array(all_hists)

vmin, vmax = 0, 270
p = ax.pcolormesh(eval_epochs, 10**np.linspace(*hist_range), np.transpose(all_hists), 
                  cmap='YlGn', vmin=vmin, vmax=vmax)


from mpl_toolkits.axes_grid1 import make_axes_locatable

divider = make_axes_locatable(ax)
ax2 = divider.append_axes("right", size="15%", pad=0.05)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(p, cax=cax)

ax2.hist(log10_infidelity[:,-1], bins=hist_range[-1], orientation='horizontal')
ax2.set_ylim(np.log10(ymin),np.log10(ymax))
ax2.yaxis.set_ticks_position('none')
ax2.yaxis.set_ticklabels([])
ax2.xaxis.set_ticks_position('none')
ax2.xaxis.set_ticklabels([])

### Show the histogram of last epoch
cm = plt.cm.get_cmap('YlGn')
n, bins, patches = ax2.hist(log10_infidelity[:,-1], bins=hist_range[-1], orientation='horizontal')
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# scale values to interval [0,1]
col = (n-vmin)/(vmax-vmin)

for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c))

# plt.subplots_adjust(wspace=0.05)
plt.tight_layout()
fig.savefig(savename, dpi=600)
