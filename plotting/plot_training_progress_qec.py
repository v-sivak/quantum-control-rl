# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 14:52:42 2020

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
import matplotlib as mpl
import matplotlib.pyplot as plt
from time import time
import tensorflow as tf
import qutip as qt
from rl_tools.tf_env import helper_functions as hf
from rl_tools.tf_env import tf_env_wrappers as wrappers
from rl_tools.tf_env import env_init
from rl_tools.tf_env import policy as plc
import rl_tools.action_script as action_scripts
from tensorflow.keras.backend import batch_dot
from simulator.utils import expectation
import h5py
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
### Setup matplotlib
figname = r'E:\VladGoogleDrive\Qulab\GKP\paper\figs\train' # where to save

fontsize = 9
fontsize_tick = 8
linewidth = 0.75
spinewidth = 1.0
markersize = linewidth*6
tick_size = 3.0
pad = 2

mpl.rcParams['savefig.dpi'] = 600
mpl.rcParams['savefig.format'] = 'pdf'
mpl.rcParams['savefig.transparent'] = True
#mpl.rcParams['figure.subplot.bottom'] = 0.2
#mpl.rcParams['figure.subplot.right'] = 0.85
#mpl.rcParams['figure.subplot.left'] = 0.18
mpl.rcParams['axes.linewidth'] = spinewidth
mpl.rcParams['axes.labelsize'] = fontsize
mpl.rcParams['axes.labelpad'] = pad

mpl.rcParams['xtick.major.size'] = tick_size
mpl.rcParams['xtick.major.width'] = spinewidth
mpl.rcParams['xtick.minor.size'] = tick_size / 1.5
mpl.rcParams['xtick.minor.width'] = spinewidth / 1.5

mpl.rcParams['ytick.major.size'] = tick_size
mpl.rcParams['ytick.major.width'] = spinewidth
mpl.rcParams['ytick.minor.size'] = tick_size / 1.5
mpl.rcParams['ytick.minor.width'] = spinewidth / 1.5

mpl.rcParams['xtick.major.pad']= pad
mpl.rcParams['ytick.major.pad']= pad
mpl.rcParams['xtick.minor.pad']= pad / 2.0
mpl.rcParams['ytick.minor.pad']= pad / 2.0

mpl.rcParams['xtick.labelsize'] = fontsize_tick
mpl.rcParams['ytick.labelsize'] = fontsize_tick

mpl.rcParams['legend.fontsize'] = fontsize_tick
mpl.rcParams['legend.frameon'] = True

mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['lines.markersize'] = markersize
mpl.rcParams['lines.markeredgewidth'] = linewidth / 2
            
mpl.rcParams['legend.markerscale'] = 2.0

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------


root_dir = {
    'ST' : r'E:\data\gkp_sims\PPO\examples\gkp_square_qec_ST_B100_lr3e-4',
    'BsB' : r'E:\data\gkp_sims\PPO\examples\gkp_square_qec_BsB_B100_lr3e-4'
    }

log = {'ST' : {}, 'BsB' : {}}

for protocol in ['ST', 'BsB']:
    
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


# Plot returns
figname = r'E:\VladGoogleDrive\Qulab\GKP\paper\figs\train_qec' # where to save
fig, ax = plt.subplots(1,1, figsize=(3.375, 2), dpi=300)
ax.set_ylabel('Return')
ax.set_xlabel('Epoch')
ax.set_ylim(-0.05,1.05)
plt.grid(True)
palette = plt.get_cmap('tab10')

for i, protocol in enumerate(log.keys()):
    for sim_name in log[protocol].keys():
        ax.plot(log[protocol][sim_name]['epochs'], log[protocol][sim_name]['returns'],
                linestyle='--', color=palette(i), alpha=0.4)


    epochs = log[protocol][sim_name]['epochs']
    # calculate and plot the meadian (less sensitive to outliers)
    avg_return = np.median(list(log[protocol][i]['returns'] for i in log[protocol].keys()), axis=0)
    ax.plot(epochs, avg_return, color=palette(i), linewidth=1.0, label=protocol)

ax.legend(loc='lower right')

fig.tight_layout()
fig.savefig(figname)





# def plot_paper_T1():

#     figname = r'E:\VladGoogleDrive\Qulab\GKP\paper\figs\T1.pdf'
#     fig, axes = plt.subplots(1,2, figsize=(3.6, 1.70), dpi=300, sharey=True, sharex=True)

#     x = np.linspace(0,1,100)
#     y0 = 0.8*np.exp(-10*x)
#     y1 = 0.8*np.exp(-3*x)
#     y2 = 0.8*np.exp(-1*x)

#     # Plot returns
#     fig, ax = plt.subplots(1,1, figsize=(3.375, 2), dpi=300)
#     ax.set_ylabel(r'$\langle X_L\rangle$, $\langle Y_L\rangle$, $\langle Z_L\rangle$')
#     ax.set_xlabel('Time (us)')
#     plt.grid(True)
#     # ax.set_xscale('log')
#     # ax.set_yscale('log')
#     ax.set_ylim(0,1)
#     palette = plt.get_cmap('tab10')
#     ax.plot(x,y0)
#     ax.plot(x,y1)
#     ax.plot(x,y2)
#     fig.tight_layout()
#     fig.savefig(figname, dpi=300)