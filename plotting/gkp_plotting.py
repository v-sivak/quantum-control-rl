import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import h5py
import qutip as qt
import tensorflow as tf
from numpy import sqrt,pi

#from rl_tools.tf_env import helper_functions as hf

fontsize = 9 #6 #1.5
fontsize_tick = 8 #6 #4 #15
linewidth = 1.5 #0.5 #0.25
spinewidth = 1.0 #0.1
markersize = linewidth*6
tick_size = 3.0 #0.5 #3
pad = 2 #0.05 #2

mpl.rcParams['savefig.dpi'] = 600
mpl.rcParams['savefig.format'] = 'pdf'
mpl.rcParams['savefig.transparent'] = True
#mpl.rcParams['figure.subplot.bottom'] = 0.2
#mpl.rcParams['figure.subplot.right'] = 0.85
#mpl.rcParams['figure.subplot.left'] = 0.18
mpl.rcParams['axes.linewidth'] = spinewidth #1.0 #2.0
mpl.rcParams['axes.labelsize'] = fontsize
mpl.rcParams['axes.labelpad'] = pad #4

mpl.rcParams['xtick.major.size'] = tick_size
mpl.rcParams['xtick.major.width'] = spinewidth#2.0
mpl.rcParams['xtick.minor.size'] = tick_size / 1.5
mpl.rcParams['xtick.minor.width'] = spinewidth / 1.5

mpl.rcParams['ytick.major.size'] = tick_size
mpl.rcParams['ytick.major.width'] = spinewidth #2.0
mpl.rcParams['ytick.minor.size'] = tick_size / 1.5 #3.0
mpl.rcParams['ytick.minor.width'] = spinewidth / 1.5

mpl.rcParams['xtick.major.pad']= pad #4
mpl.rcParams['ytick.major.pad']= pad #4
mpl.rcParams['xtick.minor.pad']= pad / 2.0 #4
mpl.rcParams['ytick.minor.pad']= pad / 2.0 #4

mpl.rcParams['xtick.labelsize'] = fontsize_tick
mpl.rcParams['ytick.labelsize'] = fontsize_tick

mpl.rcParams['legend.fontsize'] = fontsize_tick
mpl.rcParams['legend.frameon'] = True

mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['lines.markersize'] = markersize
mpl.rcParams['lines.markeredgewidth'] = linewidth / 2

mpl.rcParams['legend.markerscale'] = 2.0


def plot_rl_learning_progress(logs_to_plot, baseline=None):
    """
    Plot learning progres and training time results from the log.

    Inputs:
        logs_to_plot -- dictionary whose keys are filenames and values are
                        lists of groupnames from hdf5 log files.

    """

    all_logs= {}
    for fname in logs_to_plot.keys():
        # Load all requested training logs for this fname
        all_logs[fname] = []
        h5file = h5py.File(fname,'r+')
        try:
            for grpname in logs_to_plot[fname]:
                log = {}
                grp = h5file[grpname]
                for name in grp.keys():
                    log[name] = np.array(grp.get(name))
                all_logs[fname].append(log)
        finally:
            h5file.close()


    # Plot returns
    fig, ax = plt.subplots(1,1,dpi=200)
    ax.set_ylabel('Return')
    ax.set_xlabel('Epoch')
    plt.grid(True)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    palette = plt.get_cmap('tab10')
    max_epoch = 0
    for i, fname in enumerate(all_logs.keys()):
        for log in all_logs[fname]:
            ax.plot(log['epochs'], log['returns'], color=palette(i))
            max_epoch = max(max_epoch, np.max(log['epochs']))
    if baseline:
        ax.plot([0,max_epoch], [baseline,baseline], color='k')

    # # Plot experience and training time
    # fig, ax = plt.subplots(1,1)
    # ax.set_ylabel('Time (hrs)')
    # ax.set_xlabel('Epoch')
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    # for i, fname in enumerate(all_logs.keys()):
    #     for log in all_logs[fname]:
    #         ax.plot(log['epochs'], log['experience_time']/3600,
    #                 label='Experience: %.1f hrs' %(log['experience_time'][-1]/3600))
    #         ax.plot(log['epochs'], log['train_time']/3600,
    #                 label='Training: %.1f mins' %(log['train_time'][-1]/60))
    # ax.legend()

