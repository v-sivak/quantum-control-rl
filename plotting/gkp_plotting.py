# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 13:36:40 2020

@author: qulab
"""

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


def plot_supervised_training_log(logslist, xscale=None, yscale=None, metric='loss',
                                 ylabel=None, xlabel=None, labels=None):
    """
    Input:
        logslist -- list of log files produced by CSVLogger callback
        xscale -- optional list of scaling factors for x-axis. Use to convert 
                  epoch number to wall time, etc
        yscale -- optional list of scaling factors for y-axis. Use to convert
                  to loss per time step if return_sequences=True, etc 
        metric -- a thing to plot. Can be 'lr', 'loss', 'mse' etc
        
    """

    fig, ax = plt.subplots(1,1)
    ax.set_ylabel(ylabel if ylabel else metric)
    ax.set_xlabel(xlabel if xlabel else 'Epoch')
    ax.set_yscale('log')
    ax.set_xscale('log')    
    for j, filename in enumerate(logslist):
        log = {'Epoch' : []}
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    items = row
                    for p in items: log[p] = []
                else:
                    log['Epoch'].append(line_count)
                    for i, p in enumerate(items): log[p].append(float(row[i]))
                line_count += 1
        print(len(log['Epoch']))
        if metric in log.keys():
            data = np.array(log[metric])*yscale[j] if yscale else log[metric]
            time = np.array(log['Epoch'])*xscale[j] if xscale else log['Epoch']
            label = labels[j] if labels else None
            ax.plot(time, data, label=label)
    ax.legend()
    

def plot_feedback_amplitude_sweep():
    path = r'E:\VladGoogleDrive\Qulab\GKP\sims\Benchmarking_HybridMarkovian4Rounds\Sweep_of_feedback_displacement'

    amplitudes = np.load(os.path.join(path,'amplitudes.npy'))
    lifetimes = np.load(os.path.join(path,'lifetimes.npy'))
    mean_rewards = np.load(os.path.join(path,'mean_rewards.npy'))
    
    fig, ax = plt.subplots(2,1)
    ax[0].set_ylabel('T1 (us)')
    ax[0].plot(amplitudes, lifetimes)
    ax[1].set_ylabel('Mean reward')    
    ax[1].plot(amplitudes, mean_rewards)
    ax[1].set_xlabel('Feedback displacement amplitude')
    # fig.savefig(os.path.join(savepath,'summary.png' %a))
    


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
            ax.plot(log['epochs'][1:], log['returns'][1:], color=palette(i))
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
        
        
        
def plot_tensorflow_benchmarking(fname, groupname):

    dic = {}
    h5file = h5py.File(fname,'r+')
    try:
        grp = h5file[groupname]
        for name in grp.keys():
            dic[name] = np.array(grp.get(name))
    finally:
        h5file.close()

    # Clean up the dictionary
    Hilbert_space_size = dic['Hilbert_space_size']
    del(dic['displacements'])
    del(dic['Hilbert_space_size'])

    # Plot stuff
    fig, ax = plt.subplots(1,1) # figsize=(3.375, 2.0)
    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Hilbert space size')
    ax.set_yscale('log')
    ax.set_xticks(Hilbert_space_size)
    plt.grid(True, which="both")
    for key, val in dic.items():
        ax.plot(Hilbert_space_size, val, label=key)
    ax.legend()
    
    fig.savefig(os.path.join(os.path.dirname(fname), 'plot.png'),
                figsize=(3.375, 2.0))
        


def plot_wigner_all_states():
    
    stabilizers, paulis, states, displacements = \
        hf.GKP_state(False,100, np.array([[1,0],[0,1]]))
    
    fig, axes = plt.subplots(2,3, sharex=True, sharey=True, figsize=(6,6))
    for i, s1 in zip([0,1],['+','-']):
        for j, s2 in zip([0,1,2],['X','Y','Z']):            
            state = states[s2+s1]
            xvec = np.linspace(-7,7,201)
            W = qt.wigner(state, xvec, xvec, g=sqrt(2))            
            
            ax = axes[i,j]
            ax.grid(linestyle='--',zorder=2)
            lim = 3.5
            ax.set_xlim(-lim,lim)
            ax.set_ylim(-lim,lim)
            ticks = [-2,-1,0,1,2]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            # ax.set_title(s2+s1)
            ax.set_aspect('equal')
            p = ax.pcolormesh(xvec/sqrt(pi), xvec/sqrt(pi), W, 
                              cmap='RdBu_r', vmin=-1/pi, vmax=+1/pi)
    # cbar = fig.colorbar(p, ax=ax, ticks=[-1/pi,0,1/pi])
    # cbar.ax.set_yticklabels([r'$-\frac{1}{\pi}$','0',r'$+\frac{1}{\pi}$'])
    # axes[1,1].set_xlabel(r'$q/\sqrt{\pi}$')
    # axes[1,0].set_ylabel(r'$p/\sqrt{\pi}$')
    plt.tight_layout()
    # plt.savefig(r'E:\VladGoogleDrive\Qulab\GKP\Notes\wigners2.pdf')
