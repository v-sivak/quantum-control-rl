# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 13:36:40 2020

@author: qulab
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py


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
    fig, ax = plt.subplots(1,1)
    ax.set_ylabel('Return')
    ax.set_xlabel('Epoch')
    plt.grid(True)
    # ax.set_xscale('log')
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