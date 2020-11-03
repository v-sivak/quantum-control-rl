# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 17:32:33 2020

@author: qulab
"""

import numpy as np
from fpga_lib.scripting import get_experiment, wait_complete, get_last_results
from init_script import *
import matplotlib.pyplot as plt
from time import time
import pickle
import os

savepath = r'C:\Users\qulab\Desktop\qubit_T1_T2_interleaved_v2'

experiments = [get_experiment('fpga_std.characterize.T1'),
               get_experiment('fpga_std.characterize.T2'),
               get_experiment('exp_code.from_nick_fpga.T2_software_det')
               ]



tau = [[],[],[]]
data = [[],[],[]]
time_start = [[],[],[]]
time_stop = [[],[],[]]

reps = 400
for i in range(reps):
    for j, e in enumerate(experiments):
        time_start[j].append(time())
        e.run()
        wait_complete(e)
        r = get_last_results(e)
        time_stop[j].append(time())
        data[j].append(r['default'].data)
        tau[j].append(e.fit_params['tau'])
        print('%d/%d Time: %.1f s' %(i,reps,time_stop[j][-1]-time_start[j][-1]))

for j, e in enumerate(experiments):
    d = dict(tau=tau[j], data = data[j], time_start = time_start[j], time_stop = time_stop[j])
    with open(os.path.join(savepath, e.name + '.pickle'), 'w') as f:
        pickle.dump(d, f)


# Process and plot
mid_time = [(np.array(time_start[i]) + np.array(time_stop[i]))/2.0 for i in range(len(experiments))]
mid_time = [mid_time[i] - time_start[i][0] for i in range(len(experiments))]
mid_time = [mid_time[i]/60.0/60.0 for i in range(len(experiments))] # hrs
tau = [np.array(tau[i])*1e-3 for i in range(len(experiments))] # us

for j, e in enumerate(experiments):
    fig, ax = plt.subplots(1,1)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('tau (us)')
    ax.set_title(e.name)
    ax.plot(mid_time[j], tau[j], linestyle='none', marker='o')
    plt.tight_layout()
    
    fig, ax = plt.subplots(1,1)
    ax.set_ylabel('Counts')
    ax.set_xlabel('tau (us)')
    ax.set_title(e.name)
    ax.hist(tau, bins=30)

#with open(os.path.join(savepath,'T2.pickle'), 'r') as f:
#    loaded = pickle.load(f)