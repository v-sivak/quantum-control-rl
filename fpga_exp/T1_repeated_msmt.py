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

savepath = r'D:\DATA\exp\2020-11-18_cooldown\qubit_T1_T2_3'

experiments = [
        'gkp_exp.fpga_exp.T1',
        'gkp_exp.fpga_exp.T2_software_det'
        ]

tau = {exp : [] for exp in experiments}
time_start = {exp : [] for exp in experiments}
time_stop = {exp : [] for exp in experiments}

reps = 500
for i in range(reps):
    for exp in experiments:
        e = get_experiment(exp)
        time_start[exp].append(time())
        e.run()
        wait_complete(e)
        r = get_last_results(e)
        time_stop[exp].append(time())
        tau[exp].append(e.fit_params['tau'])
        duration = time_stop[exp][-1] - time_start[exp][-1]
        print('%d/%d Time: %.1f s' %(i, reps, duration))

# save
time_start = {k:np.array(v) for (k,v) in time_start.items()}
time_stop = {k:np.array(v) for (k,v) in time_stop.items()}
tau = {k:np.array(v) for (k,v) in tau.items()}

for exp in experiments:
    d = dict(tau=tau[exp], time_start=time_start[exp], time_stop=time_stop[exp])
    with open(os.path.join(savepath, exp + '.pickle'), 'w') as f:
        pickle.dump(d, f)

# Process and plot
mid_time = {exp : (time_start[exp] + time_stop[exp])/2.0 for exp in experiments}
mid_time = {exp : mid_time[exp] - time_start[exp][0] for exp in experiments}
mid_time = {exp : mid_time[exp]/60.0/60.0 for exp in experiments} # hrs

for exp in experiments:
    fig, ax = plt.subplots(1,1)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('tau (us)')
    ax.set_title(exp)
    ax.plot(mid_time[exp], tau[exp]*1e-3, linestyle='none', marker='o')
    plt.tight_layout()
    
    fig, ax = plt.subplots(1,1)
    ax.set_ylabel('Counts')
    ax.set_xlabel('tau (us)')
    ax.set_title(exp)
    ax.hist(tau[exp]*1e-3, bins=50)

#with open(os.path.join(savepath,'T2.pickle'), 'r') as f:
#    loaded = pickle.load(f)