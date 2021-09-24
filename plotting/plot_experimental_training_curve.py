# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 13:14:42 2021

@author: qulab
"""
import pandas as pd
import tensorboard as tb
import matplotlib.pyplot as plt
import h5py
import numpy as np
import plot_config

# LOAD TENSORBOARD DATA (RETURNS OF STOCHASTIC POLICY)
experiment_id = "5OQmXGmjSkOYEOU5Geg76A"
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
df = experiment.get_scalars()
print(df.tag.unique())

# this is just returns
returns = df[df.tag.str.endswith('value_actual_avg')].to_dict('list') 

# predictions of the value network on the observations
value_preds = df[df.tag.str.endswith('value_pred_avg')].to_dict('list')


# LOAD LOG SAVED TO HDF5 (DETERMINISTIC POLICY)
fname = r'E:\data\gkp_sims\PPO\ECD\EXP_Vlad\sbs_pauli_hex\run_2\log.hdf5'
grpname = '002000'

h5file = h5py.File(fname,'r+')
try:
    log = {}
    grp = h5file[grpname]
    for name in grp.keys():
        log[name] = np.array(grp.get(name))
finally:
    h5file.close()


# WHICH POLICY IS USED EVENTUALLY
policy_epoch = 1460

fig, axes = plt.subplots(1, 2, figsize = (3.375, 2), dpi=200, sharey=True)
ax = axes[0]
ax.set_ylabel('Average stabilizer')
ax.set_xlabel('Epoch')
ax.grid(True)
palette = plt.get_cmap('Set1')

ind = np.arange(0, len(returns['step']), 4) # to avoid plotting every epoch

ax.plot(np.array(returns['step'])[ind]/20, np.array(returns['value'])[ind], label='stochastic', color=palette(2))
ax.plot(log['epochs'], log['returns'], label='deterministic', color=palette(4))

ind = np.where(log['epochs'] == policy_epoch)
ax.scatter(log['epochs'][ind], log['returns'][ind], zorder=3, marker='*', color=palette(0))

ax.legend()

ax = axes[1]
ax.set_ylim(0, 1)
ax.set_ylabel(r'Success probability, $P(m_1=1)$')

savename = r'E:\VladGoogleDrive\Qulab\GKP\experiment\figures\training_progress_state_prep.pdf' 
# fig.savefig(savename)