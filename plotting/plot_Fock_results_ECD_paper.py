# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 11:05:29 2021

@author: qulab
"""
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

focks = [1,2,3,4,5]
Ts = [1,2,3,4,5]
seeds = [0,1,2,3,4,5,6,7,8,9]
evaluations = 31

F = np.zeros([len(focks), len(Ts), len(seeds), evaluations])
epochs = np.zeros([len(focks), len(Ts), len(seeds), evaluations])

load_from_logs = False
load_from_npz = True

if load_from_logs:
    root_dir = r'E:\data\gkp_sims\PPO\ECD_paper'
    # load data from the log
    for idx_T, T in enumerate(Ts):
        T_dir = os.path.join(root_dir, 'T='+str(T))
        for idx_fock, fock in enumerate(focks):
            fock_dir = os.path.join(T_dir, 'fock'+str(fock))
            for idx_seed, seed in enumerate(seeds):
                seed_dir = os.path.join(fock_dir, 'seed'+str(seed))
    
                fname = os.path.join(seed_dir, 'log.hdf5')
                h5file = h5py.File(fname,'r+')
                grpname = list(h5file.keys())[-1] # pick the latest log
                try:
                    grp = h5file[grpname]
                    F[idx_fock, idx_T, idx_seed] = np.array(grp.get('returns'))
                    epochs[idx_fock, idx_T, idx_seed] = np.array(grp.get('epochs'))
                finally:
                    h5file.close()
    np.savez(os.path.join(root_dir, 'data.npz'), F=F, epochs=epochs)

if load_from_npz:
    filename = r'E:\data\gkp_sims\PPO\ECD_paper\data.npz'
    data = np.load(filename)
    F, epochs = data['F'], data['epochs']

#-----------------------------------------------------------------------------
best_F = np.max(F, axis=(2,3)) # shape = [len(focks), len(Ts)]

fig, ax = plt.subplots(1,1, dpi=600)
ax.set_xlabel('Circuit depth')
ax.set_ylabel('Fock state')
ax.imshow(np.log(1-best_F))
plt.xticks(np.arange(len(Ts)), Ts)
plt.yticks(np.arange(len(focks)), focks)
plt.tight_layout()
