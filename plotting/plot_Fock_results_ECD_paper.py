# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 11:05:29 2021

@author: qulab
"""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import h5py
import matplotlib.pyplot as plt

from rl_tools.tf_env import env_init
from rl_tools.tf_env import tf_env_wrappers as wrappers
import tensorflow as tf
import rl_tools.action_script as action_scripts
from math import pi
import numpy as np
import importlib
import matplotlib.pyplot as plt
from simulator.utils import expectation


focks = [1,2,3,4,5]
Ts = [1,2,3,4,5]
seeds = [0,1,2,3,4,5,6,7,8,9]
evaluations = 31

F = np.zeros([len(focks), len(Ts), len(seeds), evaluations])
epochs = np.zeros([len(focks), len(Ts), len(seeds), evaluations])

load_from_logs = False
load_from_npz = True

root_dir = r'E:\data\gkp_sims\PPO\ECD_paper'

if load_from_logs:
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
    filename = os.path.join(root_dir, r'data.npz')
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

#-----------------------------------------------------------------------------
# Find actions in the shortest protocols that achieve F>0.99


for idx_fock, fock in enumerate(focks):
    F_fock = F[idx_fock] # shape = [len(Ts), len(seeds), evaluations]
    best_F_fock = np.max(F_fock, axis=(1,2)) # shape = [len(Ts)]
    idx_T = np.where(best_F_fock>0.99)[0][0]
    min_T = np.array(Ts)[idx_T]
    print('Fock %d, min T = %d' %(fock, min_T))
    
    idx_seed, idx_evals = np.unravel_index(np.argmax(F_fock[idx_T]), F_fock[idx_T].shape)
    print('Fidelity %.4f' %F_fock[idx_T, idx_seed,idx_evals])
    
    this_dir = os.path.join(root_dir, 'T='+str(min_T), 'fock'+str(fock), 
                            'seed'+str(seeds[idx_seed]), 'policy')
    policy_name = os.listdir(this_dir)[idx_evals]
    policy_dir = os.path.join(this_dir, policy_name)
    
    policy = tf.compat.v2.saved_model.load(policy_dir)
    
    reward_kwargs = {'reward_mode' : 'zero'}
    
    # Params for environment
    env_kwargs = {
        'control_circuit' : 'snap_and_displacement',
        'init' : 'vac',
        'T' : min_T, 
        'N' : 100}
    
    # Params for action wrapper
    action_script = 'snap_and_displacements'
    action_scale = {'alpha':4, 'theta':pi}
    to_learn = {'alpha':True, 'theta':True}
    
    env = env_init(batch_size=1, **env_kwargs, episode_length=env_kwargs['T'])
    
    action_script_obj = importlib.import_module('rl_tools.action_script.' + action_script)
    env = wrappers.ActionWrapper(env, action_script_obj, action_scale, to_learn)
    
    
    action_names = list(to_learn.keys())
    all_actions = {a : [] for a in action_names}
    
    time_step = env.reset()
    policy_state = policy.get_initial_state(env.batch_size)
    max_alpha, max_n = 0, 0
    while not time_step.is_last():
        action_step = policy.action(time_step, policy_state)
        policy_state = action_step.state
        time_step = env.step(action_step.action)
        alpha = float(np.abs(expectation(env._state, env.a)))
        n = float(np.abs(expectation(env._state, env.n)))
        if alpha > max_alpha: max_alpha = alpha
        if n > max_n: max_n = n
    print('Max n %.3f' %max_n)
    print('Max alpha %.3f\n' %max_alpha)
        
    actions = {action_name : np.squeeze(np.array(action_history)[1:])
                for action_name, action_history in env.history.items()
                if not action_name=='msmt'}
    
    for a in action_names:
        all_actions[a].append(actions[a])
