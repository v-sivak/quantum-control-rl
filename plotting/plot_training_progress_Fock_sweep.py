# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 20:55:17 2020
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
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import qutip as qt
from gkp.gkp_tf_env import tf_env_wrappers as wrappers
from gkp.gkp_tf_env import gkp_init
import gkp.action_script as action_scripts
import plot_config

root_dir = r'E:\data\gkp_sims\PPO\paper_data\Fock_sweep_large_nn'
fock_states = [1,2,3,4,5,6,7,8,9,10]

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

# # Initialize the environment and simulation/training parameters
# env = gkp_init(simulate='snap_and_displacement_miscalibrated', 
#                 init='vac', H=1, T=5, attn_step=1, batch_size=1, N=100,
#                 episode_length=5)

# action_script = 'snap_and_displacements'
# action_scale = {'alpha':4, 'theta':pi}
# to_learn = {'alpha':True, 'theta':True}

# action_script = action_scripts.__getattribute__(action_script)
# env = wrappers.ActionWrapper(env, action_script, action_scale, to_learn)


# # Collect episodes for each policy
# rewards, final_states, epochs = {}, {}, {}

# for fock in fock_states:
#     state = 'fock' + str(fock)
#     print('='*5 + '  ' + state + '  ' + '='*5)
#     fock_dir = os.path.join(root_dir, state)
#     rewards[state], final_states[state], epochs[state] = {}, {}, {}
    
#     #setup overlap reward for this Fock state
#     reward_kwargs = {'reward_mode'  : 'overlap', 
#                       'target_state' : qt.tensor(qt.basis(2,0), qt.basis(env.N, fock))}
#     env.setup_reward(reward_kwargs)
    
#     # collect episodes with different policies
#     for sim_name in os.listdir(fock_dir):
#         print(sim_name)
#         sim_dir = os.path.join(fock_dir, sim_name)    
#         rewards[state][sim_name] = []
#         epochs[state][sim_name] = []
#         final_states[state][sim_name] = []
#         for policy_name in os.listdir(os.path.join(sim_dir, 'policy')):
#             policy_dir = os.path.join(sim_dir, 'policy', policy_name)
#             policy = tf.compat.v2.saved_model.load(policy_dir)
            
#             time_step = env.reset()
#             policy_state = policy.get_initial_state(env.batch_size)
#             counter = 0
#             while not time_step.is_last()[0]:
#                 counter += 1
#                 action_step = policy.action(time_step, policy_state)
#                 policy_state = action_step.state
#                 time_step = env.step(action_step.action)
            
#             rewards[state][sim_name].append(np.mean(time_step.reward))
#             epochs[state][sim_name].append(int(policy_name))
#             final_states[state][sim_name].append(env.info['psi_cached'])

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

# load data from the log
rewards, final_states, epochs = {}, {}, {}
for fock in fock_states:
    state = 'fock' + str(fock)
    print('='*5 + '  ' + state + '  ' + '='*5)
    fock_dir = os.path.join(root_dir, state)
    rewards[state], epochs[state] = {}, {}

    for sim_name in os.listdir(fock_dir):
        fname = os.path.join(fock_dir, sim_name, 'log.hdf5')
        h5file = h5py.File(fname,'r+')
        grpname = list(h5file.keys())[-1] # pick the latest log
        try:
            grp = h5file[grpname]
            epochs[state][sim_name] = np.array(grp.get('epochs'))
            rewards[state][sim_name] = np.array(grp.get('returns'))
        finally:
            h5file.close()

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

### save evaluation data
import pickle
with open(os.path.join(root_dir, 'eval.pickle'), 'wb') as f:
    pickle.dump(dict(rewards=rewards, final_states=final_states,epochs=epochs), f)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

### Plot training progress
figname = r'E:\VladGoogleDrive\Qulab\GKP\paper\figs\Fock_prep_sweep' # where to save
fig, ax = plt.subplots(1,1, figsize=(3.5, 2.7), dpi=300)
plt.grid(True)
ax.set_ylabel(r'$1-\cal F$')
ax.set_xlabel('Epoch')
ax.set_yscale('log')
# ax.set_xscale('log')
ax.set_xlim(-50,4050)
# ax.set_ylim(3e-4,1)
ax.set_ylim(3e-4,1.3)
palette = plt.get_cmap('tab10')

for fock in fock_states:
    state = 'fock' + str(fock)
    # plot training progress of each policy in the background
    for sim_name in rewards[state].keys():
        ax.plot(epochs[state][sim_name], 1-np.array(rewards[state][sim_name]), 
                linestyle='--', alpha=0.25, color=palette(fock-1))    
    all_seeds = np.array([rews for seed, rews in rewards[state].items()])

    # calculate mean log infidelity
    log_infidelity = np.mean(np.log10(1-all_seeds), axis=0)
    infidelity = 10 ** log_infidelity
    
    # calculate the best infidelity
    ind = np.argmax(all_seeds[:,-1])
    infidelity = 1 - all_seeds[ind,:]
    
    median_reward = np.median(all_seeds, axis=0)
    train_epochs = np.array(epochs[state]['seed0'])
    
    ind = [i for i in range(len(median_reward)) if i%1==0]
    ax.plot(train_epochs[ind], infidelity[ind], color=palette(fock-1), linewidth=1.0)
    ax.plot(train_epochs[ind], infidelity[ind], color=palette(fock-1), linewidth=1.0,
            label=fock, linestyle='none', marker='.')
ax.legend(ncol=3, prop={'size': 7})

fig.tight_layout()
fig.savefig(figname)