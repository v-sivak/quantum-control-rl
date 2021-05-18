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
from rl_tools.tf_env import tf_env_wrappers as wrappers
from rl_tools.tf_env import env_init
import rl_tools.action_script as action_scripts
import plot_config
import importlib

root_dir = r'E:\data\gkp_sims\PPO\examples\gkp_T9'
deltas = [0.45, 0.35, 0.25, 0.15]


#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

# load data from the log
rewards, final_states, epochs = {}, {}, {}
for Delta in deltas:
    state = 'delta' + str(Delta)
    print('='*5 + '  ' + state + '  ' + '='*5)
    delta_dir = os.path.join(root_dir, state)
    rewards[state], epochs[state] = {}, {}

    for sim_name in os.listdir(delta_dir):
        fname = os.path.join(delta_dir, sim_name, 'log.hdf5')
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
    pickle.dump(dict(rewards=rewards, final_states=final_states, epochs=epochs), f)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

### Plot training progress
figname = r'E:\VladGoogleDrive\Qulab\GKP\paper\figs\GKP_finite_energy_prep_sweep' # where to save
fig, ax = plt.subplots(1,1, figsize=(3.375, 2.4), dpi=300)
plt.grid(True)
ax.set_ylabel(r'$(\langle S_{x,\Delta}\rangle + \langle S_{p,\Delta}\rangle)/2$')
ax.set_xlabel('Epoch')
ax.set_xlim(-400,10400)
ax.set_ylim(-0.2,1.05)
palette = plt.get_cmap('tab10')

cmap1 = plt.get_cmap('Dark2')
cmap2 = plt.get_cmap('tab10')
colors = [cmap2(0), cmap1(0), cmap1(5), cmap2(3)]
palette = lambda j: colors[j]
# palette = plt.get_cmap('tab10')

best_seed = {}
for j, Delta in enumerate(deltas):
    state = 'delta' + str(Delta)
    # plot training progress of each policy in the background
    for sim_name in rewards[state].keys():
        ax.plot(epochs[state][sim_name], np.array(rewards[state][sim_name]), 
                linestyle='--', alpha=0.25, color=palette(j))
    all_seeds = np.array([rews for seed, rews in rewards[state].items()])
    seed_names = list(rewards[state].keys())
    
    # find the best random seed
    i_best= np.argmax(all_seeds[:,-1])
    stabilizer = all_seeds[i_best,:]
    best_seed[Delta] = seed_names[i_best]
    
    train_epochs = np.array(epochs[state]['seed0'])
    ind = [i for i in range(len(train_epochs)) if i%1==0]
    ax.plot(train_epochs[ind], stabilizer[ind], color=palette(j), linewidth=1.0)
    ax.plot(train_epochs[ind], stabilizer[ind], color=palette(j), linewidth=1.0,
            label=Delta, linestyle='none', marker='.')
ax.legend(ncol=1, prop={'size': 6}, loc='upper left')




avg_ideal_stabilizer, delta_effective = {}, {}

for Delta in deltas:
    
    # from rl_tools.tf_env import helper_functions as hf
    # target_state = hf.GKP_1D_state(False, 200, Delta*sqrt(2))
    # reward_kwargs = {'reward_mode' : 'overlap',
    #                   'target_state' : target_state,
    #                   'postselect_0' : False}
    
    reward_kwargs = {'reward_mode' : 'stabilizers_v2',
                      'Delta' : 0.0, 'beta' : sqrt(pi),
                      'sample' : False}
    
    env = env_init(control_circuit='snap_and_displacement', reward_kwargs=reward_kwargs,
                   init='vac', T=9, batch_size=1, N=200, episode_length=9)
    
    action_script = 'snap_and_displacements'
    action_scale = {'alpha':6, 'theta':pi}
    to_learn = {'alpha':True, 'theta':True}
    
    module_name = 'rl_tools.action_script.' + action_script
    action_script = importlib.import_module(module_name)
    env = wrappers.ActionWrapper(env, action_script, action_scale, to_learn)

    delta_dir = os.path.join(root_dir, 'delta' + str(Delta))
    seed_dir = os.path.join(delta_dir, best_seed[Delta])
    policy_dir = r'policy\010000'
    policy = tf.compat.v2.saved_model.load(os.path.join(seed_dir,policy_dir))
    

    time_step = env.reset()
    policy_state = policy.get_initial_state(env.batch_size)
    while not time_step.is_last():
        action_step = policy.action(time_step, policy_state)
        policy_state = action_step.state
        time_step = env.step(action_step.action)
    env.render()
    
    S = time_step.reward.numpy()
    avg_ideal_stabilizer[Delta] = S
    delta_effective[Delta] = sqrt(1/pi*np.log(1/S**2))

Delta_dB = 10*np.log10(1/np.array(deltas)**2)
Delta_eff_dB = 10*np.log10(1/np.array([v for k,v in delta_effective.items()])**2)


rect = [0.58, 0.34, 0.38, 0.38] # left, bottom, width, height
ax1 = fig.add_axes(rect)
ax1.set_aspect('equal')
ax1.set_xlim(0,20)
ax1.set_ylim(0,20)
ax1.plot([0,20], [0,20], color='grey')
for i in range(len(deltas)):
    ax1.scatter(Delta_dB[i], Delta_eff_dB[i], color=palette(i), zorder=3)
ax1.set_xlabel(r'Target $\Delta$ (dB)', fontsize=7)
ax1.set_ylabel(r'Achieved $\Delta_{\rm eff}$ (dB)', fontsize=7)
ax1.set_xticks([0,10,20])
ax1.set_yticks([0,10,20])
ax1.xaxis.set_tick_params(labelsize=7)
ax1.yaxis.set_tick_params(labelsize=7)

fig.tight_layout()
# fig.savefig(figname)