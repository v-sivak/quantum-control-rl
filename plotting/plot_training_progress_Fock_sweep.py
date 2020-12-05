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
import matplotlib as mpl
import matplotlib.pyplot as plt
from time import time
import tensorflow as tf
import qutip as qt
from gkp.gkp_tf_env import helper_functions as hf
from gkp.gkp_tf_env import tf_env_wrappers as wrappers
from gkp.gkp_tf_env import gkp_init
from gkp.gkp_tf_env import policy as plc
import gkp.action_script as action_scripts
from tensorflow.keras.backend import batch_dot
from simulator.utils import expectation
# from simulator.operators import projector
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
### Setup matplotlib
fontsize = 9
fontsize_tick = 8
linewidth = 0.75
spinewidth = 1.0
markersize = linewidth*6
tick_size = 3.0
pad = 2

mpl.rcParams['savefig.dpi'] = 600
mpl.rcParams['savefig.format'] = 'pdf'
mpl.rcParams['savefig.transparent'] = True
#mpl.rcParams['figure.subplot.bottom'] = 0.2
#mpl.rcParams['figure.subplot.right'] = 0.85
#mpl.rcParams['figure.subplot.left'] = 0.18
mpl.rcParams['axes.linewidth'] = spinewidth
mpl.rcParams['axes.labelsize'] = fontsize
mpl.rcParams['axes.labelpad'] = pad

mpl.rcParams['xtick.major.size'] = tick_size
mpl.rcParams['xtick.major.width'] = spinewidth
mpl.rcParams['xtick.minor.size'] = tick_size / 1.5
mpl.rcParams['xtick.minor.width'] = spinewidth / 1.5

mpl.rcParams['ytick.major.size'] = tick_size
mpl.rcParams['ytick.major.width'] = spinewidth
mpl.rcParams['ytick.minor.size'] = tick_size / 1.5
mpl.rcParams['ytick.minor.width'] = spinewidth / 1.5

mpl.rcParams['xtick.major.pad']= pad
mpl.rcParams['ytick.major.pad']= pad
mpl.rcParams['xtick.minor.pad']= pad / 2.0
mpl.rcParams['ytick.minor.pad']= pad / 2.0

mpl.rcParams['xtick.labelsize'] = fontsize_tick
mpl.rcParams['ytick.labelsize'] = fontsize_tick

mpl.rcParams['legend.fontsize'] = fontsize_tick
mpl.rcParams['legend.frameon'] = True

mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['lines.markersize'] = markersize
mpl.rcParams['lines.markeredgewidth'] = linewidth / 2
            
mpl.rcParams['legend.markerscale'] = 2.0

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
### Initialize the environment and simulation/training parameters
env = gkp_init(simulate='snap_and_displacement', 
               init='vac', H=1, T=5, attn_step=1, batch_size=1, N=100,
               episode_length=5, phase_space_rep='wigner')

action_script = 'snap_and_displacements'
action_scale = {'alpha':4, 'theta':pi}
to_learn = {'alpha':True, 'theta':True}

action_script = action_scripts.__getattribute__(action_script)
env = wrappers.ActionWrapper(env, action_script, action_scale, to_learn)


root_dir = r'E:\data\gkp_sims\PPO\examples\Fock_states_sweep'
fock_states = [1,2,3,4,5,6,7,8,9,10]


#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
### Collect episodes for each policy

rewards, final_states, epochs = {}, {}, {}

for fock in fock_states:
    state = 'fock' + str(fock)
    print('='*5 + '  ' + state + '  ' + '='*5)
    fock_dir = os.path.join(root_dir, state)
    rewards[state], final_states[state], epochs[state] = {}, {}, {}
    
    #setup overlap reward for this Fock state
    reward_kwargs = {'reward_mode'  : 'overlap', 
                     'target_state' : qt.basis(env.N, fock)}
    env.setup_reward(reward_kwargs)
    
    # collect episodes with different policies
    for sim_name in os.listdir(fock_dir):
        print(sim_name)
        sim_dir = os.path.join(fock_dir, sim_name)    
        rewards[state][sim_name] = []
        epochs[state][sim_name] = []
        final_states[state][sim_name] = []
        for policy_name in os.listdir(os.path.join(sim_dir, 'policy')):
            policy_dir = os.path.join(sim_dir, 'policy', policy_name)
            policy = tf.compat.v2.saved_model.load(policy_dir)
            
            time_step = env.reset()
            policy_state = policy.get_initial_state(env.batch_size)
            counter = 0
            while not time_step.is_last()[0]:
                counter += 1
                action_step = policy.action(time_step, policy_state)
                policy_state = action_step.state
                time_step = env.step(action_step.action)
            
            rewards[state][sim_name].append(np.mean(time_step.reward))
            epochs[state][sim_name].append(int(policy_name))
            final_states[state][sim_name].append(env.info['psi_cached'])

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
fig, ax = plt.subplots(1,1, figsize=(3.375, 2.85), dpi=300)
plt.grid(True)
ax.set_ylabel(r'$1-\cal F$')
ax.set_xlabel('Epoch')
ax.set_yscale('log')
# ax.set_xscale('log')
ax.set_xlim(-50,2050)
ax.set_ylim(8e-4,1)
palette = plt.get_cmap('tab10')

for fock in fock_states:
    state = 'fock' + str(fock)
    # plot training progress of each policy in the background
    for sim_name in rewards[state].keys():
        ax.plot(epochs[state][sim_name], 1-np.array(rewards[state][sim_name]), 
                linestyle='--', alpha=0.2, color=palette(fock-1))    
    all_seeds = np.array([rews for seed, rews in rewards[state].items()])

    # calculate mean log infidelity
    log_infidelity = np.mean(np.log10(1-all_seeds), axis=0)
    infidelity = 10 ** log_infidelity
    
    # # calculate the best infidelity
    # ind = np.argmax(all_seeds[:,-1])
    # infidelity = 1 - all_seeds[ind,:]
    
    median_reward = np.median(all_seeds, axis=0)
    train_epochs = np.array(epochs[state]['seed0'])
    
    ind = [i for i in range(len(median_reward)) if i%1==0]
    ax.plot(train_epochs[ind], infidelity[ind], color=palette(fock-1), linewidth=1.0)
    ax.plot(train_epochs[ind], infidelity[ind], color=palette(fock-1), linewidth=1.0,
            label=fock, linestyle = 'none', marker='.')
ax.legend(ncol=2)

fig.tight_layout()
fig.savefig(figname)