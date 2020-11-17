# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 19:55:57 2020

@author: qulab
"""

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="1"

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

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
### Setup matplotlib
figname = r'E:\VladGoogleDrive\Qulab\GKP\paper\figs\train' # where to save

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
N=40

env = gkp_init(simulate='snap_and_displacement', channel='quantum_jumps',
               init='vac', H=1, T=3, attn_step=1, batch_size=1, N=N,
               episode_length=3, phase_space_rep='wigner')

action_script = 'snap_and_displacements_3round'
action_scale = {'alpha':4, 'theta':pi}
to_learn = {'alpha':True, 'theta':True}

action_script = action_scripts.__getattribute__(action_script)
env = wrappers.ActionWrapper(env, action_script, action_scale, to_learn)


root_dir = {
    'bin0' : r'E:\data\gkp_sims\PPO\examples\bin0_state_prep_lr1e-4',
    'bin1' : r'E:\data\gkp_sims\PPO\examples\bin1_state_prep_lr1e-3'
    }

target_state_qt_vector = {
    'bin0' : qt.tensor(qt.basis(2,0), qt.basis(N,2)),
    'bin1' : qt.tensor(qt.basis(2,0),(qt.basis(N,0)+qt.basis(N,4)).unit())
    }

epochs = {
    'bin0' : np.arange(0, 200 + 1e-10, 10),
    'bin1' : np.arange(0, 2000 + 1e-10, 100)
    }

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

rewards = {'bin0' : {}, 'bin1' : {}}

for state in ['bin0', 'bin1']:
    # setup overlap reward for this state
    reward_kwargs = {'reward_mode'  : 'overlap', 
                     'target_state' : target_state_qt_vector[state]}
    env.setup_reward(reward_kwargs)
    
    # collect episodes with different policies
    for sim_name in os.listdir(root_dir[state]):
        print(sim_name)
        rewards[state][sim_name] = []
        sim_dir = os.path.join(root_dir[state], sim_name)    
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

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# Plot everything

fig, ax = plt.subplots(1,1, figsize=(3.375, 2), dpi=300)
plt.grid(True)
ax.set_ylabel(r'$1-\cal F$')
ax.set_xlabel('Epoch')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim(8,2200)
ax.set_ylim(1e-4,1)

for state in ['bin0', 'bin1']:
    # plot training progress of each policy in the background
    for sim_name in rewards[state].keys():
        ax.plot(epochs[state], 1-np.array(rewards[state][sim_name]), linestyle='--', alpha=0.4)
    
    # calculate and plot the meadian (less sensitive to outliers)
    avg_reward = np.median(list(rewards[state][i] for i in rewards[state].keys()), axis=0)
    ax.plot(epochs[state], 1-avg_reward, color='black', linewidth=1.0)

fig.tight_layout()
fig.savefig(figname)
