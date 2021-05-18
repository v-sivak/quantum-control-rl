# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 18:26:18 2021

@author: qulab
"""

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# first test the protocols in new environemnts and create "rewards" dataset
import tensorflow as tf
import numpy as np
import qutip as qt
from rl_tools.tf_env import env_init
from math import pi
import rl_tools.action_script as action_scripts
from rl_tools.tf_env import tf_env_wrappers as wrappers
import matplotlib.pyplot as plt
from plotting import plot_config

# CREATE THE ENVIRONMENT
env_kwargs = dict(control_circuit='snap_and_displacement_miscalibrated', init='vac',
                  H=1, T=5, attn_step=1, N=50, batch_size=1, episode_length=5)

target_state = qt.tensor(qt.basis(2,0), qt.basis(50,3))
reward_kwargs = {'reward_mode' : 'overlap',
                  'target_state' : target_state,
                  'postselect_0' : True}

action_script = 'snap_and_displacements'
action_scale = {'alpha':4, 'theta':pi}
to_learn = {'alpha':True, 'theta':True}
action_script = action_scripts.__getattribute__(action_script)

env = env_init(**env_kwargs, reward_kwargs=reward_kwargs)
env = wrappers.ActionWrapper(env, action_script, action_scale, to_learn)

env._env.SNAP_miscalibrated.T = 0.4e-6


# LOAD POLICIES 
root_dir = r'E:\data\gkp_sims\PPO\paper_data\miscalibrated_snap_S7\miscal_t=0.4\seed5'
policy_dir = r'policy\025000'
policy1 = tf.compat.v2.saved_model.load(os.path.join(root_dir,policy_dir))

root_dir = r'E:\data\gkp_sims\PPO\paper_data\miscalibrated_snap_S7\ideal_overlap\seed2'
policy_dir = r'policy\003000'
policy2 = tf.compat.v2.saved_model.load(os.path.join(root_dir,policy_dir))

policies = [policy1, policy2]


figname = r'E:\VladGoogleDrive\Qulab\GKP\paper\figs\bitstrings.pdf'
fig, axes = plt.subplots(2,1, sharex=True, figsize=(1.2,1.9), dpi=300)
axes[0].set_ylim(-0.05,1.05)
palette = plt.get_cmap('tab10')

for j, policy in enumerate(policies):
    ax= axes[j]
    bit_strings = [np.binary_repr(i).zfill(5) for i in range(32)]    
    rewards, norms, probs = [], [], []
    
    for bit_string in bit_strings:
        env._env.bit_string = bit_string
        env._env.norms = []    
        time_step = env.reset()
        policy_state = policy.get_initial_state(env.batch_size)
        while not time_step.is_last()[0]:
            action_step = policy.action(time_step, policy_state)
            policy_state = action_step.state
            time_step = env.step(action_step.action)
        rewards.append(np.mean(time_step.reward))
        norms.append(np.squeeze(env.norms).astype(float))
        
    probs = np.prod(norms, axis=1)**2
    rewards = np.array(rewards)
    print('Total probability of all bit strings: %.4f' %np.sum(probs))
    
    cumulative_probs = [np.sum(probs[np.argsort(probs)][:i+1]) for i in range(32)]

    ax.plot(np.arange(32), cumulative_probs, marker='.', label='probs', color=palette(0))
    ax.plot(np.arange(32), rewards[np.argsort(probs)], marker='.', label='fidelity',color=palette(1))
    
    print('Average fidelity: %.4f' %(np.sum(probs*rewards)))
    
    
    for i in np.flip(np.argsort(probs)):
        print(bit_strings[i] + '  prob: %.5f  fid: %.4f' %(probs[i],rewards[i]))

# ax.legend()
# plt.tight_layout()
fig.savefig(figname)
    
# env._env.bit_string = '01001'
# env._env.norms = []    
# time_step = env.reset()
# env.render()
# policy_state = policy.get_initial_state(env.batch_size)
# while not time_step.is_last()[0]:
#     action_step = policy.action(time_step, policy_state)
#     policy_state = action_step.state
#     time_step = env.step(action_step.action)
#     env.render()
