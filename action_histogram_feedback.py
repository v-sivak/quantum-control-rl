# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:03:22 2020

@author: Vladimir Sivak
"""

import os
os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"]="2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from time import time
import gkp_tf_env_wrappers as wrappers
from gkp_tf_env import GKP
import policy as plc


env = GKP(init='random', H=1, batch_size=600, episode_length=24, 
          reward_mode = 'pauli', quantum_circuit_type='v1')


# import action_script_phase_estimation_4round as action_script
# policy = plc.ScriptedPolicyV1(env.time_step_spec(), action_script)
# env.MAX_AMPLITUDE = 1.0

import action_script_phase_estimation_4round as action_script
env = wrappers.ActionWrapperFeedback(env, action_script)
env = wrappers.FlattenObservationsWrapperTF(env, 
                    observations_whitelist=['msmt','alpha','beta','phi'])

root_dir = r'E:\VladGoogleDrive\Qulab\GKP\sims\PPO\pauli_sample_reward'
policy_dir = r'rnn_maxsteps24_lr1e-5_clip\policy\001450000'
policy = tf.compat.v2.saved_model.load(os.path.join(root_dir,policy_dir))


action_cache = []
reward_cache = []

time_step = env.reset()
policy_state = policy.get_initial_state(env.batch_size)
counter = 0
while not time_step.is_last()[0]:
    t = time()
    counter += 1
    action_step = policy.action(time_step, policy_state)
    policy_state = action_step.state
    time_step = env.step(action_step.action)
    action_cache.append(action_step.action)
    reward_cache.append(time_step.reward)
    print('%d: Time %.3f sec' %(counter, time()-t))


action_cache = np.array(action_cache)*env.MAX_AMPLITUDE   # shape=[t,b,?]
action_cache = action_cache[:,:,:2] # leave only feedback, shape=[t,b,2]

# Plot histogram of all actions
all_actions = action_cache.reshape([env.episode_length*env.batch_size,2])
H, Re_edges, Im_edges = np.histogram2d(all_actions[:,0], all_actions[:,1], 
                                    bins=51, range=[[-1,1],[-1,1]])
Re_centers = (Re_edges[1:] + Re_edges[:-1]) / 2
Im_centers = (Im_edges[1:] + Im_edges[:-1]) / 2

fig, ax = plt.subplots(1,1)
ax.set_title(r'Feedback amplitude $\alpha$')
ax.set_xlabel(r'${\rm Re}\, \alpha$')
ax.set_xlabel(r'${\rm Im}\, \alpha$')
ax.pcolormesh(Re_centers, Im_centers, np.log10(np.transpose(H)), cmap='Reds')

# Scatter plot of actions in separate panels for each time step
fig, axes = plt.subplots(6,5, sharex=True, sharey=True)
plt.suptitle(r'Feedback amplitude $\alpha$')
palette = plt.get_cmap('tab10')
axes = axes.ravel()
axes[0].set_xlim(-1.05,1.05)
axes[0].set_ylim(-1.05,1.05)
for t in range(env.episode_length):
    axes[t].plot(action_cache[t,:,0],action_cache[t,:,1],linestyle='none',
            marker='.', markersize=2, color=palette(t % 10))
    
# Plot rewards during the episode
fig, ax = plt.subplots(1,1)
ax.set_xlabel('Time step')
ax.set_ylabel('Reward')
reward_cache = np.array(reward_cache)
mean = np.mean(reward_cache, axis=1)
std = np.std(reward_cache, axis=1)
ax.plot(range(env.episode_length), mean, color='black')
ax.fill_between(range(env.episode_length), 
                np.clip(mean-std,-1,1), np.clip(mean+std,-1,1))
print(np.sum(mean))