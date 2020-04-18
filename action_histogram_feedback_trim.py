# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 21:07:58 2020

@author: Vladimir Sivak
"""

import os
os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"]="2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import matplotlib.pyplot as plt
from time import time
from math import sqrt, pi

import tensorflow as tf
from tf_agents import specs
from tf_agents.specs import tensor_spec

from gkp.gkp_tf_env import helper_functions as hf
from gkp.gkp_tf_env import tf_env_wrappers as wrappers
from gkp.gkp_tf_env.gkp_tf_env import GKP
from gkp.gkp_tf_env import policy as plc

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

env = GKP(init='random', H=1, batch_size=200, episode_length=30, 
          reward_mode = 'pauli', quantum_circuit_type='v3')

from gkp.action_script import Baptiste_4round as action_script
env = wrappers.ActionWrapper(env, action_script, 'v3')
env = wrappers.FlattenObservationsWrapperTF(env, 
                observations_whitelist=['msmt','alpha','beta','eps','phi'])

root_dir = r'E:\VladGoogleDrive\Qulab\GKP\sims\PPO\Baptiste'
policy_dir = r'rnn_maxstep24_lr1e-5_pauli_4round_lr3e-6\policy\002500000'
policy = tf.compat.v2.saved_model.load(os.path.join(root_dir,policy_dir))

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

obs_cache = []
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
    obs_cache.append(time_step.observation)
    reward_cache.append(time_step.reward)
    print('%d: Time %.3f sec' %(counter, time()-t))

reward_cache = np.array(reward_cache)
obs_cache = np.array(obs_cache) # shape=[t,b,2]
alpha_cache = obs_cache[:,:,0:2]
epsilon_cache = obs_cache[:,:,4:6]

# Plot histogram of all ALPHA
all_actions = alpha_cache.reshape([env.episode_length*env.batch_size,2])
H, Re_edges, Im_edges = np.histogram2d(all_actions[:,0], all_actions[:,1], 
                                    bins=51, range=[[-1,1],[-1,1]])
Re_centers = (Re_edges[1:] + Re_edges[:-1]) / 2
Im_centers = (Im_edges[1:] + Im_edges[:-1]) / 2

fig, ax = plt.subplots(1,1)
ax.set_title(r'Feedback amplitude $\alpha$')
ax.set_xlabel(r'${\rm Re}\, \alpha$')
ax.set_ylabel(r'${\rm Im}\, \alpha$')
ax.pcolormesh(Re_centers, Im_centers, np.log10(np.transpose(H)), cmap='Reds')

# Plot histogram of all EPSILON
all_actions = epsilon_cache.reshape([env.episode_length*env.batch_size,2])
H, Re_edges, Im_edges = np.histogram2d(all_actions[:,0], all_actions[:,1], 
                                    bins=51, range=[[-1,1],[-1,1]])
Re_centers = (Re_edges[1:] + Re_edges[:-1]) / 2
Im_centers = (Im_edges[1:] + Im_edges[:-1]) / 2

fig, ax = plt.subplots(1,1)
ax.set_title(r'Trim amplitude $\varepsilon$')
ax.set_xlabel(r'${\rm Re}\, \varepsilon$')
ax.set_ylabel(r'${\rm Im}\, \varepsilon$')
ax.pcolormesh(Re_centers, Im_centers, np.log10(np.transpose(H)), cmap='Reds')


# Scatter plot of ALPHA in separate panels for each time step
fig, axes = plt.subplots(6,5, sharex=True, sharey=True)
plt.suptitle(r'Feedback amplitude $\alpha$')
palette = plt.get_cmap('tab10')
axes = axes.ravel()
axes[0].set_xlim(-1.05,1.05)
axes[0].set_ylim(-1.05,1.05)
for t in range(30):
    axes[t].plot(alpha_cache[t,:,0],alpha_cache[t,:,1],linestyle='none',
            marker='.', markersize=2, color=palette(t % 10))

# Scatter plot of EPSILON in separate panels for each time step
fig, axes = plt.subplots(6,5, sharex=True, sharey=True)
plt.suptitle(r'Trim amplitude $\varepsilon$')
palette = plt.get_cmap('tab10')
axes = axes.ravel()
axes[1].set_xlim(-1.05,1.05)
axes[1].set_ylim(-1.05,1.05)
for t in range(30):
    axes[t].plot(epsilon_cache[t,:,0],epsilon_cache[t,:,1],linestyle='none',
            marker='.', markersize=2, color=palette(t % 10))
    
# Plot rewards during the episode
fig, ax = plt.subplots(1,1)
ax.set_xlabel('Time step')
ax.set_ylabel('Reward')
mean = np.mean(reward_cache, axis=1)
std = np.std(reward_cache, axis=1)
ax.plot(range(env.episode_length), mean, color='black')
ax.fill_between(range(env.episode_length), 
                np.clip(mean-std,-1,1), np.clip(mean+std,-1,1))
print(np.sum(mean))

