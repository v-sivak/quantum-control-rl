# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 21:07:58 2020

@author: Vladimir Sivak
"""

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
import matplotlib.pyplot as plt
from time import time
import tensorflow as tf

from gkp.gkp_tf_env import tf_env_wrappers as wrappers
from gkp.gkp_tf_env import gkp_init
from gkp.gkp_tf_env import policy as plc

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
### Initialize env and policy

env = gkp_init(simulate='oscillator', 
                init='random', H=1, batch_size=600, episode_length=30, 
                reward_mode = 'pauli', quantum_circuit_type='v2')

from gkp.action_script import phase_estimation_symmetric_with_trim_4round as action_script
# from gkp.action_script import phase_estimation_4round as action_script
to_learn = {'alpha':True, 'beta':True, 'phi':True}
env = wrappers.ActionWrapper(env, action_script, to_learn)
env = wrappers.FlattenObservationsWrapperTF(env)

root_dir = r'E:\VladGoogleDrive\Qulab\GKP\sims\PPO\OscillatorGKP'
policy_dir = r'rnn2_mlp_steps24_aw_lr1e-5_v2\policy\000560000'
policy = tf.compat.v2.saved_model.load(os.path.join(root_dir,policy_dir))


# from gkp.action_script import Baptiste_4round as action_script
# policy = plc.ScriptedPolicy(env.time_step_spec(), action_script)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

### Collect a batch of episodes

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
    reward_cache.append(time_step.reward)
    print('%d: Time %.3f sec' %(counter, time()-t))


### Plot actions and rewards

for a in to_learn.keys() - ['phi']:
    # shape=[t,b,2] 
    action_cache = np.stack(env.history[a][-env.episode_length:]).squeeze()    
    all_actions = action_cache.reshape([env.episode_length*env.batch_size,2])

    lim = env.scale[a]
    # Plot combined histogram of actions at all time steps
    H, Re_edges, Im_edges = np.histogram2d(all_actions[:,0], all_actions[:,1], 
                                        bins=51, 
                                        range=[[-lim-0.2, lim+0.2],
                                               [-lim-0.2, lim+0.2]])
    Re_centers = (Re_edges[1:] + Re_edges[:-1]) / 2
    Im_centers = (Im_edges[1:] + Im_edges[:-1]) / 2
    
    fig, ax = plt.subplots(1,1)
    ax.set_aspect('equal')
    ax.set_title(a)
    ax.set_xlabel(r'${\rm Re}$ ' + a)
    ax.set_ylabel(r'${\rm Im}$ ' + a)
    ax.set_xlim(-lim-0.2, lim+0.2)
    ax.set_ylim(-lim-0.2, lim+0.2)
    ax.pcolormesh(Re_centers, Im_centers, np.log10(np.transpose(H)), 
                  cmap='Reds')
    plt.grid(True)

    # Scatter plot of actions in separate panels for each time step
    fig, axes = plt.subplots(6,5, sharex=True, sharey=True)
    plt.suptitle(a)
    palette = plt.get_cmap('tab10')
    axes = axes.ravel()
    axes[0].set_xlim(-lim-0.2, lim+0.2) #(-0.5,0.5)
    axes[0].set_ylim(-lim-0.2, lim+0.2) #(-0.5,0.5)
    for t in range(30):
        axes[t].plot(action_cache[t,:,0], action_cache[t,:,1],
                     linestyle='none', marker='.', markersize=2, 
                     color=palette(t % 10))


# Plot histogram of 'phi'
a = 'phi'
action_cache = np.stack(env.history[a][-env.episode_length:]).squeeze()
all_actions = action_cache.flatten()
H, edges = np.histogram(all_actions, bins=101, range=[-2,2])
centers = (edges[1:] + edges[:-1]) / 2
fig, ax = plt.subplots(1,1)
ax.set_title(a)
ax.set_ylabel('Counts')
ax.plot(centers, H)


# # Plot rewards during the episode
# fig, ax = plt.subplots(1,1)
# ax.set_xlabel('Time step')
# ax.set_ylabel('Reward')
# reward_cache = np.array(reward_cache)
# mean = np.mean(reward_cache, axis=1)
# std = np.std(reward_cache, axis=1)
# ax.plot(range(env.episode_length), mean, color='black')
# ax.fill_between(range(env.episode_length), 
#                 np.clip(mean-std,-1,1), np.clip(mean+std,-1,1))
# print(np.sum(mean))

