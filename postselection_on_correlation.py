# -*- coding: utf-8 -*-
"""
This script is to study the effect of a sudden translation on the correlation
of measurement outcomes. 

Based on this correlation it is possible to implement a post-selection scheme 
which trashes the trajectories with high correlation -- an indication that 
there was a qubit decay. 

Created on Fri Aug 28 10:15:10 2020

@author: Vladimir Sivak
"""

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import matplotlib.pyplot as plt
from gkp.gkp_tf_env import policy as plc
from gkp.gkp_tf_env import gkp_init
from time import time
from tensorflow.keras.backend import batch_dot
from math import sqrt, pi
import tensorflow as tf
from scipy.optimize import curve_fit
from gkp.gkp_tf_env import helper_functions as hf


#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

# initialize environment and policy
env = gkp_init(simulate='oscillator',
                init='X+', H=1, batch_size=6000, episode_length=31, 
                reward_mode='zero', quantum_circuit_type='v2',
                encoding='square')

from gkp.action_script import phase_estimation_symmetric_with_trim_4round as action_script
policy = plc.ScriptedPolicy(env.time_step_spec(), action_script)


translations = np.linspace(-sqrt(pi), sqrt(pi), 100)

T = env.episode_length
R = np.zeros([len(translations),T,T])    # correlation matrix (empty)

for k, a in enumerate(translations):
    # collect trajectories
    time_step = env.reset()
    policy_state = policy.get_initial_state(env.batch_size)
    counter = 0
    while not time_step.is_last()[0]:
        if counter == 10: # apply a sudden large translation
            Tr = env.translate(np.array([a]*env.batch_size))
            env._state = batch_dot(Tr, env._state)
        t = time()
        action_step = policy.action(time_step, policy_state)      
        policy_state = action_step.state
        time_step = env.step(action_step.action)
        counter += 1
        print('%d: Time %.3f sec' %(counter, time()-t))
    obs = tf.squeeze(tf.stack(env.history['msmt'][1:], axis=0))
    
    # compute correlations
    corr = lambda x, y : tf.reduce_mean(x*y, axis=0) - \
        tf.reduce_mean(x, axis=0)*tf.reduce_mean(y, axis=0)
    r = lambda x, y : corr(x,y)/np.sqrt(corr(x,x)*corr(y,y))
    
    for i in range(T):
        for j in range(T):
            R[k,i,j] = r(obs[i],obs[j])
    
    # plot correlation matrix
    if 0:
        fig, ax = plt.subplots(1,1)
        ax.set_xlabel('Episode step')
        ax.set_ylabel('Episode step')
        ax.pcolormesh(range(T), range(T), np.transpose(R[k]), cmap='seismic', 
                      vmax=1,vmin=-1)
    
# plot correlators as a function of translation amplitude
fig, ax = plt.subplots(1,1)
ax.set_ylabel('Correlation')
ax.set_xlabel('Translation amplitude')
for k in range(1,6):
    ax.plot(translations, R[:,10, 10+4*k])
    
    
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# This part simulates oscillator_qubit and post-selects trajectories

env = gkp_init(simulate='oscillator_qubit', 
                init='X+', H=1, batch_size=1000, episode_length=60, 
                reward_mode = 'fidelity', quantum_circuit_type='v2',
                encoding='square')

from gkp.action_script import phase_estimation_symmetric_with_trim_4round as action_script
policy = plc.ScriptedPolicy(env.time_step_spec(), action_script)


reps = 100 # serialize episode collection in a loop if can't fit into GPU memory
B = env.batch_size
states = ['X+']
rewards = {state : np.zeros((env.episode_length, B*reps)) for state in states}
obs = {state : np.zeros((env.episode_length, B*reps)) for state in states}

# can do this only with fidelity reward, because it is used to fit T1
assert env.reward_mode == 'fidelity'

for state in states:
    if '_env' in env.__dir__(): 
        env._env.init = state
    else:
        env.init = state
    
    # Collect batch of episodes, loop if can't fit in GPU memory
    all_obs = []
    for i in range(reps):
        time_step = env.reset()
        policy_state = policy.get_initial_state(B)
        j = 0
        while not time_step.is_last()[0]:
            t = time()
            action_step = policy.action(time_step, policy_state)
            policy_state = action_step.state
            time_step = env.step(action_step.action)
            rewards[state][j][i*B:(i+1)*B] = time_step.reward
            j += 1
            print('%d: Time %.3f sec' %(j, time()-t))
        all_obs.append(tf.squeeze(tf.stack(env.history['msmt'][1:], axis=0)))
    obs[state] = tf.concat(all_obs, axis=1).numpy()
            

# Function that produces a mask of shape (episode_length,batch_size) to indicate
# which trajectory are likely faulty and need to be trashed
def mask(obs):
    T = obs.shape[0]
    B = obs.shape[1]
    error_mask = np.array([False]*B)
    
    def logical_and(condition):
    X = np.array([True]*len(condition[0]))
    for i in range(len(condition)):
        X = np.logical_and(condition[i], X)
        return X
    
    for i in np.arange(0,T-8,2):
        # define multiple conditions to be checked
        C1 = obs[i,:] == obs[i+4,:]
        C2 = obs[i,:] == obs[i+8,:]
        # C3 = obs[i,:] == obs[i+12,:]
        # C4 = obs[i,:] == obs[i+16,:]
        
        mask = logical_and([C1,C2])
        error_mask = np.logical_or(mask, error_mask)
        
    return np.stack([error_mask]*T)

# Plot average reward from every time step and fit T1 (on post-selected trajs)
fig, ax = plt.subplots(1,1)
ax.set_xlabel('Step')
ax.set_ylabel(r'$\langle X\rangle$, $\langle Y\rangle$, $\langle Z\rangle$')
palette = plt.get_cmap('tab10')
steps = np.arange(env.episode_length)
for i, state in enumerate(states):
    mean_rewards = np.ma.array(rewards[state], mask=mask(obs[state])).mean(axis=1)
    ax.plot(steps, mean_rewards, color=palette(i),
            linestyle='none', marker='.')
    times = steps*float(env.step_duration)
    popt, pcov = curve_fit(hf.exp_decay, times, mean_rewards,
                            p0=[1, env._T1_osc])
    ax.plot(steps, hf.exp_decay(times, popt[0], popt[1]), 
            label = state + ' : %.2f us' %(popt[1]*1e6),
            linestyle='--', color=palette(i))
ax.set_title('Logical lifetime')
ax.legend()


# Plot logical lifetime as a function of post-selection survival probability
fig, ax = plt.subplots(1,1)
ax.set_xscale('log')
ax.set_xlabel('Fraction of post-selected trajectories')
ax.set_ylabel('Logical lifetime')
ax.plot([1.0, 0.36966, 0.10666, 0.00307],
        [263, 287, 298, 312])