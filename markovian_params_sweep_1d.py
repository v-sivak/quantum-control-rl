# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 17:13:46 2020

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
from math import sqrt, pi
from tf_agents.specs import tensor_spec
from tf_agents import specs
import gkp_helper_functions as hf
from scipy.optimize import curve_fit
import gkp_tf_env_wrappers as wrappers

from gkp_tf_env import GKP
import policy as plc


class ActionScript(object):
    
    def __init__(self, delta):
        self.delta = delta

        self.period = 4

        self.beta = [2*sqrt(pi)+0j, 2j*sqrt(pi), -2*sqrt(pi)+0j, -2j*sqrt(pi)]
        
        self.alpha = [0j, 0j, 1j*delta, -delta+0j]
        
        self.phi = [0, 0, pi/2, pi/2]



env = GKP(init='random', H=1, batch_size=200, episode_length=200, 
          reward_mode = 'stabilizers', quantum_circuit_type='v4')


savepath = r'E:\VladGoogleDrive\Qulab\GKP\sims\Error_model\1d_sweep\gate_and_read_errors'
amplitudes = np.linspace(0.0, 1.0, 20, dtype=complex)
lifetimes = np.zeros(amplitudes.shape)
returns = np.zeros(amplitudes.shape)

for jj, a in enumerate(amplitudes):

    action_script = ActionScript(delta=a)
    policy = plc.ScriptedPolicyV1(env.time_step_spec(), action_script)
    
    # This big part is essentially the same as in T1 measurement  
    states = ['Z+']
    results = {state : np.zeros(env.episode_length) for state in states}
    rewards = {state : np.zeros((env.episode_length, env.batch_size)) 
               for state in states}
    for state in states:
        pauli = env.code_displacements[state[0]]
        cache = [] # store intermediate states
        reward_cache = []
        env.init = state
        
        time_step = env.reset()
        policy_state = policy.get_initial_state(env.batch_size)
        counter = 0
        while not time_step.is_last()[0]:
            t = time()
            counter += 1
            action_step = policy.action(time_step, policy_state)      
            policy_state = action_step.state
            time_step = env.step(action_step.action)
            cache.append(env.info['psi_cached'])
            reward_cache.append(time_step.reward)
            print('%d: Time %.3f sec' %(counter, time()-t))

        # measure T1 using cached states
        pauli_batch = tf.stack([pauli]*env.batch_size)
        pauli_batch = tf.cast(pauli_batch, tf.complex64)    
        phi_batch = tf.stack([0.0]*env.batch_size)
        for j, psi in enumerate(cache):   
            _, z = env.phase_estimation(psi, pauli_batch, phi_batch)
            results[state][j] = np.mean(z)
            rewards[state][j] = reward_cache[j]        
    # Plot T1
    fig, ax = plt.subplots(1,1)
    ax.set_title('T1')
    ax.set_xlabel('Time (us)')
    palette = plt.get_cmap('tab10')
    times = np.arange(env.episode_length)*env.step_duration
    for i, state in enumerate(states):
        ax.plot(times*1e6, results[state], color=palette(i),
                marker='.', linestyle='None')
        popt, pcov = curve_fit(hf.exp_decay, times, np.abs(results[state]),
                               p0=[1, env.T1_osc])
        ax.plot(times*1e6, hf.exp_decay(times, popt[0], popt[1]), 
                label = state + ' : %.2f us' %(popt[1]*1e6),
                linestyle='--', color=palette(i),)
        lifetimes[jj] = popt[1]*1e6
    ax.legend()
    fig.savefig(os.path.join(savepath,'T1_feedback_amplitude_%.2f.png' %a))
    
    # Plot rewards
    fig, ax = plt.subplots(1,1)
    ax.set_title('Reward')
    ax.set_xlabel('Time (us)')
    palette = plt.get_cmap('tab10')
    times = np.arange(env.episode_length)*env.step_duration
    for i, state in enumerate(states):
        mean_rewards = rewards[state].mean(axis=1) #average across episodes
        ind = np.where(mean_rewards!=0)[0]
        ax.plot(times[ind]*1e6, mean_rewards[ind], color=palette(i),
                label = 'mean: %.4f' %(np.mean(mean_rewards[ind])))
        returns[jj] = np.sum(mean_rewards)
    ax.legend()
    fig.savefig(os.path.join(savepath,'Reward_feedback_amplitude_%.2f.png' %a))

# Plot summary of the sweep and save the sweep data
fig, ax = plt.subplots(2,1, dpi=300)
ax[0].set_title('State ' + states[-1])
ax[0].set_ylabel(r'$T_1$ (us)')
ax[0].plot(amplitudes, lifetimes)
ax[1].set_ylabel('Mean return')    
ax[1].plot(amplitudes, returns)
ax[1].set_xlabel('Feedback displacement amplitude')
fig.savefig(os.path.join(savepath,'summary.png'))

np.save(os.path.join(savepath,'amplitudes'), amplitudes)
np.save(os.path.join(savepath,'lifetimes'), lifetimes)
np.save(os.path.join(savepath,'returns'), returns)