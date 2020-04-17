# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 17:18:12 2020

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
from scipy.optimize import curve_fit

from gkp.gkp_tf_env import helper_functions as hf
from gkp.gkp_tf_env import tf_env_wrappers as wrappers
from gkp.gkp_tf_env.gkp_tf_env import GKP
from gkp.gkp_tf_env import policy as plc


class ActionScript(object):
    
    def __init__(self, delta, eps):
        self.delta = delta
        self.eps = eps

        self.period = 4

        self.beta = [2*sqrt(pi)+0j, 2j*sqrt(pi), 2*sqrt(pi)+0j, 2j*sqrt(pi)]
        
        self.alpha = [0+0j, 0+0j, -1j*delta, delta+0j]
        
        self.epsilon = [0, 0, pi/2, pi/2]
        
        self.phi = [0, 0, pi/2, pi/2]



env = GKP(init='Z+', H=1, batch_size=200, episode_length=200, 
          reward_mode = 'mixed', quantum_circuit_type='v3')


savepath = r'E:\VladGoogleDrive\Qulab\GKP\sims\reward_function\mixed_4round_Baptiste'
feedback_amplitudes = np.linspace(0.0, 1.0, 20, dtype=complex)
trim_amplitudes = np.linspace(0.0, 1.0, 20, dtype=complex)
lifetimes = np.zeros((len(feedback_amplitudes), len(trim_amplitudes)))
returns = np.zeros((len(feedback_amplitudes), len(trim_amplitudes)))

for jj, fa in enumerate(feedback_amplitudes):
    for ii, ta in enumerate(trim_amplitudes):

        action_script = ActionScript(delta=fa, eps=ta)
        policy = plc.ScriptedPolicyV2(env.time_step_spec(), action_script)

        
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
        # fig, ax = plt.subplots(1,1)
        # ax.set_title('T1')
        # ax.set_xlabel('Time (us)')
        # palette = plt.get_cmap('tab10')
        times = np.arange(env.episode_length)*env.step_duration
        for i, state in enumerate(states):
            # ax.plot(times*1e6, results[state], color=palette(i),
            #         marker='.', linestyle='None')
            popt, pcov = curve_fit(hf.exp_decay, times, np.abs(results[state]),
                                   p0=[1, env.T1_osc])
            # ax.plot(times*1e6, hf.exp_decay(times, popt[0], popt[1]), 
            #         label = state + ' : %.2f us' %(popt[1]*1e6),
            #         linestyle='--', color=palette(i),)
            lifetimes[jj,ii] = popt[1]*1e6
        # ax.legend()
        # fig.savefig(os.path.join(savepath,'T1_feedback_%.2f_trim_%.2f.png' %(fa,ta)))
        
        # Plot rewards
        # fig, ax = plt.subplots(1,1)
        # ax.set_title('Reward')
        # ax.set_xlabel('Time (us)')
        # palette = plt.get_cmap('tab10')
        times = np.arange(env.episode_length)*env.step_duration
        for i, state in enumerate(states):
            mean_rewards = rewards[state].mean(axis=1) #average across episodes
            ind = np.where(mean_rewards!=0)[0]
            # ax.plot(times[ind]*1e6, mean_rewards[ind], color=palette(i),
            #         label = 'mean: %.4f' %(np.mean(mean_rewards[ind])))
            returns[jj,ii] = np.sum(mean_rewards)
        # ax.legend()
        # fig.savefig(os.path.join(savepath,'Reward_feedback_%.2f_trim_%.2f.png' %(fa,ta)))

# Plot summary of the sweep and save the sweep data
fig, ax = plt.subplots(2,1, dpi=300, figsize=(6,10))
ax[0].set_title(r'$T_1$ (us)')
ax[0].set_ylabel('Trim amplitude')
ax[0].set_xlabel('Feedback amplitude')
ax[0].pcolormesh(feedback_amplitudes, trim_amplitudes, np.transpose(lifetimes))

i_max, j_max = np.where(lifetimes==lifetimes.max())
ax[0].plot([feedback_amplitudes[i_max]],[trim_amplitudes[j_max]], 
           marker='.', color='black', markersize='15')

ax[1].set_title('Mean return')
ax[1].set_ylabel('Trim amplitude')
ax[1].set_xlabel('Feedback amplitude')
ax[1].pcolormesh(feedback_amplitudes, trim_amplitudes, np.transpose(returns))

i_max, j_max = np.where(returns==returns.max())
ax[1].plot([feedback_amplitudes[i_max]],[trim_amplitudes[j_max]], 
           marker='.', color='black', markersize='15')


fig.savefig(os.path.join(savepath,'summary.png'))

np.save(os.path.join(savepath,'feedback_amplitudes'), feedback_amplitudes)
np.save(os.path.join(savepath,'trim_amplitudes'), trim_amplitudes)
np.save(os.path.join(savepath,'lifetimes'), lifetimes)
np.save(os.path.join(savepath,'returns'), returns)