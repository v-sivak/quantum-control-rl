# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 17:18:12 2020

@author: Vladimir Sivak
"""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from time import time
from numpy import sqrt, pi, exp
from scipy.optimize import curve_fit

from rl_tools.tf_env import helper_functions as hf
from rl_tools.tf_env import tf_env_wrappers as wrappers
from rl_tools.tf_env import env_init
from rl_tools.tf_env import policy as plc


class ActionScript(object):
    
    def __init__(self, delta, eps):
        self.delta = delta
        self.eps = eps

        self.period = 6

        b_amp = 2*sqrt(pi)
        a_amp = sqrt(pi)

        self.beta = [b_amp+0j, 1j*b_amp]*2 + [eps+0j, 1j*eps]
        
        self.alpha = [a_amp+0j] + [-1j*delta, delta+0j]*2 + [-1j*a_amp]

        self.phi = [pi/2]*6

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

env = env_init(control_circuit='oscillator',
               init='Z+', H=1, batch_size=800, episode_length=60, 
               reward_mode = 'fidelity', quantum_circuit_type='v2',
               encoding = 'square')

savepath = r'E:\VladGoogleDrive\Qulab\GKP\sims\osc_sims\test'
feedback_amps = np.linspace(0.15, 0.24, 10, dtype=complex)
trim_amps = np.linspace(0.15, 0.24, 10, dtype=complex)
states = ['Z+']
make_figure = False

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

lifetimes = np.zeros((len(feedback_amps), len(trim_amps)))
returns = np.zeros((len(feedback_amps), len(trim_amps)))

for j in range(len(feedback_amps)):
    for i in range(len(trim_amps)):
        t = time()
        action_script = ActionScript(delta=feedback_amps[j], eps=trim_amps[i])
        policy = plc.ScriptedPolicy(env.time_step_spec(), action_script)
        
        for state in states:
            if '_env' in env.__dir__(): 
                env._env.init = state
            else:
                env.init = state

            # Collect batch of episodes
            time_step = env.reset()
            policy_state = policy.get_initial_state(env.batch_size)
            rewards = np.zeros((env.episode_length, env.batch_size))
            counter = 0
            while not time_step.is_last()[0]:
                action_step = policy.action(time_step, policy_state)
                policy_state = action_step.state
                time_step = env.step(action_step.action)
                rewards[counter] = time_step.reward
                counter += 1
            
            # Fit T1
            mean_rewards = rewards.mean(axis=1) # average across episodes
            returns[j,i] = np.sum(mean_rewards)
            times = np.arange(env.episode_length)*env.step_duration
            T1_guess = (times[-1]-times[0])/(mean_rewards[0]-mean_rewards[-1])
            popt, pcov = curve_fit(hf.exp_decay, times, mean_rewards,
                                   p0=[1, T1_guess])
            lifetimes[j,i] = popt[1]*1e6
        
        print('(%d,%d): Time %.3f sec' %(j,i, time()-t))



# Plot summary of the sweep and save the sweep data
if make_figure:
    fig, ax = plt.subplots(2,1, dpi=300, figsize=(6,10))
    ax[0].set_title(r'$T_1$ (us)')
    ax[0].set_ylabel('Trim amplitude')
    ax[0].set_xlabel('Feedback amplitude')
    ax[0].pcolormesh(feedback_amps, trim_amps, np.transpose(lifetimes))
    
    i_max, j_max = np.where(lifetimes==lifetimes.max())
    ax[0].plot([feedback_amps[i_max]],[trim_amps[j_max]], 
               marker='.', color='black', markersize='15')
    
    ax[1].set_title('Mean return')
    ax[1].set_ylabel('Trim amplitude')
    ax[1].set_xlabel('Feedback amplitude')
    ax[1].pcolormesh(feedback_amps, trim_amps, np.transpose(returns))
    
    i_max, j_max = np.where(returns==returns.max())
    ax[1].plot([feedback_amps[i_max]],[trim_amps[j_max]], 
               marker='.', color='black', markersize='15')
    
    fig.savefig(os.path.join(savepath,'summary.png'))

np.save(os.path.join(savepath,'feedback_amps'), feedback_amps)
np.save(os.path.join(savepath,'trim_amps'), trim_amps)
np.save(os.path.join(savepath,'lifetimes'), lifetimes)
np.save(os.path.join(savepath,'returns'), returns)