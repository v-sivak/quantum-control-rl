# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 17:13:46 2020

@author: Vladimir Sivak
"""

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from time import time
from numpy import sqrt, pi, exp
from scipy.optimize import curve_fit

from gkp.gkp_tf_env import helper_functions as hf
from gkp.gkp_tf_env import gkp_init
from gkp.gkp_tf_env import policy as plc


class ActionScript(object):
    
    def __init__(self, param):
        
        delta = 0.19
        eps = 0.19
        self.period = 6
        b_amp = sqrt(8*pi/sqrt(3))
        a_amp = sqrt(2*pi/sqrt(3))
        
        self.script = {
            'alpha' : [1j*a_amp*exp(-0j*pi/3), delta*exp(-2j*pi/3),
                       1j*a_amp*exp(-2j*pi/3), delta*exp(-1j*pi/3),
                       1j*a_amp*exp(-1j*pi/3), delta*exp(-0j*pi/3)],
            'beta'  : [1j*b_amp*exp(-2j*pi/3), -eps*exp(-2j*pi/3), 
                       1j*b_amp*exp(-1j*pi/3), -eps*exp(-1j*pi/3), 
                       1j*b_amp*exp(-0j*pi/3), -eps*exp(-0j*pi/3)],
            'phi' : [pi/2]*6,
            'theta' : [param]*6
            }


#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

env = gkp_init(simulate='oscillator_qubit',
               init='Z+', H=1, batch_size=500, episode_length=100, 
               reward_mode = 'fidelity', quantum_circuit_type='v5',
               encoding = 'hexagonal')

savepath = r'E:\VladGoogleDrive\Qulab\GKP\sims\osc_sims\test'
params = [0j] + list(np.linspace(0.005, 0.025, 11, dtype=complex))
states = ['Z+']
make_figure = True

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

lifetimes = np.zeros(len(params))
returns = np.zeros(len(params))

for j in range(len(params)):
    t = time()
    action_script = ActionScript(param=params[j])
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
        returns[j] = np.sum(mean_rewards)
        times = np.arange(env.episode_length)*env.step_duration
        T1_guess = (times[-1]-times[0])/(mean_rewards[0]-mean_rewards[-1])
        popt, pcov = curve_fit(hf.exp_decay, times, mean_rewards,
                               p0=[1, T1_guess])
        lifetimes[j] = popt[1]*1e6
    
    print('(%d): Time %.3f sec' %(j, time()-t))



# Plot summary of the sweep and save the sweep data
if make_figure:
    fig, ax = plt.subplots(1,1, dpi=300, figsize=(10,6))
    ax.set_title(r'Logical lifetime')
    ax.set_ylabel(r'$T_Z$')
    ax.set_xlabel('Sweep parameter')
    ax.plot(params, lifetimes)
    
    fig.savefig(os.path.join(savepath,'summary.png'))

np.save(os.path.join(savepath,'params'), params)
np.save(os.path.join(savepath,'lifetimes'), lifetimes)
np.save(os.path.join(savepath,'returns'), returns)