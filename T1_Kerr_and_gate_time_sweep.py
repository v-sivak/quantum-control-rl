# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:44:12 2020

@author: Vladimir Sivak
"""

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
from numpy import sqrt, pi, exp
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from gkp.gkp_tf_env import helper_functions as hf
from gkp.gkp_tf_env import policy as plc
from gkp.gkp_tf_env import gkp_init

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

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


# Define Kerr sweep range and Kerr-dependent parameters
Kerr = np.linspace(1,51,11) 
t_gate = 1.2e-6/np.sqrt(Kerr) # assume gate time can scale as 1/chi
rotation_angle = 2*np.pi*Kerr*(1.2e-6 + t_gate)*20 # simple heuristic 

states = ['X+', 'Y+', 'Z+']
lifetimes = {state : np.zeros(len(Kerr)) for state in states}

savepath = r'E:\VladGoogleDrive\Qulab\GKP\sims\Kerr\hexagonal_sweep\no_rotation_perfect_qubit'

# Initialize environment and policy
env = gkp_init(simulate='oscillator', encoding='hexagonal', 
               init='X+', H=1, batch_size=2000, episode_length=200, 
               reward_mode='fidelity', quantum_circuit_type='v2')

from gkp.action_script import hexagonal_phase_estimation_symmetric_6round as action_script
policy = plc.ScriptedPolicy(env.time_step_spec(), action_script)

for k in range(len(Kerr)):

    env = gkp_init(simulate='oscillator', encoding='hexagonal', 
                   init='X+', H=1, batch_size=2000, episode_length=200, 
                   reward_mode='fidelity', quantum_circuit_type='v2',
                   K_osc=Kerr[k], t_gate=t_gate[k])  

    # action_script = ActionScript(param=rotation_angle[k])
    # policy = plc.ScriptedPolicy(env.time_step_spec(), action_script)
    
    reps = 1 # serialize episode collection in a loop if can't fit into GPU memory
    B = env.batch_size
    rewards = {state : np.zeros((env.episode_length, B*reps))
               for state in states}
    
    for state in states:
        if '_env' in env.__dir__(): 
            env._env.init = state
        else:
            env.init = state
        
        # Collect batch of episodes, loop if can't fit in GPU memory
        for i in range(reps):
            time_step = env.reset()
            policy_state = policy.get_initial_state(B)
            j = 0
            while not time_step.is_last()[0]:
                action_step = policy.action(time_step, policy_state)
                policy_state = action_step.state
                time_step = env.step(action_step.action)
                rewards[state][j][i*B:(i+1)*B] = time_step.reward
                j += 1


    # Plot average reward from every time step and fit T1
    fig, ax = plt.subplots(1,1)
    ax.set_xlabel('Step')
    ax.set_ylabel(r'$\langle X\rangle$, $\langle Y\rangle$, $\langle Z\rangle$')
    palette = plt.get_cmap('tab10')
    steps = np.arange(env.episode_length)
    for i, state in enumerate(states):
        mean_rewards = rewards[state].mean(axis=1) # average across episodes
        ax.plot(steps, mean_rewards, color=palette(i),
                linestyle='none', marker='.')
        times = steps*env.step_duration
        popt, pcov = curve_fit(hf.exp_decay, times, mean_rewards,
                               p0=[1, env._T1_osc])
        ax.plot(steps, hf.exp_decay(times, popt[0], popt[1]), 
                label = state + ' : %.2f us' %(popt[1]*1e6),
                linestyle='--', color=palette(i))
        lifetimes[state][k] = popt[1]*1e6
    ax.set_title('Logical lifetime')
    ax.legend()
    
# Plot sweep results and save data
fig, ax = plt.subplots(1,1)
ax.set_ylabel('Logical lifetime')
ax.set_xlabel('Kerr (Hz)')
ax.set_ylim(0,3000)
for state in states:
    ax.plot(Kerr, lifetimes[state], label=state)
    np.save(os.path.join(savepath, 'T_'+state[0]), lifetimes[state])
    np.save(os.path.join(savepath, 'Kerr'), Kerr)
ax.plot(Kerr, [env.T1_osc*1e6]*len(Kerr), color='black', label=r'$T_1$')
ax.legend()