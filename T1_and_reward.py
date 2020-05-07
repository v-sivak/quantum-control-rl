# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:16:45 2020

@author: Vladimir Sivak
"""

import os
os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"]="2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from time import time
from scipy.optimize import curve_fit
from gkp.gkp_tf_env import helper_functions as hf
from gkp.gkp_tf_env import tf_env_wrappers as wrappers
from gkp.gkp_tf_env import policy as plc
from gkp.gkp_tf_env import gkp_init

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------


# env = OscillatorGKP(init='X+', H=1, batch_size=600, episode_length=200, 
#                     reward_mode = 'pauli', quantum_circuit_type='v1')

# from gkp.action_script import phase_estimation_4round as action_script
# to_learn = {'alpha':True, 'beta':False, 'phi':True}
# env = wrappers.ActionWrapper(env, action_script, to_learn)
# env = wrappers.FlattenObservationsWrapperTF(env)

# root_dir = r'E:\VladGoogleDrive\Qulab\GKP\sims\PPO\dict_actions\alpha_eps_phi'
# policy_dir = r'rnn_maxstep24_batch100_v4\policy\001200000'
# policy = tf.compat.v2.saved_model.load(os.path.join(root_dir,policy_dir))



env = gkp_init(simulate='oscillator',
               init='X+', H=1, batch_size=200, episode_length=200, 
               reward_mode = 'pauli', quantum_circuit_type='v2')

from gkp.action_script import phase_estimation_symmetric_with_trim_4round as action_script
# from gkp.action_script import phase_estimation_8round as action_script
# from gkp.action_script import Baptiste_4round as action_script
policy = plc.ScriptedPolicy(env.time_step_spec(), action_script)


#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

states = ['X+'] #['X+', 'Y+', 'Z+']
results = {state : np.zeros(env.episode_length) for state in states}
rewards = {state : np.zeros((env.episode_length, env.batch_size)) 
           for state in states}
for state in states:
    pauli = env.code_map[state[0]]
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
            linestyle='--', color=palette(i))
ax.legend()


# Plot mean reward from every time step
fig, ax = plt.subplots(1,1)
ax.set_xlabel('Step')
ax.set_ylabel('Reward')
palette = plt.get_cmap('tab10')
steps = np.arange(env.episode_length)
for i, state in enumerate(states):
    mean_rewards = rewards[state].mean(axis=1) #average across episodes
    avg_return = np.sum(mean_rewards)
    ax.plot(steps, mean_rewards, color=palette(i),
            label = 'state ' + state +': %.4f' %avg_return, 
            linestyle='none', marker='.')
ax.set_title('Average return')
ax.legend()