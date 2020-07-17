# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:16:45 2020

@author: Vladimir Sivak
"""

import os
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

env = gkp_init(simulate='oscillator', 
               init='X+', H=4, T=4, batch_size=2500, episode_length=400, 
               reward_mode='fidelity', quantum_circuit_type='v2',
               encoding='square')

# from gkp.action_script import phase_estimation_symmetric_with_trim_4round as action_script
# # from gkp.action_script import hexagonal_phase_estimation_symmetric_6round as action_script
# to_learn = {'alpha':True, 'beta':True, 'phi':False}
# env = wrappers.ActionWrapper(env, action_script, to_learn)

# root_dir = r'E:\VladGoogleDrive\Qulab\GKP\sims\PPO\July\OscillatorGKP'
# exp_name = 'test'
# policy_dir = r'policy\000040000'
# policy = tf.compat.v2.saved_model.load(os.path.join(root_dir,exp_name,policy_dir))


# from gkp.action_script import hexagonal_phase_estimation_symmetric_6round as action_script
from gkp.action_script import phase_estimation_symmetric_with_trim_4round as action_script
policy = plc.ScriptedPolicy(env.time_step_spec(), action_script)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
reps = 1 # serialize episode collection in a loop if can't fit into GPU memory
B = env.batch_size
states = ['X+', 'Y+', 'Z+']
results = {state : np.zeros(env.episode_length) for state in states}
rewards = {state : np.zeros((env.episode_length, B*reps))
           for state in states}

# can do this only with fidelity reward, because it is used to fit T1
assert env.reward_mode == 'fidelity'

for state in states:
    if '_env' in env.__dir__(): 
        env._env.init = state
    else:
        env.init = state

    pauli = env.code_map[state[0]] # which Pauli to measure
    
    # Collect batch of episodes, loop if can't fit in GPU memory
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


# Plot average reward from every time step and fit T1
fig, ax = plt.subplots(1,1)
ax.set_xlabel('Step')
ax.set_ylabel('Reward')
palette = plt.get_cmap('tab10')
steps = np.arange(env.episode_length)
for i, state in enumerate(states):
    mean_rewards = rewards[state].mean(axis=1) # average across episodes
    ax.plot(steps, mean_rewards, color=palette(i),
            linestyle='none', marker='.')
    times = steps*env.step_duration
    popt, pcov = curve_fit(hf.exp_decay, times, mean_rewards,
                           p0=[1, env.T1_osc])
    ax.plot(steps, hf.exp_decay(times, popt[0], popt[1]), 
            label = state + ' : %.2f us' %(popt[1]*1e6),
            linestyle='--', color=palette(i))
ax.set_title('Logical lifetime')
ax.legend()