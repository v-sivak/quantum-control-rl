# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:24:22 2020

@author: Vladimir Sivak
"""

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from time import time
from math import pi
from rl_tools.tf_env import tf_env_wrappers as wrappers
from rl_tools.tf_env import policy as plc
from rl_tools.tf_env import env_init

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

env = env_init(control_circuit='gkp_qec_autonomous_sBs_osc_qb', 
                reward_kwargs={'reward_mode':'zero'},
                init='vac', H=1, T=2, attn_step=1, batch_size=2000, episode_length=60,
                encoding='square')

from rl_tools.action_script import gkp_qec_autonomous_sBs_2round as action_script
policy = plc.ScriptedPolicy(env.time_step_spec(), action_script)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# # What operators to measure 
# names = [r'Re($S_1$)', r'Re($S_2$)',
#          r'Im($S_1$)', r'Im($S_2$)']
# # Translation amplitudes
# stabilizers = [np.sqrt(pi), 2j*np.sqrt(pi)]*2
# # Qubit measurement angles
# angles = [0]*2 + [-pi/2]*2
# # Params for plotting
# lines = ['-']*2 + ['--']*2
# palette = plt.get_cmap('tab10')
# colors = [palette(0), palette(1), palette(2)]*2


# What operators to measure 
names = [r'Re($S_x$)', r'Re($S_y$)', r'Re($S_z$)',
         r'Im($S_x$)', r'Im($S_y$)', r'Im($S_z$)']
# Translation amplitudes
stabilizers = [env.code_map[S] for S in ['S_x','S_y','S_z']*2]
# Qubit measurement angles
angles = [0]*3 + [-pi/2]*3
# Params for plotting
lines = ['-']*3 + ['--']*3
palette = plt.get_cmap('tab10')
colors = [palette(0), palette(1), palette(2)]*2

results = {name : np.zeros(env.episode_length) for name in names}
cache = [] # store intermediate states (after feedback)
times = [] # for timing every round

# Collect batch of episodes
time_step = env.reset()
policy_state = policy.get_initial_state(env.batch_size)
counter = 0
while not time_step.is_last()[0]:
    t_start = time()
    counter += 1
    action_step = policy.action(time_step, policy_state)      
    policy_state = action_step.state
    time_step = env.step(action_step.action)
    t_stop = time()
    cache.append(env.info['psi_cached'])
    times.append(t_stop - t_start)
    print('%d: Time %.3f sec' %(counter, times[-1]))

# Measure stabilizers on cached wavefunctions
for name, stabilizer, phi in zip(names, stabilizers, angles):
    for i, psi in enumerate(cache):
        stabilizer_unitary = env.translate([stabilizer])
        _, z = env.phase_estimation(psi, stabilizer_unitary, [phi])
        results[name][i] = np.mean(z)

# Plot stabilizers Re and Im
fig, ax = plt.subplots(1,1)
ax.set_xlabel('Step')
ax.set_title('Stabilizers')
steps = np.arange(env.episode_length)
ax.set_ylim(-0.05, 0.7)
for i, name in enumerate(names):
    ax.plot(steps, results[name], label=name, 
            linestyle=lines[i], color=colors[i])
ax.legend()
