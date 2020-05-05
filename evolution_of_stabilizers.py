# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:24:22 2020

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
from math import pi
from gkp.gkp_tf_env import tf_env_wrappers as wrappers
from gkp.gkp_tf_env.oscillator_env import OscillatorGKP
from gkp.gkp_tf_env import policy as plc





# env = GKP(init='random', H=1, batch_size=600, episode_length=100, 
#           reward_mode = 'mixed', quantum_circuit_type='v3')


# from gkp.action_script import Baptiste_4round as action_script
# to_learn = {'alpha':True, 'beta':True, 'epsilon':True, 'phi':True}
# env = wrappers.ActionWrapper(env, action_script, to_learn)
# env = wrappers.FlattenObservationsWrapperTF(env)

# root_dir = r'E:\VladGoogleDrive\Qulab\GKP\sims\PPO\dict_actions\learn_everything'
# policy_dir = r'deep_rnn_all_maxsteps12_lr1e-3_sqrt_v3\policy\002600000'
# policy = tf.compat.v2.saved_model.load(os.path.join(root_dir,policy_dir))



env = OscillatorGKP(init='vac', H=1, batch_size=100, episode_length=200, 
          reward_mode = 'stabilizers', quantum_circuit_type='v3')

from gkp.action_script import Baptiste_4round as action_script
policy = plc.ScriptedPolicy(env.time_step_spec(), action_script)





names = ['Re(S_p)', 'Im(S_p)', 'Re(S_q)', 'Im(S_q)']
stabilizers = [env.code_map['S_p'], env.code_map['S_p'], 
               env.code_map['S_q'], env.code_map['S_q']]
angles = [0, -pi/2, 0, -pi/2]
results = {name : np.zeros(env.episode_length) for name in names}
cache = [] # store intermediate states
reward_cache = []
times = []

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
    reward_cache.append(np.mean(time_step.reward))
    times.append(t_stop-t_start)
    print('%d: Time %.3f sec' %(counter, times[-1]))

for name, stabilizer, phi in zip(names, stabilizers, angles):
    for i, psi in enumerate(cache):
        stabilizer_batch = tf.stack([stabilizer]*env.batch_size)
        stabilizer_batch = tf.cast(stabilizer_batch, tf.complex64)
        phi_batch = tf.stack([phi]*env.batch_size)
        _, z = env.phase_estimation(psi, stabilizer_batch, phi_batch)
        results[name][i] = np.mean(z)

# Plot stabilizers Re and Im
fig, ax = plt.subplots(1,1)
ax.set_xlabel('Step')
ax.set_title('Stabilizers')
steps = np.arange(env.episode_length)
for name in names:
    ax.plot(steps, results[name], label = name)

# Plot average reward from reward-collecting rounds only
reward_cache = np.array(reward_cache)
ind = np.where(reward_cache!=0)[0]
ax.plot(steps[ind], reward_cache[ind], color='k')
ax.legend()