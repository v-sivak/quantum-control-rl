# -*- coding: utf-8 -*-
"""
Created on Wed May 27 08:28:27 2020

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

# initialize environment and policy
env = gkp_init(simulate='oscillator',
                init='Z+', H=1, batch_size=2000, episode_length=30, 
                reward_mode='fidelity', quantum_circuit_type='v2',
                T1_osc=1000e-6)

from gkp.action_script import phase_estimation_symmetric_with_trim_4round as action_script
policy = plc.ScriptedPolicy(env.time_step_spec(), action_script)

# collect trajectories
all_obs = []
reps = 5 # serialize if batch size is small due to memory issues
for i in range(reps):
    time_step = env.reset()
    policy_state = policy.get_initial_state(env.batch_size)
    counter = 0
    while not time_step.is_last()[0]:
        t = time()
        action_step = policy.action(time_step, policy_state)      
        policy_state = action_step.state
        time_step = env.step(action_step.action)
        counter += 1
        print('%d: Time %.3f sec' %(counter, time()-t))
    all_obs.append(np.array(env.history['msmt'][1:]).squeeze())

obs = np.concatenate(all_obs, axis=1)
T = env.episode_length
R = np.zeros([T,T])    # correlation matrix

# compute correlations
corr = lambda x, y : (x*y).mean(axis=0) - x.mean(axis=0)*y.mean(axis=0)
r = lambda x, y : corr(x,y)/np.sqrt(corr(x,x)*corr(y,y))

for i in range(T):
    for j in range(T):
        R[i,j] = r(obs[i],obs[j])
        
# plot things
fig, ax = plt.subplots(1,1)
ax.set_xlabel('Episode step')
ax.set_ylabel('Episode step')
ax.pcolormesh(range(T), range(T), np.transpose(R), cmap='seismic', 
              vmax=1,vmin=-1)
