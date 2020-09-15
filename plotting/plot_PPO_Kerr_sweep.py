# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 18:38:44 2020

@author: Vladimir Sivak
"""

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as  np
import tensorflow as tf
import matplotlib.pyplot as plt
from gkp.gkp_tf_env import helper_functions as hf
from gkp.gkp_tf_env import gkp_init
from gkp.action_script import hexagonal_phase_estimation_symmetric_6round as action_script
from gkp.gkp_tf_env import tf_env_wrappers as wrappers

root_dir = r'E:\VladGoogleDrive\Qulab\GKP\sims\PPO\Kerr_sweep_4000'

names = [
    # '50us_qubit_with_rotation',
    # '50us_qubit_without_rotation',
    # '120us_qubit_with_rotation',
    # '120us_qubit_without_rotation',
    'perfect_qubit_with_rotation',
    'perfect_qubit_without_rotation']

palette = plt.get_cmap('tab10')
    
K_vals =  [1] + list(np.arange(5,55,5))
T1 = {}
  
for i in range(len(names)):
    train_dir = os.path.join(root_dir,names[i])
    theta_flag = True if 'with_rotation' in names[i] else False
    to_learn = {'alpha':True, 'beta':True, 'phi':False, 'theta':theta_flag}
    T1[names[i]] = []
    
    for K in K_vals:
        # Load the policy
        data_dir = os.path.join(train_dir, 'K' + str(K), 'policy')
        policy_dir = os.path.join(data_dir, '000060000')
        policy = tf.compat.v2.saved_model.load(policy_dir)
        
        # Additional simulation parameters
        kwargs = {'K' : K, 't_gate' : 1.2e-6/np.sqrt(np.sqrt(K)), 'T1_osc' : 250e-6}
        if 'perfect' in names[i]:
            kwargs['simulate'] = 'oscillator'
        else:
            kwargs['simulate'] = 'oscillator_qubit'
            T1_qb = int(names[i][:names[i].find('us')])
            kwargs['T1_qb'] = T1_qb*1e-6
        
        # Initialize environment
        env = gkp_init(init='X+', H=1, T=6, attn_step=1, batch_size=3000, 
                   episode_length=200, reward_mode = 'fidelity',
                   quantum_circuit_type='v2', encoding='hexagonal', **kwargs)
        env = wrappers.ActionWrapper(env, action_script, to_learn)

        # Fit logical lifetime
        fit_params = hf.fit_logical_lifetime(env, policy, plot=False, reps=1, 
                                             states=['X+'], save_dir=data_dir)
        T1[names[i]].append(fit_params['X+'][1]*1e6) # convert to us

# Plot things
fig, ax = plt.subplots(1,1, figsize=(7,4))
ax.set_title(r'Hexagonal code, $t_{gate}\propto 1\,/\,\sqrt[4]{Kerr}$')
ax.set_ylabel(r'Logical  lifetime ($\,\mu s\, $)')
ax.set_xlabel('Kerr (Hz)')
for i in range(len(names)):
    color = palette(i//2)
    linestyle = '-' if 'with_' in names[i] else '--'
    ax.plot(K_vals, T1[names[i]], label=names[i], color=color, linestyle=linestyle)
ax.plot(K_vals, [250]*len(K_vals), color='black', label=r'$T_1$ oscillator')
# Put a legend to the right of the current axis
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.62, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        