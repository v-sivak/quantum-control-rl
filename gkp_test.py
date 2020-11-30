# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:13:49 2020

@author: Vladimir Sivak
"""

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import tensorflow as tf
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from gkp.gkp_tf_env import policy as plc
from gkp.gkp_tf_env import helper_functions as hf
from gkp.gkp_tf_env import tf_env_wrappers as wrappers
from gkp.gkp_tf_env import gkp_init
from simulator.utils import expectation

N=40
target_state = qt.tensor(qt.basis(2,0), qt.basis(N,3))
reward_kwargs = {'reward_mode' : 'overlap', 
                 'target_state' : target_state}
kwargs = {'N': N}


env = gkp_init(simulate='Alec_universal_gate_set', 
               channel='quantum_jumps',
               reward_kwargs=reward_kwargs,
               init='vac', H=1, T=6, attn_step=1, batch_size=1, episode_length=6,
               encoding='square', **kwargs)

# from gkp.action_script import v2_phase_estimation_with_trim_4round as action_script
from gkp.action_script import Alec_universal_gate_set_6round as action_script
# to_learn = {'alpha':True, 'beta':True, 'phi':False, 'theta':False}
to_learn = {'beta':True, 'phi':True}
env = wrappers.ActionWrapper(env, action_script, to_learn)

root_dir = r'E:\VladGoogleDrive\Qulab\GKP\sims\PPO\CT_qubit_rot\fock3_beta3_B100_tomo100_lr1e-3_baseline_2'
policy_dir = r'policy\000076000'
policy = tf.compat.v2.saved_model.load(os.path.join(root_dir,policy_dir))

# from gkp.action_script import v2_phase_estimation_with_trim_4round as action_script
# policy = plc.ScriptedPolicy(env.time_step_spec(), action_script)


### Plot cardinal points
if 0:
    for state_name in env.states.keys():
        state = tf.reshape(env.states[state_name], [1,env.N])
        hf.plot_wigner_tf_wrapper(state, title=state_name)

### Simulate one episode
if 1:
    n = [] # average photon number 
    time_step = env.reset()
    policy_state = policy.get_initial_state(env.batch_size)
    while not time_step.is_last():
        action_step = policy.action(time_step, policy_state)
        policy_state = action_step.state
        time_step = env.step(action_step.action)
        n.append(float(expectation(env._state, env.n)))
        env.render()
    # hf.plot_wigner_tf_wrapper(env.info['psi_cached'], tensorstate=env.tensorstate)
    
    fig, ax = plt.subplots(1,1)
    ax.set_xlabel('Step')
    ax.set_ylabel(r'$\langle \, n \, \rangle$')
    ax.plot(range(len(n)), n)
