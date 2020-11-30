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
from numpy import pi
import matplotlib.pyplot as plt
from gkp.gkp_tf_env import policy as plc
from gkp.gkp_tf_env import helper_functions as hf
from gkp.gkp_tf_env import tf_env_wrappers as wrappers
from gkp.gkp_tf_env import gkp_init
from simulator.utils import expectation
from simulator import operators as ops
from simulator.utils import tensor, basis
import gkp.action_script as action_scripts

N=100

# reward_kwargs = dict(reward_mode = 'overlap', 
#                      target_projector = ops.projector(5,N))

# reward_kwargs = dict(reward_mode = 'overlap', 
                     # target_state = qt.basis(N,5)
                     # )

reward_kwargs = dict(reward_mode = 'overlap', 
                      target_projector = tensor([ops.identity(2), ops.projector(5,N)])
                      )


env = gkp_init(simulate='snap_and_displacement', 
                reward_kwargs=reward_kwargs,
                init='vac', H=1, T=5, N=N, attn_step=1, batch_size=1, episode_length=5,
                phase_space_rep='wigner', T1_osc=10e-6)


action_script = 'snap_and_displacements'
action_scale = {'alpha':4, 'theta':pi}
to_learn = {'alpha':True, 'theta':True}

action_script = action_scripts.__getattribute__(action_script)
env = wrappers.ActionWrapper(env, action_script, action_scale, to_learn)

root_dir = r'E:\data\gkp_sims\PPO\examples\Fock_states_sweep\fock5\seed6'
policy_dir = r'policy\002000'
policy = tf.compat.v2.saved_model.load(os.path.join(root_dir,policy_dir))

# from gkp.action_script import Alec_universal_gate_set_6round as action_script
# policy = plc.ScriptedPolicy(env.time_step_spec(), action_script)


# N=40
# target_state = qt.tensor(qt.basis(2,0), qt.basis(N,2))
# # target_state = qt.tensor(qt.basis(2,0),(qt.coherent(N,2)+qt.coherent(N,-2)).unit())
# # target_state = qt.tensor(qt.basis(2,0),(qt.basis(N,0)+qt.basis(N,4)).unit())
# reward_kwargs = {'reward_mode' : 'overlap', 
#                  'target_state' : target_state}
# kwargs = {'N': N}


# env = gkp_init(simulate='Alec_universal_gate_set', 
#                channel='quantum_jumps',
#                reward_kwargs=reward_kwargs,
#                init='vac', H=1, T=6, attn_step=1, batch_size=1, episode_length=6,
#                phase_space_rep='characteristic_fn', **kwargs)

# action_script = 'Alec_universal_gate_set_6round'
# action_scale = {'alpha':1, 'beta':2, 'phi':pi}
# to_learn = {'alpha':True, 'beta':True, 'phi':True}

# action_script = action_scripts.__getattribute__(action_script)
# env = wrappers.ActionWrapper(env, action_script, action_scale, to_learn)

# root_dir = r'E:\data\gkp_sims\PPO\examples\state_prep_universal_gate_set'
# policy_dir = r'policy\000006000'
# policy = tf.compat.v2.saved_model.load(os.path.join(root_dir,policy_dir))


env = gkp_init(simulate='phase_estimation_osc_v2', 
                reward_kwargs={'reward_mode':'fidelity', 'code_flips':True},
                init='X+', H=1, T=4, attn_step=1, batch_size=1, episode_length=10,
                encoding='square')

# from gkp.action_script import v3_Baptiste_autonomous_2round as action_script
from gkp.action_script import v2_phase_estimation_with_trim_4round as action_script
policy = plc.ScriptedPolicy(env.time_step_spec(), action_script)


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

    fig, ax = plt.subplots(1,1)
    fig.suptitle('Mean photon number')
    ax.set_xlabel('Step')
    ax.set_ylabel(r'$\langle \, n \, \rangle$')
    ax.plot(range(len(n)), n)
    print(time_step.reward)
