# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:16:45 2020

@author: Vladimir Sivak
"""

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
from math import pi
from rl_tools.tf_env import helper_functions as hf
from rl_tools.tf_env import tf_env_wrappers as wrappers
import rl_tools.action_script as action_scripts
from rl_tools.tf_env import policy as plc
from rl_tools.tf_env import env_init

# env = env_init(control_circuit='phase_estimation_osc_qb_v2', 
#                 reward_kwargs={'reward_mode':'fidelity', 'code_flips':True},
#                 init='X+', H=1, T=4, attn_step=1, batch_size=1000, episode_length=100,
#                 encoding='square')

env = env_init(control_circuit='gkp_qec_autonomous_sBs_osc_qb', 
                reward_kwargs={'reward_mode':'fidelity', 'code_flips':True},
                init='X+', H=1, T=2, attn_step=1, batch_size=500, episode_length=60,
                encoding='square')

# action_script = 'gkp_qec_autonomous_BsB_2round'
# action_scale = {'beta':1, 'phi':pi, 'epsilon':1}
# to_learn = {'beta':False, 'phi':False, 'epsilon':True}
# action_script = action_scripts.__getattribute__(action_script)
# env = wrappers.ActionWrapper(env, action_script, action_scale, to_learn)

# root_dir = r'E:\data\gkp_sims\PPO\examples\gkp_qec_autonomous_BsB'
# policy_dir = r'policy\000200'
# policy = tf.compat.v2.saved_model.load(os.path.join(root_dir,policy_dir))



# action_script = 'gkp_qec_autonomous_sBs_2round'
# action_scale = {'beta':1, 'phi':pi, 'eps1':1, 'eps2':1}
# to_learn = {'beta':False, 'phi':False, 'eps1':True, 'eps2':True}
# action_script = action_scripts.__getattribute__(action_script)
# env = wrappers.ActionWrapper(env, action_script, action_scale, to_learn)

# root_dir = r'E:\data\gkp_sims\PPO\examples\test'
# policy_dir = r'policy\000300'
# policy = tf.compat.v2.saved_model.load(os.path.join(root_dir,policy_dir))


# from rl_tools.action_script import gkp_qec_autonomous_BsB_2round as action_script
from rl_tools.action_script import gkp_qec_autonomous_sBs_2round as action_script
# from rl_tools.action_script import v2_phase_estimation_with_trim_4round as action_script
policy = plc.ScriptedPolicy(env.time_step_spec(), action_script)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

fit_params = hf.fit_logical_lifetime(env, policy, plot=True, reps=1, states=['X+','Y+','Z+'])