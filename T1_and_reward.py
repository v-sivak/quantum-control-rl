# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:16:45 2020

@author: Vladimir Sivak
"""

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# import logging
# logging.disable(logging.WARNING)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from numpy import pi
import tensorflow as tf
from gkp.gkp_tf_env import helper_functions as hf
from gkp.gkp_tf_env import tf_env_wrappers as wrappers
from gkp.gkp_tf_env import policy as plc
from gkp.gkp_tf_env import gkp_init


env = gkp_init(simulate='phase_estimation_osc_v2', 
                reward_kwargs={'reward_mode':'fidelity', 'code_flips':True},
                init='X+', H=1, T=4, attn_step=1, batch_size=1000, episode_length=50,
                encoding='square')

# from gkp.action_script import v2_phase_estimation_with_trim_4round as action_script
# action_scale = {'alpha':1, 'beta':1, 'phi':pi, 'theta':0.02}
# to_learn = {'alpha':True, 'beta':True, 'phi':False, 'theta':False}
# env = wrappers.ActionWrapper(env, action_script, action_scale, to_learn)

# root_dir = r'E:\data\gkp_sims\PPO\examples\gkp_square_qec_ST_B100_lr3e-4\0'
# policy_dir = r'policy\000040000'
# policy = tf.compat.v2.saved_model.load(os.path.join(root_dir,policy_dir))


# env = gkp_init(simulate='gkp_qec_autonomous_BsB_osc', 
#                 channel='quantum_jumps',
#                 reward_kwargs={'reward_mode':'fidelity', 'code_flips':False},
#                 init='X+', H=1, T=2, attn_step=1, batch_size=1000, episode_length=200,
#                 encoding='square')

# from gkp.action_script import v3_Baptiste_autonomous_2round as action_script
# action_scale = {'beta':1, 'phi':pi, 'epsilon':1}
# to_learn = {'beta':False, 'phi':True, 'epsilon':True}
# env = wrappers.ActionWrapper(env, action_script, action_scale, to_learn)

# root_dir = r'E:\data\gkp_sims\PPO\examples\gkp_square_qec_BsB_B100_lr3e-4\2'
# policy_dir = r'policy\000020000'
# policy = tf.compat.v2.saved_model.load(os.path.join(root_dir,policy_dir))


# from gkp.action_script import v3_Baptiste_autonomous_2round as action_script
from gkp.action_script import v2_phase_estimation_with_trim_4round as action_script
policy = plc.ScriptedPolicy(env.time_step_spec(), action_script)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

fit_params = hf.fit_logical_lifetime(env, policy, plot=True, reps=1, states=['X+','Y+','Z+'])