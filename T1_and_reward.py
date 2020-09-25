# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:16:45 2020

@author: Vladimir Sivak
"""

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
from gkp.gkp_tf_env import helper_functions as hf
from gkp.gkp_tf_env import tf_env_wrappers as wrappers
from gkp.gkp_tf_env import policy as plc
from gkp.gkp_tf_env import gkp_init


env = gkp_init(simulate='phase_estimation_osc_v2', 
               channel='quantum_jumps',
               reward_mode='fidelity_with_code_flips',
               init='X+', H=1, T=4, attn_step=1, batch_size=1000, episode_length=100,
               encoding='square')

# from gkp.action_script import phase_estimation_symmetric_with_trim_4round as action_script
# from gkp.action_script import hexagonal_phase_estimation_symmetric_6round as action_script
# to_learn = {'alpha':True, 'beta':True, 'phi':False, 'theta':False}
# env = wrappers.ActionWrapper(env, action_script, to_learn)

# root_dir = r'E:\VladGoogleDrive\Qulab\GKP\sims\PPO\Kerr_sweep_4000\perfect_qubit_without_rotation\K1'
# policy_dir = r'policy\000020000'
# policy = tf.compat.v2.saved_model.load(os.path.join(root_dir,policy_dir))


# from gkp.action_script import v3_Baptiste_autonomous_2round as action_script
from gkp.action_script import v2_phase_estimation_with_trim_4round as action_script
policy = plc.ScriptedPolicy(env.time_step_spec(), action_script)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

fit_params = hf.fit_logical_lifetime(env, policy, plot=True, reps=1, states=['X+','Y+','Z+'])