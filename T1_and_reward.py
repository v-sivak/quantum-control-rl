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
from gkp.gkp_tf_env import helper_functions as hf
from gkp.gkp_tf_env import tf_env_wrappers as wrappers
from gkp.gkp_tf_env import policy as plc
from gkp.gkp_tf_env import gkp_init


env = gkp_init(simulate='oscillator', 
                init='X+', H=1, T=6, attn_step=1, batch_size=3000, episode_length=200, 
                reward_mode = 'zero', quantum_circuit_type='v2',
                encoding='hexagonal')

# from gkp.action_script import phase_estimation_symmetric_with_trim_4round as action_script
from gkp.action_script import hexagonal_phase_estimation_symmetric_6round as action_script
to_learn = {'alpha':True, 'beta':True, 'phi':False, 'theta':False}
env = wrappers.ActionWrapper(env, action_script, to_learn)

root_dir = r'E:\VladGoogleDrive\Qulab\GKP\sims\PPO\Kerr_sweep_4000\perfect_qubit_without_rotation\K1'
policy_dir = r'policy\000020000'
policy = tf.compat.v2.saved_model.load(os.path.join(root_dir,policy_dir))


# from gkp.action_script import hexagonal_phase_estimation_symmetric_6round as action_script
# from gkp.action_script import phase_estimation_symmetric_with_trim_4round as action_script
# policy = plc.ScriptedPolicy(env.time_step_spec(), action_script)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

fit_params = hf.fit_logical_lifetime(env, policy, plot=True, reps=1, states=['X+'])