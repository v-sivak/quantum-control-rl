# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:08:16 2021

@author: Vladimir Sivak
"""

# import numpy as np
# import os
# filename = 'fock_9.npz'
# z = np.load(os.path.join(r'C:\Users\qulab\Downloads', filename))
# beta = np.stack([z['betas'].real, z['betas'].imag], axis=-1)
# phi = np.stack([z['phis'], z['thetas']], axis=-1)
# savename = os.path.join(r'Z:\tmp\for Vlad\from_vlad', filename)
# np.savez(savename, beta=beta, phi=phi)



import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"


from rl_tools.tf_env import env_init
from rl_tools.tf_env import tf_env_wrappers as wrappers
import tensorflow as tf
import rl_tools.action_script as action_scripts
from math import pi
import numpy as np
import importlib

root_dir = r'E:\data\gkp_sims\PPO\ECD\EXP_Vlad\sbs_stabilizers\run_15'

# Params for environment
env_kwargs = {
    'control_circuit' : 'SBS_remote',
    'init' : 'vac',
    'T' : 1,
    'N' : 20}

# Params for action wrapper
action_script = 'SBS_remote_residuals'
action_scale = {'beta':0.3, 'phi':0.4, 'flip':0.4, 'detune':5e6}
to_learn = {'beta':True, 'phi':True, 'flip':True, 'detune':True}



# Evaluate some of the protocols at after the training is finished
policy_str= '000402'


env = env_init(batch_size=1, **env_kwargs, episode_length=env_kwargs['T'])

action_script_obj = importlib.import_module('rl_tools.action_script.' + action_script)
env = wrappers.ActionWrapper(env, action_script_obj, action_scale, to_learn,
                              learn_residuals=True)

policy_dir = 'policy\\' + policy_str
policy = tf.compat.v2.saved_model.load(os.path.join(root_dir,policy_dir))


time_step = env.reset()
policy_state = policy.get_initial_state(env.batch_size)
while not time_step.is_last():
    action_step = policy.action(time_step, policy_state)
    policy_state = action_step.state
    time_step = env.step(action_step.action)


actions = {action_name : np.squeeze(np.array(action_history)[1:])
            for action_name, action_history in env.history.items()
            if not action_name=='msmt'}

filename = os.path.join(r'Z:\tmp\for Vlad\from_vlad', policy_str+'_sbs_run15.npz')
np.savez(filename, **actions)