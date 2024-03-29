# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:08:16 2021

@author: Vladimir Sivak
"""

# import numpy as np
# import os
# filename = 'GKP_plus_Z_delta_0.306.npz'
# z = np.load(os.path.join(r'E:\data\gkp_sims\PPO\ECD\EXP_Vlad', filename))
# beta = np.stack([z['betas'].real, z['betas'].imag], axis=-1)
# phi = np.stack([z['phis'], z['thetas']], axis=-1)
# savename = os.path.join(r'Z:\tmp\for Vlad\from_vlad', filename)
# np.savez(savename, beta=beta, phi=phi)



import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="1"


from rl_tools.tf_env import env_init
from rl_tools.tf_env import tf_env_wrappers as wrappers
import tensorflow as tf
import rl_tools.action_script as action_scripts
from math import pi
import numpy as np
import importlib


# root_dir = r'E:\data\gkp_sims\PPO\ECD\EXP_Vlad\fock1'
# exp_name = 'run_5'
# policy_str= '000516'

# # Params for environment
# env_kwargs = {
#     'control_circuit' : 'ECD_control_remote',
#     'init' : 'vac',
#     'T' : 5,
#     'N' : 100}

# # Params for action wrapper
# action_script = 'ECD_control_residuals_fock1'
# action_scale = {'beta':0.3, 'phi':0.3}
# to_learn = {'beta':True, 'phi':True}



# root_dir = r'E:\data\gkp_sims\PPO\ECD\EXP_Vlad\GKP_plus_Y'
# exp_name = 'run_5'
# policy_str= '000180'

# # Params for environment
# env_kwargs = {
#     'control_circuit' : 'ECD_control_remote',
#     'init' : 'vac',
#     'T' : 11,
#     'N' : 100}

# # Params for action wrapper
# action_script = 'ECD_control_residuals_GKP_plusY'
# action_scale = {'beta':0.2, 'phi':0.2}
# to_learn = {'beta':True, 'phi':True}



root_dir = r'E:\data\gkp_sims\PPO\ECD\EXP_Vlad\sbs_pauli'
exp_name = 'run_50'
policy_str= '001040'

# Params for environment
env_kwargs = {
    'control_circuit' : 'SBS_remote',
    'init' : 'vac',
    'T' : 1,
    'N' : 20}

# Params for action wrapper
action_script = 'SBS_remote_residuals'
action_scale = {'beta':0.2, 'phi':0.3, 'flip':0.3, 
                'cavity_phase':0.5, 'Kerr_drive_amp':0.5, 'alpha_correction':0.2,
                'qb_detune':3e6, 'qb_drag':4.0}
to_learn = {'beta':True, 'phi':True, 'flip':True, 
            'cavity_phase':True, 'Kerr_drive_amp':True, 'alpha_correction':True,
            'qb_detune':True, 'qb_drag':True}


env = env_init(batch_size=1, **env_kwargs, episode_length=env_kwargs['T'])

action_script_obj = importlib.import_module('rl_tools.action_script.' + action_script)
env = wrappers.ActionWrapper(env, action_script_obj, action_scale, to_learn,
                              learn_residuals=True)

policy_dir = 'policy\\' + policy_str
policy = tf.compat.v2.saved_model.load(os.path.join(root_dir, exp_name, policy_dir))


time_step = env.reset()
policy_state = policy.get_initial_state(env.batch_size)
while not time_step.is_last():
    action_step = policy.action(time_step, policy_state)
    policy_state = action_step.state
    time_step = env.step(action_step.action)


actions = {action_name : np.squeeze(np.array(action_history)[1:])
            for action_name, action_history in env.history.items()
            if not action_name=='msmt'}

savename = action_script + '_' + policy_str + '_' + exp_name + '.npz'
filename = os.path.join(r'Z:\tmp\for Vlad\from_vlad', savename)
np.savez(filename, **actions)