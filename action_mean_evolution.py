# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 16:40:39 2021

@author: qulab
"""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"


from rl_tools.tf_env import env_init
from rl_tools.tf_env import tf_env_wrappers as wrappers
import tensorflow as tf
from math import pi
import numpy as np
import importlib
import matplotlib.pyplot as plt
from plotting import plot_config

N_epochs = 1200
cavity_phases = []
Kerr_amps = []

root_dir = r'E:\data\gkp_sims\PPO\ECD\EXP_Vlad\sbs_pauli'
exp_name = 'run_56'


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
            'qb_detune':True, 'qb_drag':False}



env = env_init(batch_size=1, **env_kwargs, episode_length=env_kwargs['T'])

action_script_obj = importlib.import_module('rl_tools.action_script.' + action_script)
env = wrappers.ActionWrapper(env, action_script_obj, action_scale, to_learn,
                              learn_residuals=True)

action_names = list(to_learn.keys())
all_actions = {a : [] for a in action_names}

epochs = np.arange(0,N_epochs,2)
for epoch in epochs:
    
    policy_dir = 'policy\\' + str(epoch).zfill(6)
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
    
    for a in action_names:
        all_actions[a].append(actions[a])
        
for a in action_names:
    all_actions[a] = np.array(all_actions[a])
    




fig, axes = plt.subplots(3,3, sharex=True, dpi=200)
axes = axes.ravel()
ax = axes[0]
ax.set_title('Kerr_drive_amp')
ax.plot(epochs, all_actions['Kerr_drive_amp'])


ax = axes[1]
ax.set_title('cavity_phase')
ax.plot(epochs, -all_actions['cavity_phase'][:,0])
ax.plot(epochs, all_actions['cavity_phase'][:,1])


ax = axes[2]
ax.set_title('beta_real')
ax.plot(epochs, all_actions['beta'][:,1,0])

# ax = axes[2]
# ax.set_title('beta_real')
# ax.plot(epochs, all_actions['beta'][:,0,0])
# ax.plot(epochs, all_actions['beta'][:,2,0])

ax = axes[3]
ax.set_title('eps1 & eps2')
ax.plot(epochs, all_actions['beta'][:,0,0], color='red', linestyle='--')
ax.plot(epochs, all_actions['beta'][:,0,1], color='red', label='eps1')

ax.plot(epochs, all_actions['beta'][:,2,0], color='blue', linestyle='--')
ax.plot(epochs, all_actions['beta'][:,2,1], color='blue', label='eps2')

# ax = axes[3]
# ax.set_title('eps')
# ax.plot(epochs, all_actions['beta'][:,1,0], color='red', linestyle='--')
# ax.plot(epochs, all_actions['beta'][:,1,1], color='red')

ax = axes[4]
ax.set_xlabel('Epoch')
ax.set_title('phi_diff')
ax.plot(epochs, all_actions['alpha_correction'][:,1,0])

# ax = axes[4]
# ax.set_xlabel('Epoch')
# ax.set_title('phi_diff')
# ax.plot(epochs, all_actions['alpha_correction'][:,0,0])
# ax.plot(epochs, all_actions['alpha_correction'][:,2,0])


ax = axes[5]
ax.set_xlabel('Epoch')
ax.set_title('phi_sum')
ax.plot(epochs, all_actions['alpha_correction'][:,1,1])

# ax = axes[5]
# ax.set_xlabel('Epoch')
# ax.set_title('phi_sum')
# ax.plot(epochs, all_actions['alpha_correction'][:,0,1])
# ax.plot(epochs, all_actions['alpha_correction'][:,2,1])

ax = axes[6]
ax.set_xlabel('Epoch')
ax.set_title('Detune (MHz)')
ax.plot(epochs, all_actions['qb_detune']*1e-6, linestyle='--')


ax = axes[7]
ax.set_xlabel('Epoch')
ax.set_title('drag')
ax.plot(epochs, all_actions['qb_drag'], linestyle='--')

plt.tight_layout()



#####################################################################
#####################################################################
#####################################################################
## This is useful for plotting the progress in SBS

fig, axes = plt.subplots(2,2, sharex=True, dpi=200, figsize=(3.375, 3))
colors = plt.get_cmap('Dark2')
axes = axes.ravel()
ax = axes[0]
ax.set_ylabel('Kerr pump (DAC amp.)')
ax.plot(epochs, all_actions['Kerr_drive_amp'][:,0], color=colors(0), label='g')
ax.plot(epochs, all_actions['Kerr_drive_amp'][:,1], color=colors(1), label='e')
ax.legend()

ax = axes[1]
ax.set_ylabel('Oscillator phase (radians)')
ax.plot(epochs, all_actions['cavity_phase'][:,0]+pi/2, color=colors(0), label='g')
ax.plot(epochs, all_actions['cavity_phase'][:,1], color=colors(1), label='e')
ax.legend()

ax = axes[2]
ax.set_ylabel('Small CD amplitude')
ax.plot(epochs, all_actions['beta'][:,0,0], color=colors(2), linestyle='--')
ax.plot(epochs, all_actions['beta'][:,0,1], color=colors(2), label=r'$\varepsilon_1$')

ax.plot(epochs, all_actions['beta'][:,2,0], color=colors(3), linestyle='--')
ax.plot(epochs, all_actions['beta'][:,2,1], color=colors(3), label=r'$\varepsilon_2$')
ax.set_xlabel('Epoch')
ax.legend()

ax = axes[3]
ax.set_xlabel('Epoch')
ax.set_ylabel('Qubit detuning (MHz)')
ax.plot(epochs, all_actions['qb_detune']*1e-6, color=colors(6))

plt.tight_layout()


#####################################################################
#####################################################################
#####################################################################
## This is useful for plotting the progress in BSB

fig, axes = plt.subplots(2,2, sharex=True, dpi=200, figsize=(3.375, 3))
colors = plt.get_cmap('Dark2')
axes = axes.ravel()
ax = axes[0]
ax.set_ylabel('Kerr pump (DAC amp.)')
ax.plot(epochs, all_actions['Kerr_drive_amp'][:,0], color=colors(0), label='g')
ax.plot(epochs, all_actions['Kerr_drive_amp'][:,1], color=colors(1), label='e')
ax.legend()

ax = axes[1]
ax.set_ylabel('Oscillator phase (radians)')
ax.plot(epochs, all_actions['cavity_phase'][:,0]+pi/2, color=colors(0), label='g')
ax.plot(epochs, all_actions['cavity_phase'][:,1], color=colors(1), label='e')
ax.legend()

ax = axes[2]
ax.set_ylabel('Small CD amplitude')
ax.plot(epochs, all_actions['beta'][:,1,0], color=colors(3), linestyle='--')
ax.plot(epochs, all_actions['beta'][:,1,1], color=colors(3), label=r'$\varepsilon_1$')

# ax = axes[3]
# ax.set_title('eps')
# ax.plot(epochs, all_actions['beta'][:,1,0], color='red', linestyle='--')
# ax.plot(epochs, all_actions['beta'][:,1,1], color='red')

# ax.plot(epochs, all_actions['beta'][:,2,0], color=colors(3), linestyle='--')
# ax.plot(epochs, all_actions['beta'][:,2,1], color=colors(3), label=r'$\varepsilon_2$')
ax.set_xlabel('Epoch')
# ax.legend()

ax = axes[3]
ax.set_xlabel('Epoch')
ax.set_ylabel('Qubit detuning (MHz)')
ax.plot(epochs, all_actions['qb_detune']*1e-6, color=colors(6))

plt.tight_layout()