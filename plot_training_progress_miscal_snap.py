# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 11:06:07 2021

@author: qulab
"""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="1"


import numpy as np
import matplotlib.pyplot as plt
import h5py
import plotting.plot_config

# naming convention is (num points, num averages)

root_dir = {
    'ideal' : r'E:\data\gkp_sims\PPO\examples\SNAP_with_feedback_comparison_v3\ideal',
    # 'w_feedback_angle_phase' : r'E:\data\gkp_sims\PPO\examples\SNAP_with_feedback_comparison_v3\miscal_w_msmts_w_feedback_angle_phase',
    't=3.4' : r'E:\data\gkp_sims\PPO\examples\SNAP_with_feedback_comparison_v4\t=3.40',
    't=0.4' : r'E:\data\gkp_sims\PPO\examples\SNAP_with_feedback_comparison_v4\t=0.40',
    }

batch_size = 1000

colors = {
    'ideal' : 'black',
    't=3.4' : 'darkgreen',
    't=0.4' : 'olivedrab',
    # 'w_feedback_angle_phase' : plt.get_cmap('tab20c')(17),
    }

log = {k:{} for k in root_dir.keys()}

for protocol in root_dir.keys():
    for sim_name in os.listdir(root_dir[protocol]):
        fname = os.path.join(root_dir[protocol], sim_name, 'log.hdf5')
        h5file = h5py.File(fname,'r+')
        grpname = list(h5file.keys())[-1] # pick the latest log
        log[protocol][sim_name] = {}
        try:
            grp = h5file[grpname]
            log[protocol][sim_name]['epochs'] = np.array(grp.get('epochs'))
            log[protocol][sim_name]['returns'] = np.array(grp.get('returns'))
        finally:
            h5file.close()

# figname = r'somewhere' # where to save
fig, ax = plt.subplots(1,1, figsize=(3.375, 2), dpi=300)
ax.set_ylabel(r'$1-\cal F$')
ax.set_xlabel('Epoch')
ax.set_ylim(8e-4,1.5)
# ax.set_xlim(1.5e2,6e3)
ax.set_xlim(-100,5e3+100)
ax.set_yscale('log')
# ax.set_xscale('log')
plt.grid(True, color='grey', linestyle='--', linewidth=0.4)
palette = plt.get_cmap('tab10')

for i, protocol in enumerate(log.keys()):
    # in the background, plot all random seeds
    for sim_name in log[protocol].keys():
        epochs = log[protocol][sim_name]['epochs']
        ax.plot(epochs, 1-log[protocol][sim_name]['returns'],
                linestyle='--', color=colors[protocol], alpha=0.3)
    # calculate mean log infidelity
    all_seeds = np.array(list(log[protocol][i]['returns'] for i in log[protocol].keys()))
    log_infidelity = np.mean(np.log10(1-all_seeds), axis=0)
    infidelity = 10 ** log_infidelity    
    ax.plot(epochs, infidelity, color=colors[protocol], linewidth=1.0, label=protocol)
fig.tight_layout()








env_kwargs = dict(simulate='snap_and_displacement_miscalibrated', init='vac',
                  H=1, T=5, attn_step=1, N=50, batch_size=100, episode_length=5)


import tensorflow as tf
import qutip as qt
from gkp.gkp_tf_env import gkp_init
from math import pi
import gkp.action_script as action_scripts
from gkp.gkp_tf_env import tf_env_wrappers as wrappers

target_state = qt.tensor(qt.basis(2,0), qt.basis(50,3))
reward_kwargs = {'reward_mode' : 'overlap',
                  'target_state' : target_state}

action_script = 'snap_and_displacements'
action_scale = {'alpha':4, 'theta':pi}
to_learn = {'alpha':True, 'theta':True}

action_script = action_scripts.__getattribute__(action_script)

protocol = 'ideal'
gate_times = [0.4e-6, 3.4e-6]
rewards = {t:{} for t in gate_times}

for t in gate_times:
    env = gkp_init(**env_kwargs, reward_kwargs=reward_kwargs)
    env = wrappers.ActionWrapper(env, action_script, action_scale, to_learn)
    env._env.SNAP_miscalibrated.T = t
    # collect episodes with different policies
    for sim_name in os.listdir(root_dir[protocol]):
        print(sim_name)
        rewards[t][sim_name] = []
        sim_dir = os.path.join(root_dir[protocol], sim_name)
        
        for policy_name in os.listdir(os.path.join(sim_dir, 'policy')):
            policy_dir = os.path.join(sim_dir, 'policy', policy_name)
            policy = tf.compat.v2.saved_model.load(policy_dir)
            
            time_step = env.reset()
            policy_state = policy.get_initial_state(env.batch_size)
            counter = 0
            while not time_step.is_last()[0]:
                counter += 1
                action_step = policy.action(time_step, policy_state)
                policy_state = action_step.state
                time_step = env.step(action_step.action)
            rewards[t][sim_name].append(np.mean(time_step.reward))



# figname = r'somewhere' # where to save
fig, ax = plt.subplots(1,1, figsize=(3.375, 2), dpi=300)
ax.set_ylabel(r'$1-\cal F$')
ax.set_xlabel('Epoch')
ax.set_ylim(8e-4,1.5)
ax.set_xlim(1.0e2,6e3)
# ax.set_xlim(-100,5e3+100)
ax.set_yscale('log')
ax.set_xscale('log')
plt.grid(True, color='grey', linestyle='--', linewidth=0.4)
palette = plt.get_cmap('tab10')

# FIRST PLOT RESULTS FOR ORIGINAL PROTOCOL
# in the background, plot all random seeds
protocol = 'ideal'
# NOW PLOT RESULTS FOR DIFFERENT GATE TIMES
reds = ['lightsalmon', 'tomato']
colors_t = {t: reds[i] for i,t in enumerate(gate_times)}
for t in gate_times:
    for sim_name in rewards[t].keys():
        epochs = log[protocol][sim_name]['epochs']
        ax.plot(epochs, 1-np.array(rewards[t][sim_name]),
                linestyle='--', color=colors_t[t], alpha=0.3)
    # calculate mean log infidelity
    all_seeds = np.array(list(np.array(rewards[t][i]) for i in rewards[t].keys()))
    log_infidelity = np.mean(np.log10(1-all_seeds), axis=0)
    infidelity = 10 ** log_infidelity
    ax.plot(epochs, infidelity, color=colors_t[t], linewidth=1.0)

for i, protocol in enumerate(log.keys()):
    # in the background, plot all random seeds
    for sim_name in log[protocol].keys():
        epochs = log[protocol][sim_name]['epochs']
        ax.plot(epochs, 1-log[protocol][sim_name]['returns'],
                linestyle='--', color=colors[protocol], alpha=0.3)
    # calculate mean log infidelity
    all_seeds = np.array(list(log[protocol][i]['returns'] for i in log[protocol].keys()))
    log_infidelity = np.mean(np.log10(1-all_seeds), axis=0)
    infidelity = 10 ** log_infidelity    
    ax.plot(epochs, infidelity, color=colors[protocol], linewidth=1.0, label=protocol)
# ax.legend()
fig.tight_layout()