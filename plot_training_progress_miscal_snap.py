# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 11:06:07 2021

@author: qulab
"""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import matplotlib.pyplot as plt
import h5py
import plotting.plot_config

root_dir = {
    'ideal' : r'E:\data\gkp_sims\PPO\paper_data\miscalibrated_snap_S7\ideal_overlap',
    't=3.4' : r'E:\data\gkp_sims\PPO\paper_data\miscalibrated_snap_S7\miscal_t=3.4',
    't=0.4' : r'E:\data\gkp_sims\PPO\paper_data\miscalibrated_snap_S7\miscal_t=0.4',
    }

colors = {
    'ideal' : plt.get_cmap('tab10')(0),
    't=3.4' : 'darkgreen',
    't=0.4' : 'olivedrab',
    }

log = {k:{} for k in root_dir.keys()}

# Pull out the training results for all settings and all random seeds
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


"""In a separate figure, take the protocols trained with overlap reward for 
    ideal SNAP, and apply them in the "finite-duration-SNAP" environment to 
    observe the performance degradation. Plot both the average degradation and
    post-selected to 11111 verification measurement history."""


figname = r'E:\VladGoogleDrive\Qulab\GKP\paper\figs\snap_with_feedback.pdf'
fig, ax = plt.subplots(1,1, figsize=(3, 2), dpi=300)
plt.grid(True)
ax.set_ylabel(r'$1-\cal F$')
ax.set_xlabel('Epoch')
ax.set_ylim(1e-4,1.5)
ax.set_xlim(1e2,3e4)
ax.set_yscale('log')
ax.set_xscale('log')
palette = plt.get_cmap('tab10')


# PLOT RESULTS FOR ORIGINAL PROTOCOLS
for i, protocol in enumerate(log.keys()):
    # in the background, plot all random seeds
    for sim_name in log[protocol].keys():
        epochs = log[protocol][sim_name]['epochs']
        if protocol != 'ideal':
            ind1 = np.arange(len(epochs)/8).astype(int)
            ind2 = np.arange(len(epochs)/8, len(epochs), 4).astype(int)
            ind = np.concatenate([ind1,ind2])
        else:
            ind1 = np.arange(len(epochs)/4).astype(int)
            ind2 = np.arange(len(epochs)/4, len(epochs), 4).astype(int)
            ind = np.concatenate([ind1,ind2])
        ax.plot(epochs[ind], 1-log[protocol][sim_name]['returns'][ind],
                linestyle='--', color=colors[protocol], alpha=0.3)
    
    all_seeds = np.array(list(log[protocol][i]['returns'] for i in log[protocol].keys()))    
    # # calculate mean log infidelity
    # log_infidelity = np.mean(np.log10(1-all_seeds), axis=0)
    # infidelity = 10 ** log_infidelity    

    # select the best training trajectory
    inds = np.argmax(all_seeds[:,-1])
    infidelity = 1 - all_seeds[inds,:]
    print(protocol + ', best seed %d'%inds)
    
    ax.plot(epochs[ind], infidelity[ind], color=colors[protocol], linewidth=1.0, label=protocol)


# To show model bias, take the protocols trained with ideal SNAP and test
# them with finite-duration SNAP.
import tensorflow as tf
import qutip as qt
from rl_tools.tf_env import env_init
from math import pi
import rl_tools.action_script as action_scripts
from rl_tools.tf_env import tf_env_wrappers as wrappers

env_kwargs = dict(control_circuit='snap_and_displacement_miscalibrated', init='vac',
                  H=1, T=5, attn_step=1, N=50, batch_size=1000, episode_length=5)

target_state = qt.tensor(qt.basis(2,0), qt.basis(50,3))
reward_kwargs = {'reward_mode' : 'overlap',
                  'target_state' : target_state,
                  'postselect_0' : False}

action_script = 'snap_and_displacements'
action_scale = {'alpha':4, 'theta':pi}
to_learn = {'alpha':True, 'theta':True}

action_script = action_scripts.__getattribute__(action_script)

protocol = 'ideal'
max_epochs = 3000
gate_times = [0.4e-6, 3.4e-6]
seeds = ['seed2']
rewards = {t:{} for t in gate_times}
norms = {t:{} for t in gate_times}

for t in gate_times:
    env = env_init(**env_kwargs, reward_kwargs=reward_kwargs)
    env = wrappers.ActionWrapper(env, action_script, action_scale, to_learn)
    env._env.SNAP_miscalibrated.T = t
    env._env.bit_string = None # '00000'
    # collect episodes with different policies
    for sim_name in seeds: #os.listdir(root_dir[protocol]):
        print(sim_name)
        rewards[t][sim_name] = []
        norms[t][sim_name] = []
        sim_dir = os.path.join(root_dir[protocol], sim_name)
        
        for policy_name in os.listdir(os.path.join(sim_dir, 'policy')):
            if int(policy_name) > max_epochs: break
            policy_dir = os.path.join(sim_dir, 'policy', policy_name)
            policy = tf.compat.v2.saved_model.load(policy_dir)
            
            time_step = env.reset()
            env._env.norms = []
            policy_state = policy.get_initial_state(env.batch_size)
            counter = 0
            while not time_step.is_last()[0]:
                counter += 1
                action_step = policy.action(time_step, policy_state)
                policy_state = action_step.state
                time_step = env.step(action_step.action)
            rewards[t][sim_name].append(np.mean(time_step.reward))
            norms[t][sim_name].append(np.squeeze(env.norms).astype(float))

# PLOT RESULTS FOR DIFFERENT GATE TIMES
protocol = 'ideal'
reds = ['lightsalmon', 'tomato']
colors_t = {t: reds[i] for i,t in enumerate(gate_times)}
for t in gate_times:
    for sim_name in rewards[t].keys():
        epochs = log[protocol][sim_name]['epochs']
        epochs = epochs[np.where(epochs<max_epochs+1)]
        ax.plot(epochs, 1-np.array(rewards[t][sim_name]),
                linestyle='--', color=colors_t[t], alpha=0.3)
    # calculate mean log infidelity
    all_seeds = np.array(list(np.array(rewards[t][i]) for i in rewards[t].keys()))
    log_infidelity = np.mean(np.log10(1-all_seeds), axis=0)
    infidelity = 10 ** log_infidelity
    ax.plot(epochs, infidelity, color=colors_t[t], linewidth=1.0)



ax.legend()
fig.tight_layout()
fig.savefig(figname)