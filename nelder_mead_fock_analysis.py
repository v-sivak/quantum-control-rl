# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 13:22:41 2021

@author: qulab
"""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import numpy as np
import matplotlib.pyplot as plt
from plotting import plot_config
import qutip as qt
from rl_tools.tf_env import env_init
from rl_tools.tf_env import policy as plc

class ActionScript():
    pass

rootdir = r'E:\data\gkp_sims\NM'


fidelity = {}
best_fidelity = {}


for setting in ['overlap', 'Fock2']:
    thisdir = os.path.join(rootdir, setting)

    if setting == 'overlap':        
        N_iter = 4000
        step = 50
    elif setting == 'Fock2':
        N_iter = 2000
        step = 50

    fidelity[setting] = {}
    best_fidelity[setting] = {}
    
    for fock in range(1,11):
        
        all_seeds_alpha = []
        all_seeds_theta = []
        all_seeds_cost = []
        
        epochs = np.arange(0,N_iter,step+1)
        epochs[-1] = N_iter-1
        
        for seed in range(6):
            sigma = 0.2
            fname = 'fock_' + str(fock) + '_seed_' + str(seed) + '_sigma_' + str(sigma) + '.npz'
            data = np.load(os.path.join(thisdir, fname), allow_pickle=True)
            
            if len(data['alpha']) < N_iter:
                data_alpha= np.concatenate([data['alpha'], np.array([data['alpha'][-1]]*(N_iter-len(data['alpha'])))])
                data_theta = np.concatenate([data['theta'], np.array([data['theta'][-1]]*(N_iter-len(data['theta'])))])
                data_cost = np.concatenate([data['cost'], np.array([data['cost'][-1]]*(N_iter-len(data['cost'])))])
            
                all_seeds_alpha.append(data_alpha[:N_iter][epochs])
                all_seeds_theta.append(data_theta[:N_iter][epochs])
                all_seeds_cost.append(data_cost[:N_iter][epochs])    
            
            else:
                all_seeds_alpha.append(data['alpha'][:N_iter][epochs])
                all_seeds_theta.append(data['theta'][:N_iter][epochs])
                all_seeds_cost.append(data['cost'][:N_iter][epochs])
            
        
        all_seeds_alpha = np.array(all_seeds_alpha)
        all_seeds_theta = np.array(all_seeds_theta)
        all_seeds_cost = np.array(all_seeds_cost)
    
        batch_size = 1
        T = 5
        N = 100
        
        target_state = qt.basis(N,fock)
        reward_kwargs = {'reward_mode' : 'overlap',
                         'target_state' : target_state,
                         'postselect_0' : False}
        
        env = env_init(control_circuit='snap_and_displacement', reward_kwargs=reward_kwargs,
                       init='vac', T=T, batch_size=batch_size, N=N, episode_length=T)
        
        def avg_reward(action_script):
        
            policy = plc.ScriptedPolicy(env.time_step_spec(), action_script)
            
            time_step = env.reset()
            policy_state = policy.get_initial_state(env.batch_size)
            while not time_step.is_last()[0]:
                action_step = policy.action(time_step, policy_state)
                policy_state = action_step.state
                time_step = env.step(action_step.action)
            
            return time_step.reward.numpy().mean()
        
        fidelity[setting][fock] = np.zeros([6,len(epochs)])
        
        for seed in range(6):
            for i, e in enumerate(epochs):
                action_script = ActionScript()
                action_script.period = T
                action_script.script = {}
                action_script.script['alpha'] = all_seeds_alpha[seed, i]
                action_script.script['theta'] = all_seeds_theta[seed, i]
    
                policy = plc.ScriptedPolicy(env.time_step_spec(), action_script)
                time_step = env.reset()
                policy_state = policy.get_initial_state(env.batch_size)
                while not time_step.is_last()[0]:
                    action_step = policy.action(time_step, policy_state)
                    policy_state = action_step.state
                    time_step = env.step(action_step.action)            
    
                fidelity[setting][fock][seed, i] = avg_reward(action_script)
                
    
        ind_best = np.argmax(fidelity[setting][fock][:,-1])
        best_fidelity[setting][fock] = fidelity[setting][fock][ind_best]    
    
    


fig = plt.figure(figsize=(3.95, 3), dpi=200)
from matplotlib import gridspec
palette = plt.get_cmap('tab10')

# fig, axes = plt.subplots(1,2, figsize=(3.375, 3), dpi=200, sharey=True)

spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[2, 1])
axes = [fig.add_subplot(spec[0]), fig.add_subplot(spec[1])]

for setting in ['overlap', 'Fock2']:

    
    ax = axes[0] if setting == 'overlap' else axes[1]
    ax.grid(True)
    axes[1].grid(True)
    
    
    
    ax.set_yscale('log')
    
    if setting == 'overlap':
        ax.set_ylabel(r'$1-\cal F$')
        ax.set_xlabel(r'Iteration')
    
    if setting == 'Fock2':
        ax.set_yticks([1,0.1,0.01,0.001])
        ax.set_yticklabels([])
        ax.set_xticks([0,1000,2000])
    

    if setting == 'overlap':        
        N_iter = 4000
        step = 50
    elif setting == 'Fock2':
        N_iter = 2000
        step = 50

    epochs = np.arange(0,N_iter,step+1)
    epochs[-1] = N_iter-1

    for fock in range(1,11):
        
        for seed in range(6):        
            ax.plot(epochs, 1-fidelity[setting][fock][seed], linestyle='--', alpha=0.25, color=palette(fock-1))
    
        
        ax.plot(epochs, 1-best_fidelity[setting][fock], linestyle='-', color=palette(fock-1))
    
        ax.set_xlim(-50,N_iter+50)
        ax.set_ylim(3e-4,1.3)

savename = r'E:\VladGoogleDrive\Qulab\GKP\paper\figs\NM_optimization.pdf'
# fig.savefig(savename)
    