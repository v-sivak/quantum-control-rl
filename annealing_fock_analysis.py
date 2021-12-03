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

rootdir = r'E:\data\gkp_sims\annealing\8'

sigma = 0.2
N_seeds = 6
N_focks = 10

# how many algo iterations to look at
N_iter_map = {
    'overlap' : 10000,
    'Fock' : 10000}

# step size for evaluation and plotting
step_map = {
    'overlap' : 100,
    'Fock' : 100}

fidelity, best_fidelity = {}, {}
epochs_, best_epochs = {}, {}

for setting in ['overlap', 'Fock']:
    thisdir = os.path.join(rootdir, setting)
    print(thisdir)
    
    N_iter = N_iter_map[setting]
    step = step_map[setting]

    fidelity[setting] = {}
    best_fidelity[setting] = {}
    epochs_[setting] = {}
    best_epochs[setting] = {}
    
    for fock in range(1,N_focks+1):
        
        # these lists will store data for a given Fock but ALL random seeds
        all_seeds_alpha = []
        all_seeds_theta = []
        all_seeds_cost = []

        for seed in range(N_seeds):
            fname = 'fock_' + str(fock) + '_seed_' + str(seed) + '_sigma_' + str(sigma) + '.npz'
            data = np.load(os.path.join(thisdir, fname), allow_pickle=True)
            
            all_seeds_alpha.append(data['alpha'][:N_iter])
            all_seeds_theta.append(data['theta'][:N_iter])
            all_seeds_cost.append(data['cost'][:N_iter])

        all_seeds_alpha = np.array(all_seeds_alpha)
        all_seeds_theta = np.array(all_seeds_theta)
        all_seeds_cost = np.array(all_seeds_cost)
        
        # FIXME: add option to do this once, save the new eval data, and then reuse that
        if setting == 'Fock':
            
            batch_size = 1
            T = 5
            N = 100
            
            target_state = qt.basis(N, fock)
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
            
            all_seeds_cost =  np.zeros([6,N_iter])
            
            # do evaluation of every saved policy to find its fidelity
            for seed in range(N_seeds):
                for i in range(N_iter):
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
        
                    all_seeds_cost[seed, i] = -avg_reward(action_script)
                
                # save the new evaluation data to npz file
                fname = 'fock_' + str(fock) + '_seed_' + str(seed) + '_sigma_' + str(sigma) + '.npz'
                data = np.load(os.path.join(thisdir, fname), allow_pickle=True)
                data_cp = dict(data)
                data_cp['cost_eval'] = all_seeds_cost[seed]
                np.savez(fname, **data_cp)

        # pick the best fidelity in intervals of length <step>
        selected_fidelities = []
        selected_epochs = []
        for i in range(int(np.floor(N_iter/step))):
            selected_fidelities.append(-all_seeds_cost[:, i*step : (i+1)*step].min(axis=1))
            selected_epochs.append(i*step+all_seeds_cost[:, i*step : (i+1)*step].argmin(axis=1))        
        selected_fidelities = np.array(selected_fidelities).transpose()
        selected_epochs = np.array(selected_epochs).transpose()
        
        fidelity[setting][fock] = selected_fidelities
        epochs_[setting][fock] = selected_epochs
        
        print('Fock %d, %.5f' %(fock, np.max(fidelity[setting][fock])))

        # find the best performing random seed
        ind_best = np.argmax(fidelity[setting][fock][:,-1])
        best_fidelity[setting][fock] = fidelity[setting][fock][ind_best]
        best_epochs[setting][fock] = epochs_[setting][fock][ind_best]        





fig, axes = plt.subplots(1,2, figsize=(3.375, 2.5), dpi=300)
palette = plt.get_cmap('tab10')
from plotting import plot_config

for setting in ['overlap', 'Fock']:
    ax = axes[0] if setting == 'overlap' else axes[1]
    ax.grid(True)
    ax.set_yscale('log')
    
    ax.set_xlim(-50, 4000+50)
    ax.set_ylim(3e-4,1.3)
    
    if setting == 'overlap':
        ax.set_ylabel(r'$1-\cal F$')
        ax.set_xlabel(r'Iteration')
    
    if setting == 'Fock':
        ax.set_yticks([1,0.1,0.01,0.001])
        ax.set_yticklabels([])
    
    N_iter = N_iter_map[setting]
    step = step_map[setting]

    for fock in range(1,N_focks+1):
        for seed in range(N_seeds):        
            ax.plot(epochs_[setting][fock][seed], 1-fidelity[setting][fock][seed], 
                    linestyle='--', alpha=0.25, color=palette(fock-1))

        ax.plot(best_epochs[setting][fock], 1-best_fidelity[setting][fock], 
                linestyle='-', color=palette(fock-1))    
    
plt.tight_layout()
    
savename = r'E:\VladGoogleDrive\Qulab\GKP\paper\figs\simulated_annealing'
fig.savefig(savename, fmt='.pdf')
