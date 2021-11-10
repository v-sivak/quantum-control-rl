# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 15:14:30 2020

@author: Vladimir Sivak
"""

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# append parent 'gkp-rl' directory to path 
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import qutip as qt
import tensorflow as tf
import numpy as np
from math import sqrt, pi
from rl_tools.agents import PPO
from tf_agents.networks import actor_distribution_network
from rl_tools.agents import actor_distribution_network_gkp
from rl_tools.tf_env import helper_functions as hf

"""
Train PPO agent to optimize a sqrt(H) gate on a GKP logical qubit, realized
with universal gate set consisting of SNAPs and displacements.

All training episodes start from one of the encoded states chosen at random, 
|+/-X>, |+/-Y>, or |+/-Z>, and Wigner reward is applied to the final state.

"""

root_dir = r'E:\data\gkp_sims\PPO\paper_data\sqrtH_gate_gkp'
if not os.path.isdir(root_dir): os.mkdir(root_dir)

random_seeds = [0,1,2,3,4,5]

for seed in random_seeds:
    sim_dir = os.path.join(root_dir,'seed'+str(seed))
    # Params for environment
    env_kwargs = {
        'control_circuit' : 'snap_and_displacement',
        'encoding' : 'gkp_square',
        'init' : 'random_v2',
        'T' : 1, 
        'N' : 120}

    env_kwargs_eval = {
        'control_circuit' : 'snap_and_displacement',
        'encoding' : 'gkp_square',
        'init' : 'cardinal_points',
        'T' : 1, 
        'N' : 120}
    
    # this is sqrt(H) gate (non-Clifford), should be doable with 1 SNAP in GKP
    gate_matrix = [[1/4*((1-1j)*sqrt(2)+2*(1+1j)), 1/4*sqrt(2)*(1-1j)],
                   [1/4*sqrt(2)*(1-1j), 1/4*((-1+1j)*sqrt(2)+2*(1+1j))]]


    # Params for reward function
    reward_kwargs = {
        'reward_mode' : 'gate',
        'gate_matrix' : gate_matrix,
        'tomography' : 'wigner',
        'window_size' : 12,
        'N_alpha' : 1,
        'N_msmt' : 1,
        'sampling_type' : 'abs'
        }

    reward_kwargs_eval = {
        'reward_mode' : 'gate_fidelity',
        'gate_matrix' : gate_matrix
        }
    
    # Params for action wrapper
    action_script = 'snap_and_displacements'
    action_scale = {'alpha':4, 'theta':pi}
    to_learn = {'alpha':True, 'theta':True}
    
    train_batch_size = 500
    eval_batch_size = 6

    train_episode_length = lambda x: env_kwargs['T']
    eval_episode_length = lambda x: env_kwargs['T']
    
    # Create drivers for data collection
    from rl_tools.agents import dynamic_episode_driver_sim_env
    
    collect_driver = dynamic_episode_driver_sim_env.DynamicEpisodeDriverSimEnv(
        env_kwargs, reward_kwargs, train_batch_size, 
        action_script, action_scale, to_learn, train_episode_length)
    
    eval_driver = dynamic_episode_driver_sim_env.DynamicEpisodeDriverSimEnv(
        env_kwargs_eval, reward_kwargs_eval, eval_batch_size, 
        action_script, action_scale, to_learn, eval_episode_length)
    
    PPO.train_eval(
            root_dir = sim_dir,
            random_seed = seed,
            num_epochs = 10000,
            # Params for train
            normalize_observations = True,
            normalize_rewards = False,
            discount_factor = 1.0,
            lr = 1e-3,
            lr_schedule = None, #lambda x: 1e-3 if x<1000 else 1e-4,
            num_policy_updates = 20,
            initial_adaptive_kl_beta = 0.0,
            kl_cutoff_factor = 0,
            importance_ratio_clipping = 0.1,
            value_pred_loss_coef = 0.005,
            gradient_clipping = 1.0,
            # Params for log, eval, save
            eval_interval = 50,
            save_interval = 50,
            checkpoint_interval = 100000,
            summary_interval = 10,
            # Params for data collection
            train_batch_size = train_batch_size,
            eval_batch_size = eval_batch_size,
            collect_driver = collect_driver,
            eval_driver = eval_driver,
            replay_buffer_capacity = 7000,
            # Policy and value networks
            ActorNet = actor_distribution_network_gkp.ActorDistributionNetworkGKP,
            actor_fc_layers = (50,),
            value_fc_layers = (50,),
            use_rnn = True,
            actor_lstm_size = (12,),
            value_lstm_size = (12,)
            )
    
    print('Finished: random seed %d' %(seed))