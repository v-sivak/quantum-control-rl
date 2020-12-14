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
from gkp.agents import PPO
from tf_agents.networks import actor_distribution_network
from gkp.agents import actor_distribution_network_gkp
from gkp.gkp_tf_env import helper_functions as hf

"""
Train PPO agent to do Fock state N=1...10 preparation with universal gate 
sequence consisting of SNAP gates and oscillator displacements.

The episodes start from vacuum, and photon number measurements are performed 
in the end to assign reward.

"""

root_dir = r'E:\data\gkp_sims\PPO\examples\Fock_states_sweep___test'
states = [1,2,3,4,5,6,7,8,9,10]
random_seeds = [0,1,2,3,4]

for fock in states:
    # setup directory for each simulation
    fock_dir = os.path.join(root_dir,'fock'+str(fock))
    if not os.path.isdir(fock_dir): os.mkdir(fock_dir)

    for seed in random_seeds:
        sim_dir = os.path.join(fock_dir,'seed'+str(seed))
        # Params for environment
        env_kwargs = {
            'simulate' : 'snap_and_displacement_angle_phase',
            'init' : 'vac',
            'H' : 1,
            'T' : 5, 
            'attn_step' : 1,
            'N' : 100}
        
        # Params for reward function
        target_state = qt.tensor(qt.basis(2,0), qt.basis(100,fock))
        reward_kwargs = {'reward_mode' : 'Fock',
                         'target_state' : target_state}

        reward_kwargs_eval = {'reward_mode' : 'overlap',
                              'target_state' : target_state}
        
        # Params for action wrapper
        action_script = 'snap_and_displacements_angle_phase'
        action_scale = {'alpha':4, 'theta':pi, 'phi':1}
        to_learn = {'alpha':True, 'theta':True, 'phi':True}
        
        train_batch_size = 100
        eval_batch_size = 100
        
        # Create drivers for data collection
        from gkp.agents import dynamic_episode_driver_sim_env
        
        collect_driver = dynamic_episode_driver_sim_env.DynamicEpisodeDriverSimEnv(
            env_kwargs, reward_kwargs, train_batch_size, 
            action_script, action_scale, to_learn)
        
        eval_driver = dynamic_episode_driver_sim_env.DynamicEpisodeDriverSimEnv(
            env_kwargs, reward_kwargs_eval, eval_batch_size, 
            action_script, action_scale, to_learn)
        
        
        PPO.train_eval(
                root_dir = sim_dir,
                random_seed = seed,
                num_epochs = 2000,
                # Params for train
                normalize_observations = True,
                normalize_rewards = False,
                discount_factor = 1.0,
                lr = 1e-4,
                lr_schedule = lambda x: 1e-3 if x<500 else 1e-4,
                num_policy_updates = 20,
                initial_adaptive_kl_beta = 0.0,
                kl_cutoff_factor = 0,
                importance_ratio_clipping = 0.3,
                value_pred_loss_coef = 0.005,
                # Params for log, eval, save
                eval_interval = 50,
                save_interval = 50,
                checkpoint_interval = 10000,
                summary_interval = 10000,
                # Params for data collection
                train_batch_size = train_batch_size,
                eval_batch_size = eval_batch_size,
                collect_driver = collect_driver,
                eval_driver = eval_driver,
                train_episode_length = lambda x: 5,
                eval_episode_length = 5,
                replay_buffer_capacity = 6000,
                # Policy and value networks
                ActorNet = actor_distribution_network_gkp.ActorDistributionNetworkGKP,
                actor_fc_layers = (),
                value_fc_layers = (),
                use_rnn = True,
                actor_lstm_size = (12,),
                value_lstm_size = (12,)
                )
        
        print('Fock %d, random seed %d' %(fock,seed))