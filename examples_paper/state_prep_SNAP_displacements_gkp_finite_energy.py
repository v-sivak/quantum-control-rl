# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 18:13:41 2020

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
Train PPO agent to do GKP sensor state preparation with universal gate sequence
consisting of SNAP gates and oscillator displacements.

The episodes start from vacuum, and GKP stabilizer measurements are performed
in the end to assign reward.

"""

root_dir = r'E:\data\gkp_sims\PPO\examples\gkp_S50'
deltas = [0.25]
random_seeds = [2]

for Delta in deltas:
    # setup directory for each simulation
    delta_dir = os.path.join(root_dir,'delta'+str(Delta))
    if not os.path.isdir(delta_dir): os.mkdir(delta_dir)

    for seed in random_seeds:
        sim_dir = os.path.join(delta_dir,'seed'+str(seed))

        # Params for environment
        env_kwargs = {
            'simulate' : 'snap_and_displacement',
            'init' : 'vac',
            'H' : 1,
            'T' : 6, 
            'attn_step' : 1,
            'N' : 200}
        
        # Params for reward function
        reward_kwargs = {'reward_mode' : 'stabilizers_v2',
                         'Delta' : Delta, 'beta' : sqrt(pi),
                         'sample' : True}
        
        reward_kwargs_eval = {'reward_mode' : 'stabilizers_v2',
                              'Delta' : Delta, 'beta' : sqrt(pi),
                              'sample' : False}
        
        # Params for action wrapper
        action_script = 'snap_and_displacements'
        action_scale = {'alpha':6, 'theta':pi}
        to_learn = {'alpha':True, 'theta':True}
        
        train_batch_size = 1000
        eval_batch_size = 1000
        
        train_episode_length = lambda x: env_kwargs['T']
        eval_episode_length = lambda x: env_kwargs['T']
        
        # Create drivers for data collection
        from gkp.agents import dynamic_episode_driver_sim_env
        
        collect_driver = dynamic_episode_driver_sim_env.DynamicEpisodeDriverSimEnv(
            env_kwargs, reward_kwargs, train_batch_size, 
            action_script, action_scale, to_learn, train_episode_length)
        
        eval_driver = dynamic_episode_driver_sim_env.DynamicEpisodeDriverSimEnv(
            env_kwargs, reward_kwargs_eval, eval_batch_size, 
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
                lr_schedule = None,
                num_policy_updates = 20,
                initial_adaptive_kl_beta = 0.0,
                kl_cutoff_factor = 0,
                importance_ratio_clipping = 0.25,
                value_pred_loss_coef = 0.005,
                gradient_clipping = 1.0,
                # Params for log, eval, save
                eval_interval = 200,
                save_interval = 200,
                checkpoint_interval = 100000,
                summary_interval = 10000,
                # Params for data collection
                train_batch_size = train_batch_size,
                eval_batch_size = eval_batch_size,
                collect_driver = collect_driver,
                eval_driver = eval_driver,
                replay_buffer_capacity = 7000,
                # Policy and value networks
                ActorNet = actor_distribution_network.ActorDistributionNetwork,
                actor_fc_layers = (),
                value_fc_layers = (),
                use_rnn = True,
                actor_lstm_size = (12,),
                value_lstm_size = (12,)
                )