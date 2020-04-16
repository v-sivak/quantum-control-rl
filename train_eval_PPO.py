# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:13:33 2020

@author: Vladimir Sivak
"""

from gkp.agents import PPO
import os


root_dir = r'E:\VladGoogleDrive\Qulab\GKP\sims\PPO'
root_dir = os.path.join()


PPO.train_eval(
        root_dir = root_dir,
        random_seed = 0,
        # Params for collect
        num_iterations = 1000000,
        train_batch_size = 10,
        replay_buffer_capacity = 20000,
        # Params for train
        normalize_observations = True,
        normalize_rewards = True,
        discount_factor = 0.95,
        lr = 1e-5,
        num_policy_epochs = 20,
        initial_adaptive_kl_beta = 0.0,
        kl_cutoff_factor = 0,
        importance_ratio_clipping = 0.2,
        # Params for log, eval, save
        eval_batch_size = 100,
        eval_interval = 100,
        save_interval = 1000,
        log_interval = 20,
        # Params for environment
        horizon = 1,
        max_episode_length = 24,
        eval_episode_length = 200,
        reward_mode = 'stabilizers',
        quantum_circuit_type = 'v3',
        action_script = 'Baptiste_8round',
        # Policy and value networks
        actor_fc_layers = (),
        value_fc_layers = (),
        use_rnn = True,
        actor_lstm_size = (12,),
        value_lstm_size = (12,)
        )