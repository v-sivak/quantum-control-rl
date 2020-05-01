# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:13:33 2020

@author: Vladimir Sivak
"""

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from gkp.agents import PPO


root_dir = r'E:\VladGoogleDrive\Qulab\GKP\sims\PPO\dict_actions\Baptiste'
root_dir = os.path.join(root_dir,'mlp_maxstep24_batch100_pauli_test')


PPO.train_eval(
        root_dir = root_dir,
        random_seed = 0,
        # Params for collect
        num_iterations = 1000000,
        train_batch_size = 100,
        replay_buffer_capacity = 20000,
        # Params for train
        normalize_observations = False,
        normalize_rewards = True,
        discount_factor = 1.0,
        lr = 1e-5,
        num_policy_epochs = 20,
        initial_adaptive_kl_beta = 0.0,
        kl_cutoff_factor = 0,
        importance_ratio_clipping = 0.2,
        # Params for log, eval, save
        eval_batch_size = 200,
        eval_interval = 200,
        save_interval = 1000,
        log_interval = 20,
        # Params for environment
        horizon = 12,
        max_episode_length = 24,
        eval_episode_length = 24,
        reward_mode = 'pauli',
        quantum_circuit_type = 'v3',
        action_script = 'Baptiste_4round',
        to_learn = {'alpha':True, 'beta':False, 'epsilon':True, 'phi':False},
        # Policy and value networks
        actor_fc_layers = (200,),
        value_fc_layers = (200,),
        use_rnn = False,
        actor_lstm_size = (12,),
        value_lstm_size = (12,)
        )