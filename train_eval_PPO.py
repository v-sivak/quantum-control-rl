# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:13:33 2020

@author: Vladimir Sivak
"""

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from gkp.agents import PPO


root_dir = r'E:\VladGoogleDrive\Qulab\GKP\sims\PPO\Baptiste'
root_dir = os.path.join(root_dir,'rnn_maxstep50_lr1e-5_mixed_4round')


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
        horizon = 1,
        max_episode_length = 50,
        eval_episode_length = 200,
        reward_mode = 'mixed',
        quantum_circuit_type = 'v3',
        action_script = 'Baptiste_4round',
        # Policy and value networks
        actor_fc_layers = (),
        value_fc_layers = (),
        use_rnn = True,
        actor_lstm_size = (12,),
        value_lstm_size = (12,)
        )