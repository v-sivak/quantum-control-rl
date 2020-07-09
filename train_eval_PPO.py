# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:13:33 2020

@author: Vladimir Sivak
"""

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from gkp.agents import PPO


root_dir = r'E:\VladGoogleDrive\Qulab\GKP\sims\PPO\May'
root_dir = os.path.join(root_dir,'test')


PPO.train_eval(
        root_dir = root_dir,
        random_seed = 0,
        # Params for collect
        num_iterations = 1000000,
        train_batch_size = 100,
        replay_buffer_capacity = 20000,
        # Params for train
        normalize_observations = True,
        normalize_rewards = True,
        discount_factor = 1.0,
        lr = 1e-5,
        lr_schedule = lambda t: max(1/(1e3+t), 1e-5),
        num_policy_epochs = 20,
        initial_adaptive_kl_beta = 0.0,
        kl_cutoff_factor = 0,
        importance_ratio_clipping = 0.2,
        value_pred_loss_coef = 0.5,
        # Params for log, eval, save
        eval_batch_size = 200,
        eval_interval = 200,
        save_interval = 200,
        checkpoint_interval = None,
        summary_interval = 100,
        # Params for environment
        simulate = 'oscillator',
        horizon = 1,
        clock_period = 4,
        train_episode_length = lambda x: 24,
        eval_episode_length = 24,
        reward_mode = 'pauli',
        encoding = 'square',
        quantum_circuit_type = 'v3',
        action_script = 'Baptiste_4round',
        to_learn = {'alpha':True, 'beta':True, 'epsilon':True, 'phi':True},
        observations_whitelist = None,
        # Policy and value networks
        actor_fc_layers = (),
        value_fc_layers = (),
        use_rnn = True,
        actor_lstm_size = (12,),
        value_lstm_size = (12,)
        )