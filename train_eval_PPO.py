# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:13:33 2020

@author: Vladimir Sivak
"""

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from gkp.agents import PPO
from tf_agents.networks import actor_distribution_network
from gkp.agents import actor_distribution_network_gkp

root_dir = r'E:\VladGoogleDrive\Qulab\GKP\sims\PPO\July\OscillatorGKP'
root_dir = os.path.join(root_dir,'test2')


PPO.train_eval(
        root_dir = root_dir,
        random_seed = 0,
        # Params for collect
        num_iterations = 1000000,
        train_batch_size = 10,
        replay_buffer_capacity = 70000,
        # Params for train
        normalize_observations = True,
        normalize_rewards = False,
        discount_factor = 1.0,
        lr = 1e-4,
        lr_schedule = None,
        num_policy_epochs = 20,
        initial_adaptive_kl_beta = 0.0,
        kl_cutoff_factor = 0,
        importance_ratio_clipping = 0.1,
        value_pred_loss_coef = 0.005,
        # Params for log, eval, save
        eval_batch_size = 20,
        eval_interval = 100,
        save_interval = 500,
        checkpoint_interval = 500,
        summary_interval = 100,
        # Params for environment
        simulate = 'oscillator',
        horizon = 4,
        clock_period = 4,
        train_episode_length = lambda x: 12,
        eval_episode_length = 12,
        reward_mode = 'pauli',
        encoding = 'square',
        quantum_circuit_type = 'v2',
        action_script = 'phase_estimation_symmetric_with_trim_4round',
        to_learn = {'alpha':True, 'beta':True, 'phi':False},
        # Policy and value networks
        ActorNet = actor_distribution_network_gkp.ActorDistributionNetworkGKP,
        actor_fc_layers = (200,100,50),
        value_fc_layers = (200,100,50),
        use_rnn = False,
        actor_lstm_size = (12,),
        value_lstm_size = (12,),
        )