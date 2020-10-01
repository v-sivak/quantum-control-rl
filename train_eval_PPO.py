# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:13:33 2020

@author: Vladimir Sivak
"""

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from math import sqrt, pi
from gkp.agents import PPO
from tf_agents.networks import actor_distribution_network
from gkp.agents import actor_distribution_network_gkp

root_dir = r'E:\VladGoogleDrive\Qulab\GKP\sims\PPO\overlap'
root_dir = os.path.join(root_dir,'fock5_beta4')

import qutip as qt
import tensorflow as tf
N=20
target = qt.tensor(qt.basis(2,0), qt.basis(N,5))
target_projector = qt.ket2dm(target)
target_projector = tf.constant(target_projector.full(), dtype=tf.complex64)

kwargs = {'N': N, 'target_projector': target_projector}

PPO.train_eval(
        root_dir = root_dir,
        random_seed = 0,
        # Params for collect
        num_iterations = 100000,
        train_batch_size = 100,
        replay_buffer_capacity = 15000,
        # Params for train
        normalize_observations = True,
        normalize_rewards = False,
        discount_factor = 1.0,
        lr = 1e-3,
        lr_schedule = None,
        num_policy_epochs = 20,
        initial_adaptive_kl_beta = 0.0,
        kl_cutoff_factor = 0,
        importance_ratio_clipping = 0.1,
        value_pred_loss_coef = 0.005,
        # Params for log, eval, save
        eval_batch_size = 500,
        eval_interval = 100,
        save_interval = 500,
        checkpoint_interval = 5000,
        summary_interval = 100,
        # Params for environment
        simulate = 'Alec_universal_gate_set',
        horizon = 1,
        clock_period = 6,
        attention_step = 1,
        train_episode_length = lambda x: 6,
        eval_episode_length = 6,
        init_state = 'vac',
        reward_mode = 'overlap',
        encoding = 'square',
        action_script = 'Alec_universal_gate_set_6round',
        to_learn = {'alpha':True, 'beta':True, 'phi':True},
        # Policy and value networks
        ActorNet = actor_distribution_network_gkp.ActorDistributionNetworkGKP,
        actor_fc_layers = (100,50),
        value_fc_layers = (100,50),
        use_rnn = False,
        actor_lstm_size = (12,),
        value_lstm_size = (12,),
        **kwargs
        )