# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 19:22:36 2020

@author: Vladimir Sivak
"""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import tensorflow as tf
from gkp.agents import PPO
from gkp.agents import actor_distribution_network_gkp


root_dir = r'E:\VladGoogleDrive\Qulab\GKP\sims\PPO\Kerr_sweep_4000'

Kerr = np.array([1,5,10,15,20,25,30,35,40,45,50]) # Kerr in Hz
t_gate = 1.2e-6/np.sqrt(np.sqrt(Kerr)) # assume gate time scales as 1/(chi*alpha_c)

for i in range(len(Kerr)):
    
    kwargs = {'K_osc' : Kerr[i], 't_gate' : t_gate[i]}
    save_dir = os.path.join(root_dir,'K%d' %Kerr[i])
    to_learn = {'alpha':True, 'beta':True, 'phi':False, 'theta': False}
    simulate = 'oscillator'
    
    tf.compat.v1.reset_default_graph() # to reset global_step used in PPO
    
    PPO.train_eval(
        root_dir = save_dir,
        random_seed = 0,
        # Params for collect
        num_iterations = 1000,
        train_batch_size = 1000,
        replay_buffer_capacity = 70000,
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
        eval_batch_size = 600,
        eval_interval = 100,
        save_interval = 1000,
        checkpoint_interval = 2000,
        summary_interval = 100,
        # Params for environment
        simulate = simulate,
        horizon = 1,
        clock_period = 6,
        attention_step = 1,
        train_episode_length = lambda x: 36 if x<500 else 64,
        eval_episode_length = 64,
        reward_mode = 'pauli',
        encoding = 'hexagonal',
        quantum_circuit_type = 'v2',
        action_script = 'hexagonal_phase_estimation_symmetric_6round',
        to_learn = to_learn,
        # Policy and value networks
        ActorNet = actor_distribution_network_gkp.ActorDistributionNetworkGKP,
        actor_fc_layers = (100,50),
        value_fc_layers = (100,50),
        use_rnn = False,
        **kwargs
        )