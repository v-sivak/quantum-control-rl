# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 15:36:37 2023

@author: qulab
"""

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# append parent 'gkp-rl' directory to path 
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import qutip as qt
from math import sqrt, pi
import numpy as np
from rl_tools.agents import PPO
from tf_agents.networks import actor_distribution_network
from rl_tools.remote_env_tools import remote_env_tools as rmt

root_dir = r'E:\data\exp_training\pi_pulse_1'

server_socket = rmt.Server()
(host, port) = ('172.28.142.46', 5555)
server_socket.bind((host, port))
server_socket.connect_client()

# Params for environment
env_kwargs = eval_env_kwargs = {
    'control_circuit' : 'pi_pulse_circuit_remote_env',
    'init' : 'vac',
    'T' : 1,
    'N' : 20}

# Params for reward function
reward_kwargs = {
    'reward_mode' : 'remote',
    'server_socket' : server_socket,
    'epoch_type' : 'training',
    'N_msmt' : 500}

reward_kwargs_eval = {
    'reward_mode' : 'remote',
    'server_socket' : server_socket,
    'epoch_type' : 'evaluation',
    'N_msmt' : 5000}

# Params for action wrapper
action_script = 'pi_pulse_circuit_init'
action_scale = {'amplitude':0.5, 
                'detune':5e6}
to_learn = {'amplitude':True, 
            'detune':True}
train_batch_size = 20
eval_batch_size = 1

learn_residuals = True

train_episode_length = lambda x: env_kwargs['T']
eval_episode_length = lambda x: env_kwargs['T']

# Create drivers for data collection
from rl_tools.agents import dynamic_episode_driver_sim_env

collect_driver = dynamic_episode_driver_sim_env.DynamicEpisodeDriverSimEnv(
    env_kwargs, reward_kwargs, train_batch_size, action_script, action_scale, 
    to_learn, train_episode_length, learn_residuals, remote=True)

eval_driver = dynamic_episode_driver_sim_env.DynamicEpisodeDriverSimEnv(
    eval_env_kwargs, reward_kwargs_eval, eval_batch_size, action_script, action_scale, 
    to_learn, eval_episode_length, learn_residuals, remote=True)

PPO.train_eval(
    root_dir = root_dir,
    random_seed = 0,
    num_epochs = 100,
    # Params for train
    normalize_observations = True,
    normalize_rewards = False,
    discount_factor = 1.0,
    lr = 2.5e-3, 
    lr_schedule = None,
    num_policy_updates = 20,
    initial_adaptive_kl_beta = 0.0,
    kl_cutoff_factor = 0,
    importance_ratio_clipping = 0.1,
    value_pred_loss_coef = 0.005,
    gradient_clipping = 1.0,
    entropy_regularization = 0,
    log_prob_clipping = 0.0,
    # Params for log, eval, save
    eval_interval = 20,
    save_interval = 2,
    checkpoint_interval = None,
    summary_interval = 2,
    do_evaluation = True,
    # Params for data collection
    train_batch_size = train_batch_size,
    eval_batch_size = eval_batch_size,
    collect_driver = collect_driver,
    eval_driver = eval_driver,
    replay_buffer_capacity = 15000,
    # Policy and value networks
    ActorNet = actor_distribution_network.ActorDistributionNetwork,
    zero_means_kernel_initializer = True,
    init_action_stddev = 0.08,
    actor_fc_layers = (50,20),
    value_fc_layers = (),
    use_rnn = False,
    actor_lstm_size = (12,),
    value_lstm_size = (12,))
