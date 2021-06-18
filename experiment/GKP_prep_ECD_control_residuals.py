# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 13:26:26 2021

@author: Vladimir Sivak
"""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# append parent 'gkp-rl' directory to path 
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from math import sqrt, pi
from rl_tools.agents import PPO
from tf_agents.networks import actor_distribution_network
from rl_tools.remote_env_tools import remote_env_tools as rmt



root_dir = r'E:\data\gkp_sims\PPO\ECD\EXP_Vlad\GKP_plus_Z\run_2'

server_socket = rmt.Server()
(host, port) = ('172.28.142.46', 5555)
server_socket.bind((host, port))
server_socket.connect_client()

# Params for environment
env_kwargs = eval_env_kwargs = {
    'control_circuit' : 'ECD_control_remote',
    'init' : 'vac',
    'T' : 10,
    'N' : 20}

# Params for reward function
reward_kwargs = {
    'reward_mode' : 'stabilizer_remote',
    'server_socket' : server_socket,
    'epoch_type' : 'training',
    'N_msmt' : 50,
    'stabilizer_amplitudes' : [sqrt(2*pi), -sqrt(2*pi), 
                               1j*sqrt(pi/2), -1j*sqrt(pi/2)]}

reward_kwargs_eval = {
    'reward_mode' : 'stabilizer_remote',
    'server_socket' : server_socket,
    'epoch_type' : 'training',
    'N_msmt' : 50,
    'stabilizer_amplitudes' : [sqrt(2*pi), -sqrt(2*pi), 
                               1j*sqrt(pi/2), -1j*sqrt(pi/2)]}

# Params for action wrapper
action_script = 'ECD_control_residuals_GKP'
action_scale = {'beta':0.2, 'phi':0.2}
to_learn = {'beta':True, 'phi':True}

train_batch_size = 10
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
        num_epochs = 500,
        # Params for train
        normalize_observations = True,
        normalize_rewards = False,
        discount_factor = 1.0,
        lr = 5e-3,
        lr_schedule = None,
        num_policy_updates = 20,
        initial_adaptive_kl_beta = 0.0,
        kl_cutoff_factor = 0,
        importance_ratio_clipping = 0.1,
        value_pred_loss_coef = 0.005,
        gradient_clipping = 1.0,
        entropy_regularization = 0,
        # Params for log, eval, save
        eval_interval = 100,
        save_interval = 1,
        checkpoint_interval = None,
        summary_interval = 1,
        do_evaluation = False,
        # Params for data collection
        train_batch_size = train_batch_size,
        eval_batch_size = eval_batch_size,
        collect_driver = collect_driver,
        eval_driver = eval_driver,
        replay_buffer_capacity = 15000,
        # Policy and value networks
        ActorNet = actor_distribution_network.ActorDistributionNetwork,
        zero_means_kernel_initializer = True,
        actor_fc_layers = (),
        value_fc_layers = (),
        use_rnn = False,
        actor_lstm_size = (12,),
        value_lstm_size = (12,)
        )

