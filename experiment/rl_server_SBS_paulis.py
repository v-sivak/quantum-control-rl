# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 14:20:11 2021

@author: Vladimir Sivak
"""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# append parent 'gkp-rl' directory to path 
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import qutip as qt
from math import sqrt, pi
from rl_tools.agents import PPO
from tf_agents.networks import actor_distribution_network
from rl_tools.remote_env_tools import remote_env_tools as rmt



root_dir = r'E:\data\gkp_sims\PPO\ECD\EXP_Vlad\sbs_pauli\run_18'

server_socket = rmt.Server()
(host, port) = ('172.28.142.46', 5555)
server_socket.bind((host, port))
server_socket.connect_client()

# Params for environment
env_kwargs = eval_env_kwargs = {
    'control_circuit' : 'SBS_remote',
    'init' : 'vac',
    'T' : 1,
    'N' : 20}

# Params for reward function
reward_kwargs = {
    'reward_mode' : 'stabilizer_remote',
    'server_socket' : server_socket,
    'epoch_type' : 'training',
    'N_msmt' : 50,
    'stabilizer_amplitudes' : [sqrt(pi/2), 1j*sqrt(pi/2)],
    'penalty_coeff' : 1.0}

reward_kwargs_eval ={
    'reward_mode' : 'stabilizer_remote',
    'server_socket' : server_socket,
    'epoch_type' : 'training',
    'N_msmt' : 30,
    'stabilizer_amplitudes' : [sqrt(pi/2), 1j*sqrt(pi/2)],
    'penalty_coeff' : 1.0}

# Params for action wrapper
action_script = 'SBS_remote_residuals_2'
action_scale = {'beta':0.3, 'phi':0.3, 'flip':0.3, 
                'cavity_phase':0.2, 'Kerr_drive_amp':0.2, 'alpha_correction':0.2,
                'detune_sbs':2e6, 'drag_sbs':4.0, 'detune_reset':2e6, 'drag_reset':4.0}
to_learn = {'beta':False, 'phi':False, 'flip':False, 
            'cavity_phase':True, 'Kerr_drive_amp':True, 'alpha_correction':False,
            'detune_sbs':False, 'drag_sbs':False, 'detune_reset':False, 'drag_reset':False}


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
        num_epochs = 2000,
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
        log_prob_clipping = 0.0,
        # Params for log, eval, save
        eval_interval = 100,
        save_interval = 2,
        checkpoint_interval = None,
        summary_interval = 2,
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
        init_action_stddev = 0.05,
        actor_fc_layers = (),
        value_fc_layers = (),
        use_rnn = False,
        actor_lstm_size = (12,),
        value_lstm_size = (12,)
        )

