# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 10:36:27 2020

@author: qulab
"""

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# append parent 'gkp-rl' directory to path 
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from math import pi
import tensorflow as tf
from gkp.agents import PPO
from gkp.agents import actor_distribution_network_gkp

"""
Train PPO agent calibrate CD gate. This is a total overkill to use PPO here,
but it works so why not?

"""

root_dir = r'E:\data\gkp_sims\PPO\examples\CD_cal\100_phi'

# Params for environment
env_kwargs = {
    'simulate' : 'conditional_displacement_cal',
    'init' : 'vac',
    'T' : 1, 
    'N' : 50,
    't_gate' : 100e-9}

reward_kwargs = {'reward_mode' : 'measurement',
                 'sample' : True}

reward_kwargs_eval = {'reward_mode' : 'measurement',
                      'sample' : False}

# Params for action wrapper
action_script = 'conditional_displacement_cal'
action_scale = {'alpha':27, 'phi_g':pi, 'phi_e':pi} # nbar of 800 ish
to_learn = {'alpha':True, 'phi_g':True,'phi_e':True}

train_batch_size = 200
eval_batch_size = 2

train_episode_length = lambda x: 1
eval_episode_length = lambda x: 1

# Create drivers for data collection
from gkp.agents import dynamic_episode_driver_sim_env

collect_driver = dynamic_episode_driver_sim_env.DynamicEpisodeDriverSimEnv(
    env_kwargs, reward_kwargs, train_batch_size, 
    action_script, action_scale, to_learn, train_episode_length)

eval_driver = dynamic_episode_driver_sim_env.DynamicEpisodeDriverSimEnv(
    env_kwargs, reward_kwargs_eval, eval_batch_size, 
    action_script, action_scale, to_learn, eval_episode_length)

PPO.train_eval(
        root_dir = root_dir,
        random_seed = 0,
        num_epochs = 3000,
        # Params for train
        normalize_observations = True,
        normalize_rewards = False,
        discount_factor = 1.0,
        lr = 1e-4,
        lr_schedule = None,
        num_policy_updates = 20,
        initial_adaptive_kl_beta = 0.0,
        kl_cutoff_factor = 0,
        importance_ratio_clipping = 0.1,
        value_pred_loss_coef = 0.005,
        # Params for log, eval, save
        eval_interval = 10,
        save_interval = 100,
        checkpoint_interval = 10000,
        summary_interval = 10,
        # Params for data collection
        train_batch_size = train_batch_size,
        eval_batch_size = eval_batch_size,
        collect_driver = collect_driver,
        eval_driver = eval_driver,
        replay_buffer_capacity = 2000,
        # Policy and value networks
        ActorNet = actor_distribution_network_gkp.ActorDistributionNetworkGKP,
        actor_fc_layers = (),
        value_fc_layers = (),
        use_rnn = False,
        actor_lstm_size = (12,),
        value_lstm_size = (12,)
        )