# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 13:43:31 2020

@author: Vladimir Sivak
"""

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# append parent 'gkp-rl' directory to path 
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from math import sqrt, pi
from gkp.agents import PPO
from tf_agents.networks import actor_distribution_network
from gkp.agents import actor_distribution_network_gkp
from gkp.gkp_tf_env import helper_functions as hf

"""
Train PPO agent to do quantum error correction with modified version of Baptiste 
autonomous circuit (which also includes a small feedback), on a square GKP code.

This assumes that we can start the episodes with already prepared GKP states 
and do logical Pauli measurements in the end to assign reward.

History of measurement outcomes is used to find non-Markovian strategy.

"""

root_dir = r'E:\data\gkp_sims\PPO\examples\gkp_square_qec_BsB_B100_lr3e-4'
root_dir = os.path.join(root_dir,'4')

# Params for environment
env_kwargs = {
    'simulate' : 'gkp_qec_autonomous_BsB_osc',
    'encoding' : 'square',
    'init' : 'random',
    'H' : 1,
    'T' : 2, 
    'attn_step' : 1}

# Params for reward function
reward_kwargs = {
    'reward_mode' : 'pauli', 
    'code_flips' : False}

# Params for action wrapper
action_script = 'v3_Baptiste_autonomous_2round'
action_scale = {'beta':1, 'phi':pi, 'epsilon':1}
to_learn = {'beta':False, 'phi':True, 'epsilon':True}

train_batch_size = 100
eval_batch_size = 1000

# Create drivers for data collection
from gkp.agents import dynamic_episode_driver_sim_env

collect_driver = dynamic_episode_driver_sim_env.DynamicEpisodeDriverSimEnv(
    env_kwargs, reward_kwargs, train_batch_size, 
    action_script, action_scale, to_learn)

eval_driver = dynamic_episode_driver_sim_env.DynamicEpisodeDriverSimEnv(
    env_kwargs, reward_kwargs, eval_batch_size, 
    action_script, action_scale, to_learn)


PPO.train_eval(
        root_dir = root_dir,
        random_seed = 4,
        num_epochs = 1000,
        # Params for train
        normalize_observations = True,
        normalize_rewards = False,
        discount_factor = 1.0,
        lr = 3e-4,
        lr_schedule = None,
        num_policy_updates = 20,
        initial_adaptive_kl_beta = 0.0,
        kl_cutoff_factor = 0,
        importance_ratio_clipping = 0.1,
        value_pred_loss_coef = 0.005,
        # Params for log, eval, save
        eval_interval = 50,
        save_interval = 50,
        checkpoint_interval = 1000,
        summary_interval = 100,
        # Params for data collection
        train_batch_size = train_batch_size,
        eval_batch_size = eval_batch_size,
        collect_driver = collect_driver,
        eval_driver = eval_driver,
        train_episode_length =  lambda x: 36 if x<500 else 100,
        eval_episode_length = 64,
        replay_buffer_capacity = 50000,
        # Policy and value networks
        ActorNet = actor_distribution_network_gkp.ActorDistributionNetworkGKP,
        actor_fc_layers = (100,50),
        value_fc_layers = (100,50),
        use_rnn = False,
        actor_lstm_size = (12,),
        value_lstm_size = (12,)
        )