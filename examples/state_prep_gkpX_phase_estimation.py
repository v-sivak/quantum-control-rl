# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 17:07:18 2020

@author: Vladimir Sivak
"""

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# append parent 'gkp-rl' directory to path 
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import qutip as qt
import tensorflow as tf
import numpy as np
from math import sqrt, pi
from gkp.agents import PPO
from tf_agents.networks import actor_distribution_network
from gkp.agents import actor_distribution_network_gkp
from gkp.gkp_tf_env import helper_functions as hf

"""
Train PPO agent to do GKP X+ Pauli eigenstate preparation with adaptive phase 
estimation of two state stabilizers.

The episodes start from vacuum, and stabilizer measurements are performed in 
the end to assign reward.

"""



root_dir = r'E:\VladGoogleDrive\Qulab\GKP\sims\PPO\examples'
root_dir = os.path.join(root_dir,'state_prep_gkpX_phase_estimation')


env_kwargs = {
    'simulate' : 'phase_estimation_osc_qb_v1',
    'init' : 'vac',
    'H' : 1,
    'T' : 8, 
    'attn_step' : 1}

reward_kwargs = {'reward_mode' : 'stabilizers', 
                 'stabilizer_translations' : [sqrt(pi)+0j, 2j*sqrt(pi)]}

PPO.train_eval(
        root_dir = root_dir,
        random_seed = 0,
        # Params for collect
        num_iterations = 10000,
        train_batch_size = 1000,
        replay_buffer_capacity = 15000,
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
        eval_batch_size = 1000,
        eval_interval = 100,
        save_interval = 500,
        checkpoint_interval = 1000,
        summary_interval = 100,
        # Params for environment
        train_env_kwargs = env_kwargs,
        eval_env_kwargs = env_kwargs,
        reward_kwargs = reward_kwargs,
        train_episode_length = lambda x: 8,
        eval_episode_length = 8,
        # Params for action wrapper
        action_script = 'v1_phase_estimation_X_prep_4round',
        action_scale = {'alpha':1, 'beta':1, 'phi':pi},
        to_learn = {'alpha':True, 'beta':False, 'phi':True},
        # Policy and value networks
        ActorNet = actor_distribution_network_gkp.ActorDistributionNetworkGKP,
        actor_fc_layers = (),
        value_fc_layers = (),
        use_rnn = True,
        actor_lstm_size = (12,),
        value_lstm_size = (12,)
        )