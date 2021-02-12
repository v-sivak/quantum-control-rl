# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 12:54:41 2020

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
Train PPO agent on two tasks: 
    
    1) CD gate calibration for 
    2) gkp quantum error correction with sBs protocol.

"""

root_dir = r'E:\data\gkp_sims\PPO\examples'
root_dir = os.path.join(root_dir,'test3')

# target state to prepare
stabilizers, pauli, states, code_map = hf.GKP_state(False, 100)
target_state = states['X+']

# First environment: train on fidelity reward "in simulation"
env_kwargs_0 = dict(simulate='snap_and_displacement', 
                    init='vac', H=1, T=5, attn_step=1, N=100)

rew_kwargs_0 = dict(reward_mode='overlap', target_state=target_state)

episode_length_0 =  lambda x: 5


# Second environment: train on Wigner reward "in experiment"
env_kwargs_1 = dict(simulate='snap_and_displacement', 
                    init='vac', H=1, T=5, attn_step=1, N=100)

rew_kwargs_1 = dict(reward_mode = 'tomography',
                    tomography = 'wigner',
                    target_state = target_state,
                    window_size = 16,
                    sample_from_buffer = False,
                    buffer_size = 500)

episode_length_1 =  lambda x: 5


env_kwargs_list = [env_kwargs_0, env_kwargs_1]
rew_kwargs_list = [rew_kwargs_0, rew_kwargs_1]
episode_length_list = [episode_length_0, episode_length_1]

# schedule to first train "in simulation", then switch to "experiment"
env_schedule = lambda epoch: 0 if epoch<1000 else 1 



# Params for action wrapper
action_script = 'snap_and_displacements'
action_scale = {'alpha':6, 'theta':pi}
to_learn = {'alpha':True, 'theta':True}

train_batch_size = 1000
eval_batch_size = 1000


# Create drivers for data collection
from gkp.agents import multitask_episode_driver_sim_env
collect_driver = multitask_episode_driver_sim_env.MultitaskEpisodeDriverSimEnv(
    env_kwargs_list, rew_kwargs_list, train_batch_size, 
    action_script, action_scale, to_learn, episode_length_list, env_schedule=env_schedule)

from gkp.agents import dynamic_episode_driver_sim_env
eval_driver = dynamic_episode_driver_sim_env.DynamicEpisodeDriverSimEnv(
    env_kwargs_0, rew_kwargs_0, eval_batch_size, 
    action_script, action_scale, to_learn, episode_length_0)


PPO.train_eval(
        root_dir = root_dir,
        random_seed = 0,
        num_epochs = 50000,
        # Params for train
        normalize_observations = True,
        normalize_rewards = False,
        discount_factor = 1.0,
        lr = 1e-3,
        lr_schedule = lambda x: 1e-3 if x<500 else 1e-4,
        num_policy_updates = 20,
        initial_adaptive_kl_beta = 0.0,
        kl_cutoff_factor = 0,
        importance_ratio_clipping = 0.1,
        value_pred_loss_coef = 0.005,
        gradient_clipping = 1.0,
        # Params for log, eval, save
        eval_interval = 10,
        save_interval = 10,
        checkpoint_interval = 10000,
        summary_interval = 10,
        # Params for data collection
        train_batch_size = train_batch_size,
        eval_batch_size = eval_batch_size,
        collect_driver = collect_driver,
        eval_driver = eval_driver,
        replay_buffer_capacity = 7000,
        # Policy and value networks
        ActorNet = actor_distribution_network_gkp.ActorDistributionNetworkGKP,
        actor_fc_layers = (),
        value_fc_layers = (),
        use_rnn = True,
        actor_lstm_size = (12,),
        value_lstm_size = (12,)
        )