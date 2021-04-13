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

import qutip as qt
from math import sqrt, pi
from gkp.agents import PPO
from tf_agents.networks import actor_distribution_network


root_dir = r'E:\data\gkp_sims\PPO\ECD\EXP\fock4\run_4'

# Params for environment
env_kwargs = {
    'simulate' : 'ECD_control_remote',
    'init' : 'vac',
    'T' : 8,
    'N' : 50}

# Evaluation environment params
eval_env_kwargs = {
    'simulate' : 'ECD_control',
    'init' : 'vac',
    'T' : 8, 
    'N' : 50}

# Params for reward function
target_state = qt.tensor(qt.basis(2,0), qt.basis(50,4))

reward_kwargs = {'reward_mode' : 'remote',
                 'tomography' : 'characteristic_fn',
                 'target_state' : target_state,
                 'window_size' : 16,
                 'host_port' : ('172.28.142.46', 5555),
                 'skip' : False,
                 'amplitude_type' : 'displacement'}

reward_kwargs_eval = {'reward_mode' : 'overlap',
                      'target_state' : target_state,
                      'postselect_0' : False}

# Params for action wrapper
action_script = 'ECD_control_residuals'
action_scale = {'beta':3/8, 'phi':pi/8}
to_learn = {'beta':True, 'phi':True}

train_batch_size = 8
eval_batch_size = 100

learn_residuals = True

train_episode_length = lambda x: env_kwargs['T']
eval_episode_length = lambda x: env_kwargs['T']

# Create drivers for data collection
from gkp.agents import dynamic_episode_driver_sim_env

collect_driver = dynamic_episode_driver_sim_env.DynamicEpisodeDriverSimEnv(
    env_kwargs, reward_kwargs, train_batch_size, action_script, action_scale, 
    to_learn, train_episode_length, learn_residuals, remote=True)

eval_driver = dynamic_episode_driver_sim_env.DynamicEpisodeDriverSimEnv(
    eval_env_kwargs, reward_kwargs_eval, eval_batch_size, action_script, action_scale, 
    to_learn, eval_episode_length, learn_residuals)

PPO.train_eval(
        root_dir = root_dir,
        random_seed = 0,
        num_epochs = 300,
        # Params for train
        normalize_observations = True,
        normalize_rewards = False,
        discount_factor = 1.0,
        lr = 1e-3,
        lr_schedule = None,
        num_policy_updates = 20,
        initial_adaptive_kl_beta = 0.0,
        kl_cutoff_factor = 0,
        importance_ratio_clipping = 0.1,
        value_pred_loss_coef = 0.005,
        gradient_clipping = 1.0,
        entropy_regularization = 0,
        # Params for log, eval, save
        eval_interval = 10000,
        save_interval = 1,
        checkpoint_interval = 10000,
        summary_interval = 1,
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


# Evaluate some of the protocols at after the training is finished
policies = ['000300', '000000', '000150']
ports = [5554, 5553, 5552]

for policy_str, port in zip(policies, ports): 

    # re-establish the connection and send deterministic actions
    reward_kwargs = {'reward_mode' : 'remote',
                      'tomography' : 'characteristic_fn',
                      'target_state' : target_state,
                      'window_size' : 16,
                      'host_port' : ('172.28.142.46', port),
                      'skip' : True,
                      'amplitude_type' : 'displacement'}
    
    from gkp.gkp_tf_env import gkp_init
    from gkp.gkp_tf_env import tf_env_wrappers as wrappers
    import tensorflow as tf
    env = gkp_init(batch_size=1, reward_kwargs=reward_kwargs, **env_kwargs,
                   episode_length=env_kwargs['T'])
    import gkp.action_script as action_scripts
    action_script_obj = action_scripts.__getattribute__(action_script)
    env = wrappers.ActionWrapper(env, action_script_obj, action_scale, to_learn,
                                 learn_residuals=True)
    
    policy_dir = 'policy\\' + policy_str
    policy = tf.compat.v2.saved_model.load(os.path.join(root_dir,policy_dir))
    

    time_step = env.reset()
    policy_state = policy.get_initial_state(env.batch_size)
    while not time_step.is_last():
        action_step = policy.action(time_step, policy_state)
        policy_state = action_step.state
        time_step = env.step(action_step.action)
    
    env.server_socket.disconnect_client()
