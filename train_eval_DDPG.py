# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 17:39:34 2020

@author: Vladimir Sivak
"""

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from gkp.agents import DDPG


root_dir = r'E:\VladGoogleDrive\Qulab\GKP\sims\DDPG\OscillatorGKP'
root_dir = os.path.join(root_dir,'test')


DDPG.train_eval(
    root_dir = root_dir,
    random_seed = 0,
    # Params for collect
    num_iterations = 10000000,
    train_batch_size = 400,
    replay_buffer_capacity = 100000,
    initial_collect_steps = 1000,
    collect_steps_per_iteration = 1,
    ou_stddev = 0.2,
    ou_damping = 0.15,
    # Params for target update
    target_update_tau = 0.05,
    target_update_period = 5,
    # Params for train
    reward_scale_factor = 1.0,
    gradient_clipping = None,
    dqda_clipping=None,
    discount_factor = 1.0,
    critic_learning_rate = 1e-7,
    actor_learning_rate = 1e-7,
    train_steps_per_iteration = 1,
    # Params for log, eval, save
    eval_batch_size = 200,
    eval_interval = 1000,
    save_interval = 10000,
    log_interval = 1000,
    # Params for environment
    simulate = 'oscillator',
    horizon = 4,
    max_episode_length = 24,
    eval_episode_length = 24,
    reward_mode = 'pauli',
    quantum_circuit_type = 'v2',
    action_script = 'phase_estimation_symmetric_with_trim_4round',
    to_learn = {'alpha':True, 'beta':True, 'phi':False},
    observations_whitelist = ['alpha','beta','phi','msmt','clock'],
    # Actor and critic networks
    actor_fc_layers = (200,100),
    actor_output_fc_layers = (),
    critic_obs_fc_layers = (200,),
    critic_action_fc_layers = (200,),
    critic_joint_fc_layers = (100,),
    critic_output_fc_layers = (),
    use_rnn = False,
    actor_lstm_size = (12,),
    critic_lstm_size = (12,)
    )