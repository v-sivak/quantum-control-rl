# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:13:33 2020

@author: Vladimir Sivak
"""

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from gkp.agents import SAC


root_dir = r'E:\VladGoogleDrive\Qulab\GKP\sims\SAC\OscillatorGKP'
root_dir = os.path.join(root_dir,'test')


SAC.train_eval(
        root_dir = root_dir,
        random_seed = 0,
        # Params for collect
        num_iterations = 1000000,
        train_batch_size = 100,
        replay_buffer_capacity = 1000000,
        # Params for target update
        target_update_tau = 0.05,
        target_update_period = 5,
        # Params for train
        train_sequence_length = 24,
        reward_scale_factor = 1.0,
        gradient_clipping = None,
        discount_factor = 1,
        critic_learning_rate = 1e-5,
        actor_learning_rate = 1e-5,
        alpha_learning_rate = 1e-5,
        train_steps_per_iteration = 1,
        # Params for log, eval, save
        eval_batch_size = 100,
        eval_interval = 100,
        save_interval = 1000,
        log_interval = 20,
        # Params for environment
        simulate = 'oscillator',
        horizon = 1,
        max_episode_length = 24,
        eval_episode_length = 24,
        reward_mode = 'fidelity',
        quantum_circuit_type = 'v2',
        action_script = 'phase_estimation_symmetric_with_trim_4round',
        to_learn = {'alpha':True, 'beta':True, 'phi':True},
        # Actor and critic networks
        actor_fc_layers = (),
        actor_output_fc_layers = (),
        critic_obs_fc_layers = None,
        critic_action_fc_layers = None,
        critic_joint_fc_layers = (),
        critic_output_fc_layers = (),
        use_rnn = True,
        actor_lstm_size = (12,),
        critic_lstm_size = (12,)
        )