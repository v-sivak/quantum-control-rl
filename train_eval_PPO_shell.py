# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 10:04:56 2020

@author: Vladimir Sivak
"""

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from gkp.agents import PPO


if __name__ == '__main__':

    root_dir = r'E:\VladGoogleDrive\Qulab\GKP\sims\PPO\July\OscillatorGKP\rnn_steps36_64_lr1e-4_v2'
    random_seed = 0
    # Params for collect
    num_iterations = 1000000
    train_batch_size = 1000
    replay_buffer_capacity = 80000
    # Params for train
    normalize_observations = True
    normalize_rewards = False
    discount_factor = 1.0
    lr = 1e-4
    lr_schedule = None
    num_policy_epochs = 20
    initial_adaptive_kl_beta = 0.0
    kl_cutoff_factor = 0
    importance_ratio_clipping = 0.1
    value_pred_loss_coef = 0.005
    # Params for log, eval, save
    eval_batch_size = 200
    eval_interval = 100
    save_interval = 500
    checkpoint_interval = 500
    summary_interval = 100
    # Params for environment
    simulate = 'oscillator'
    horizon = 1
    clock_period = 4
    train_episode_length = lambda x: 36 if x<1000 else 64
    eval_episode_length = 64
    reward_mode = 'pauli'
    encoding = 'square'
    quantum_circuit_type = 'v2'
    action_script = 'phase_estimation_symmetric_with_trim_4round'
    to_learn = {'alpha':True, 'beta':True, 'phi':False}
    observations_whitelist = ['msmt', 'clock']
    # Policy and value networks
    actor_fc_layers = ()
    value_fc_layers = ()
    use_rnn = True
    actor_lstm_size = (12,)
    value_lstm_size = (12,)
    
    
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str,
                        default=root_dir)
    
    parser.add_argument('--random_seed', type=int,
                        default=random_seed)

    ### Params for collect

    parser.add_argument('--num_iterations', type=int, 
                        default=num_iterations)

    parser.add_argument('--train_batch_size', type=int, 
                        default=train_batch_size)

    parser.add_argument('--replay_buffer_capacity', type=int, 
                        default=replay_buffer_capacity)

    ### Params for train

    parser.add_argument('--normalize_observations', type=bool, 
                        default=normalize_observations)

    parser.add_argument('--normalize_rewards', type=bool, 
                        default=normalize_rewards)

    parser.add_argument('--discount_factor', type=float, 
                        default=discount_factor)

    parser.add_argument('--lr', type=float, 
                        default=lr)

    parser.add_argument('--num_policy_epochs', type=int, 
                        default=num_policy_epochs)

    parser.add_argument('--initial_adaptive_kl_beta', type=float, 
                        default=initial_adaptive_kl_beta)

    parser.add_argument('--kl_cutoff_factor', type=float, 
                        default=kl_cutoff_factor)

    parser.add_argument('--importance_ratio_clipping', type=float, 
                        default=importance_ratio_clipping)

    parser.add_argument('--value_pred_loss_coef', type=float, 
                        default=value_pred_loss_coef)

    ### Params for log, eval, save

    parser.add_argument('--eval_batch_size', type=int, 
                        default=eval_batch_size)

    parser.add_argument('--eval_interval', type=int, 
                        default=eval_interval)

    parser.add_argument('--save_interval', type=int, 
                        default=save_interval)
    
    parser.add_argument('--checkpoint_interval', type=int, 
                        default=checkpoint_interval)

    parser.add_argument('--summary_interval', type=int, 
                        default=summary_interval)
    
    ### Params for environment
    
    parser.add_argument('--simulate', type=str, 
                        default=simulate)
    
    parser.add_argument('--horizon', type=int, 
                        default=horizon)

    parser.add_argument('--clock_period', type=int, 
                        default=clock_period)
    
    parser.add_argument('--eval_episode_length', type=int, 
                        default=eval_episode_length)
    
    parser.add_argument('--reward_mode', type=str,
                        default=reward_mode)

    parser.add_argument('--encoding', type=str,
                        default=encoding)
    
    parser.add_argument('--quantum_circuit_type', type=str,
                        default=quantum_circuit_type)
    
    parser.add_argument('--action_script', type=str,
                        default=action_script)

    ### Policy and value networks

    parser.add_argument('--actor_fc_layers', type=tuple,
                        default=actor_fc_layers)

    parser.add_argument('--value_fc_layers', type=tuple,
                        default=value_fc_layers)

    parser.add_argument('--actor_lstm_size', type=tuple,
                        default=actor_lstm_size)

    parser.add_argument('--value_lstm_size', type=tuple,
                        default=value_lstm_size)

    parser.add_argument('--use_rnn', type=bool,
                        default=use_rnn)

    args = parser.parse_args()



    PPO.train_eval(
        root_dir=args.root_dir, 
        random_seed=args.random_seed,
        num_iterations=args.num_iterations,
        train_batch_size=args.train_batch_size,
        replay_buffer_capacity=args.replay_buffer_capacity,
        normalize_observations=args.normalize_observations,
        normalize_rewards=args.normalize_rewards, 
        lr=args.lr,
        lr_schedule=lr_schedule,
        discount_factor=args.discount_factor, 
        num_policy_epochs=args.num_policy_epochs,
        initial_adaptive_kl_beta=args.initial_adaptive_kl_beta,
        kl_cutoff_factor=args.kl_cutoff_factor,
        importance_ratio_clipping=args.importance_ratio_clipping,
        value_pred_loss_coef=args.value_pred_loss_coef,
        eval_batch_size=args.eval_batch_size,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        checkpoint_interval=args.checkpoint_interval,
        summary_interval=args.summary_interval,
        simulate=args.simulate,
        horizon=args.horizon,
        clock_period=args.clock_period,
        train_episode_length=train_episode_length,
        eval_episode_length=args.eval_episode_length,
        reward_mode=args.reward_mode,
        encoding=args.encoding,
        quantum_circuit_type=args.quantum_circuit_type,
        action_script=args.action_script,
        to_learn=to_learn,
        observations_whitelist=observations_whitelist,
        actor_fc_layers=args.actor_fc_layers,
        value_fc_layers=args.value_fc_layers,
        use_rnn=args.use_rnn,
        actor_lstm_size=args.actor_lstm_size,
        value_lstm_size=args.value_lstm_size,
        )