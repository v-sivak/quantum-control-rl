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

    root_dir = r'E:\VladGoogleDrive\Qulab\GKP\sims\PPO\June\OscillatorGKP\mlp3_steps48_H1T4_lr1e-5_new_clock_PPO_v2'
    random_seed = 0
    # Params for collect
    num_iterations = 1000000
    train_batch_size = 1000
    replay_buffer_capacity = 50000
    # Params for train
    normalize_observations = True
    normalize_rewards = True
    discount_factor = 1.0
    lr = 1e-5
    lr_schedule = None
    num_policy_epochs = 20
    initial_adaptive_kl_beta = 0.0
    kl_cutoff_factor = 0
    importance_ratio_clipping = 0.2
    # Params for log, eval, save
    eval_batch_size = 200
    eval_interval = 100
    save_interval = 1000
    log_interval = 100
    # Params for environment
    simulate = 'oscillator'
    horizon = 1
    clock_period = 4
    max_episode_length = 48
    eval_episode_length = 48
    reward_mode = 'pauli'
    quantum_circuit_type = 'v2'
    action_script = 'phase_estimation_symmetric_with_trim_4round'
    to_learn = {'alpha':True, 'beta':True, 'phi':True}
    observations_whitelist = ['msmt', 'clock']
    # Policy and value networks
    actor_fc_layers = (200,100,50)
    value_fc_layers = (200,100,50)
    use_rnn = False
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

    ### Params for log, eval, save

    parser.add_argument('--eval_batch_size', type=int, 
                        default=eval_batch_size)

    parser.add_argument('--eval_interval', type=int, 
                        default=eval_interval)

    parser.add_argument('--save_interval', type=int, 
                        default=save_interval)
    
    parser.add_argument('--log_interval', type=int, 
                        default=log_interval)

    ### Params for environment
    
    parser.add_argument('--simulate', type=str, 
                        default=simulate)
    
    parser.add_argument('--horizon', type=int, 
                        default=horizon)

    parser.add_argument('--clock_period', type=int, 
                        default=clock_period)
    
    parser.add_argument('--max_episode_length', type=int, 
                        default=max_episode_length)
    
    parser.add_argument('--eval_episode_length', type=int, 
                        default=eval_episode_length)
    
    parser.add_argument('--reward_mode', type=str,
                        default=reward_mode)
    
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
        eval_batch_size=args.eval_batch_size,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        simulate=args.simulate,
        horizon=args.horizon,
        clock_period=args.clock_period,
        max_episode_length=args.max_episode_length,
        eval_episode_length=args.eval_episode_length,
        reward_mode=args.reward_mode,
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