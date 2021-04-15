# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 12:24:37 2020

@author: Vladimir Sivak
"""
import os
import tensorflow as tf
import numpy as np

from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import policy_saver
from tf_agents.networks import actor_distribution_network, value_network
from tf_agents.networks import actor_distribution_rnn_network, value_rnn_network
from tf_agents.utils import common
from tf_agents.eval import metric_utils
from tf_agents.agents.ppo import ppo_agent
from tf_agents.utils import timer
from tf_agents.metrics import tf_metrics

from gkp.utils.rl_train_utils import save_log

def train_eval(
        root_dir,
        random_seed = 0,
        num_epochs = 1000000,
        # Params for train
        normalize_observations = True,
        normalize_rewards = True,
        discount_factor = 1.0,
        lr = 1e-5,
        lr_schedule = None,
        num_policy_updates = 20,
        initial_adaptive_kl_beta = 0.0,
        kl_cutoff_factor = 0,
        importance_ratio_clipping = 0.2,
        value_pred_loss_coef = 0.5,
        gradient_clipping = None,
        entropy_regularization = 0.0,
        # Params for log, eval, save
        eval_interval = 100,
        save_interval = 1000,
        checkpoint_interval = None,
        summary_interval = 100,
        do_evaluation = True,
        # Params for data collection
        train_batch_size = 10,
        eval_batch_size = 100,
        collect_driver = None,
        eval_driver = None,
        replay_buffer_capacity = 20000,
        # Policy and value networks
        ActorNet = actor_distribution_network.ActorDistributionNetwork,
        zero_means_kernel_initializer = False,
        actor_fc_layers = (),
        value_fc_layers = (),
        use_rnn = True,
        actor_lstm_size = (12,),
        value_lstm_size = (12,),
        **kwargs):
    """ A simple train and eval for PPO agent. 
    
    Args:
        root_dir (str): directory for saving training and evalutaion data
        random_seed (int): seed for random number generator
        num_epochs (int): number of training epochs. At each epoch a batch
            of data is collected according to one stochastic policy, and then
            the policy is updated.
        normalize_observations (bool): flag for normalization of observations.
            Uses StreamingTensorNormalizer which normalizes based on the whole
            history of observations.
        normalize_rewards (bool): flag for normalization of rewards.
            Uses StreamingTensorNormalizer which normalizes based on the whole
            history of rewards.
        discount_factor (float): rewards discout factor, should be in (0,1]
        lr (float): learning rate for Adam optimizer
        lr_schedule (callable: int -> float, optional): function to schedule 
            the learning rate annealing. Takes as argument the int epoch
            number and returns float value of the learning rate. 
        num_policy_updates (int): number of policy gradient steps to do on each
            epoch of training. In PPO this is typically >1.
        initial_adaptive_kl_beta (float): see tf-agents PPO docs 
        kl_cutoff_factor (float): see tf-agents PPO docs 
        importance_ratio_clipping (float): clipping value for importance ratio.
            Should demotivate the policy from doing updates that significantly
            change the policy. Should be in (0,1]
        value_pred_loss_coef (float): weight coefficient for quadratic value
            estimation loss.
        gradient_clipping (float): gradient clipping coefficient.
        entropy_regularization (float): entropy regularization loss coefficient.
        eval_interval (int): interval between evaluations, counted in epochs.
        save_interval (int): interval between savings, counted in epochs. It
            updates the log file and saves the deterministic policy.
        checkpoint_interval (int): interval between saving checkpoints, counted 
            in epochs. Overwrites the previous saved one. Defaults to None, 
            in which case checkpoints are not saved.
        summary_interval (int): interval between summary writing, counted in 
            epochs. tf-agents takes care of summary writing; results can be
            later displayed in tensorboard.
        do_evaluation (bool): flag to interleave training epochs with 
            evaluation epochs.
        train_batch_size (int): training batch size, collected in parallel.
        eval_batch_size (int): batch size for evaluation of the policy.
        collect_driver (Driver): driver for training data collection
        eval_driver (Driver): driver for evaluation data collection
        replay_buffer_capacity (int): How many transition tuples the buffer 
            can store. The buffer is emptied and re-populated at each epoch.
        ActorNet (network.DistributionNetwork): a distribution actor network 
            to use for training. The default is ActorDistributionNetwork from
            tf-agents, but this can also be customized.
        zero_means_kernel_initializer (bool): flag to initialize the means
            projection network with zeros. If this flag is not set, it will
            use default tf-agent random initializer.
        actor_fc_layers (tuple): sizes of fully connected layers in actor net.
        value_fc_layers (tuple): sizes of fully connected layers in value net.
        use_rnn (bool): whether to use LSTM units in the neural net.
        actor_lstm_size (tuple): sizes of LSTM layers in actor net.
        value_lstm_size (tuple): sizes of LSTM layers in value net.
    """
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    tf.compat.v1.set_random_seed(random_seed)
    
    # Setup directories within 'root_dir'
    if not os.path.isdir(root_dir): os.mkdir(root_dir)
    policy_dir = os.path.join(root_dir, 'policy')
    checkpoint_dir = os.path.join(root_dir, 'checkpoint')
    logfile = os.path.join(root_dir,'log.hdf5')
    train_dir = os.path.join(root_dir, 'train_summaries')

    # Create tf summary writer
    train_summary_writer = tf.compat.v2.summary.create_file_writer(train_dir)
    train_summary_writer.set_as_default()
    summary_interval *= num_policy_updates
    global_step = tf.compat.v1.train.get_or_create_global_step()
    with tf.compat.v2.summary.record_if(
            lambda: tf.math.equal(global_step % summary_interval, 0)):

        # Define action and observation specs
        observation_spec = collect_driver.observation_spec()
        action_spec = collect_driver.action_spec()
        
        # Preprocessing: flatten and concatenate observation components
        preprocessing_layers = {
            obs : tf.keras.layers.Flatten() for obs in observation_spec.keys()}
        preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)
        
        # Define actor network and value network
        if use_rnn:
            actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
                input_tensor_spec = observation_spec,
                output_tensor_spec = action_spec,
                preprocessing_layers = preprocessing_layers,
                preprocessing_combiner = preprocessing_combiner,
                input_fc_layer_params = None,
                lstm_size = actor_lstm_size,
                output_fc_layer_params = actor_fc_layers)
    
            value_net = value_rnn_network.ValueRnnNetwork(
                input_tensor_spec = observation_spec,
                preprocessing_layers = preprocessing_layers,
                preprocessing_combiner = preprocessing_combiner,
                input_fc_layer_params = None,
                lstm_size = value_lstm_size,
                output_fc_layer_params = value_fc_layers)
        else:
            npn = actor_distribution_network._normal_projection_net
            normal_projection_net = lambda specs: npn(specs, 
                zero_means_kernel_initializer=zero_means_kernel_initializer)
            
            actor_net = ActorNet(
                input_tensor_spec = observation_spec,
                output_tensor_spec = action_spec,
                preprocessing_layers = preprocessing_layers,
                preprocessing_combiner = preprocessing_combiner,
                fc_layer_params = actor_fc_layers,
                continuous_projection_net=normal_projection_net)
        
            value_net = value_network.ValueNetwork(
                input_tensor_spec = observation_spec,
                preprocessing_layers = preprocessing_layers,
                preprocessing_combiner = preprocessing_combiner,
                fc_layer_params = value_fc_layers)
    
        # Create PPO agent
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
        tf_agent = ppo_agent.PPOAgent(
            time_step_spec = collect_driver.time_step_spec(),
            action_spec = action_spec,
            optimizer = optimizer,
            actor_net = actor_net,
            value_net = value_net,
            num_epochs = num_policy_updates,
            train_step_counter = global_step,
            discount_factor = discount_factor,
            normalize_observations = normalize_observations,
            normalize_rewards = normalize_rewards,
            initial_adaptive_kl_beta = initial_adaptive_kl_beta,
            kl_cutoff_factor = kl_cutoff_factor,
            importance_ratio_clipping = importance_ratio_clipping,
            gradient_clipping = gradient_clipping,
            value_pred_loss_coef = value_pred_loss_coef,
            entropy_regularization=entropy_regularization,
            debug_summaries = True)
        
        tf_agent.initialize()
        eval_policy = tf_agent.policy
        collect_policy = tf_agent.collect_policy
    
        # Create replay buffer and collection driver
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=tf_agent.collect_data_spec,
            batch_size=train_batch_size,
            max_length=replay_buffer_capacity)
    
        def train_step():
            experience = replay_buffer.gather_all()
            return tf_agent.train(experience)
    
        tf_agent.train = common.function(tf_agent.train)

        avg_return_metric = tf_metrics.AverageReturnMetric(
            batch_size=eval_batch_size, buffer_size=eval_batch_size)
        
        collect_driver.setup(collect_policy, [replay_buffer.add_batch])
        eval_driver.setup(eval_policy, [avg_return_metric])
    
        # Create a checkpointer and load the saved agent 
        train_checkpointer = common.Checkpointer(
            ckpt_dir=checkpoint_dir,
            max_to_keep=1,
            agent=tf_agent,
            policy=tf_agent.policy,
            replay_buffer=replay_buffer,
            global_step=global_step)
        
        train_checkpointer.initialize_or_restore()
        global_step = tf.compat.v1.train.get_global_step()
    
        # Saver for the deterministic policy
        saved_model = policy_saver.PolicySaver(
            eval_policy, train_step=global_step)
    
        # Evaluate policy once before training
        if do_evaluation:
            eval_driver.run(0)
            avg_return = avg_return_metric.result().numpy()
            avg_return_metric.reset()
            log = {
                'returns' : [avg_return],
                'epochs' : [0],
                'policy_steps' : [0],
                'experience_time' : [0.0],
                'train_time' : [0.0]
                }
            print('-------------------')
            print('Epoch 0')
            print('  Policy steps: 0')
            print('  Experience time: 0.00 mins')
            print('  Policy train time: 0.00 mins')
            print('  Average return: %.5f' %avg_return)
        
        # Save initial random policy
        path = os.path.join(policy_dir,('0').zfill(6))
        saved_model.save(path)
    
        # Training loop
        train_timer = timer.Timer()
        experience_timer = timer.Timer()
        for epoch in range(1,num_epochs+1):
            # Collect new experience
            experience_timer.start()
            collect_driver.run(epoch)
            experience_timer.stop()
            # Update the policy 
            train_timer.start()
            if lr_schedule: optimizer._lr = lr_schedule(epoch)
            train_loss = train_step()
            replay_buffer.clear()
            train_timer.stop()
            
            if (epoch % eval_interval == 0) and do_evaluation:
                # Evaluate the policy
                eval_driver.run(epoch)
                avg_return = avg_return_metric.result().numpy()
                avg_return_metric.reset()
                
                # Print out and log all metrics
                print('-------------------')
                print('Epoch %d' %epoch)
                print('  Policy steps: %d' %(epoch*num_policy_updates))
                print('  Experience time: %.2f mins' %(experience_timer.value()/60))
                print('  Policy train time: %.2f mins' %(train_timer.value()/60))
                print('  Average return: %.5f' %avg_return)
                log['epochs'].append(epoch)
                log['policy_steps'].append(epoch*num_policy_updates)
                log['returns'].append(avg_return)
                log['experience_time'].append(experience_timer.value())
                log['train_time'].append(train_timer.value())
                # Save updated log
                save_log(log, logfile, ('%d' % epoch).zfill(6))
                
            if epoch % save_interval == 0:
                # Save deterministic policy
                path = os.path.join(policy_dir,('%d' % epoch).zfill(6))
                saved_model.save(path)
            
            if checkpoint_interval is not None and \
                epoch % checkpoint_interval == 0:
                    # Save training checkpoint
                    train_checkpointer.save(global_step)
        collect_driver.finish_training()
        eval_driver.finish_training()
