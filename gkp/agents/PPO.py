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
from tf_agents.networks import actor_distribution_network, value_network, \
    actor_distribution_rnn_network, value_rnn_network
from tf_agents.utils import common
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.eval import metric_utils
from tf_agents.agents.ppo import ppo_agent
from tf_agents.utils import timer

from gkp.gkp_tf_env import gkp_init
from gkp.gkp_tf_env import tf_env_wrappers as wrappers
from gkp.utils.rl_train_utils import compute_avg_return, save_log
import gkp.action_script as act_scripts

# TODO: find a more elegant way to set 'episode_length' attr of a wrapped env

def random_sample_episode_length(x):
    # sample random episode duration in the range [1..x]
    return np.random.randint(1, x)

def train_eval(
        root_dir = None,
        random_seed = 0,
        # Params for collect
        num_iterations = 1000000,
        train_batch_size = 10,
        replay_buffer_capacity = 20000,
        # Params for train
        normalize_observations = True,
        normalize_rewards = True,
        discount_factor = 1.0,
        lr = 1e-5,
        lr_schedule = None,
        num_policy_epochs = 20,
        initial_adaptive_kl_beta = 0.0,
        kl_cutoff_factor = 0,
        importance_ratio_clipping = 0.2,
        value_pred_loss_coef = 0.5,
        # Params for log, eval, save
        eval_batch_size = 100,
        eval_interval = 100,
        save_interval = 1000,
        checkpoint_interval = None,
        summary_interval = 100,
        # Params for environment
        simulate = 'oscillator',
        horizon = 1,
        clock_period = 4,
        train_episode_length = lambda x: 200,
        eval_episode_length = 200,
        reward_mode = 'pauli',
        encoding = 'square',
        quantum_circuit_type = 'v3',
        action_script = 'Baptiste_8round',
        to_learn = {'alpha':True, 'beta':False, 'epsilon':True, 'phi':False},
        observations_whitelist = None,
        # Policy and value networks
        actor_fc_layers = (),
        value_fc_layers = (),
        use_rnn = True,
        actor_lstm_size = (12,),
        value_lstm_size = (12,)):
    """ A simple train and eval for PPO agent. 
    
    Args:
        root_dir (str): directory for saving training and evalutaion data
        random_seed (int): seed for random number generator
        num_iterations (int): number of training epochs. At each epoch a batch
            of data is collected according to one stochastic policy, and then
            the policy is updated.
        train_batch_size (int): training batch size, collected in parallel.
        replay_buffer_capacity (int): How many transition tuples the buffer 
            can store. The buffer is emptied and re-populated at each epoch.
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
        num_policy_epochs (int): number of policy gradient steps to do on each
            epoch of training. In PPO this is typically >1.
        initial_adaptive_kl_beta (float): see tf-agents PPO docs 
        kl_cutoff_factor (float): see tf-agents PPO docs 
        importance_ratio_clipping (float): clipping value for importance ratio.
            Should demotivate the policy from doing updates that significantly
            change the policy. Should be in (0,1]
        value_pred_loss_coef (float): weight coefficient for quadratic value
            estimation loss.
        eval_batch_size (int): batch size for evaluation of the policy.
        eval_interval (int): interval between evaluations, counted in epochs.
        save_interval (int): interval between savings, counted in epochs. It
            updates the log file and saves the deterministic policy.
        checkpoint_interval (int): interval between saving checkpoints, counted 
            in epochs. Overwrites the previous saved one. Defaults to None, 
            in which case checkpoints are not saved.
        summary_interval (int): interval between summary writing, counted in 
            epochs. tf-agents takes care of summary writing; results can be
            later displayed in tensorboard.
        simulate (str): type of simulator to use in the GKP environment. Can 
            be either 'oscillator' or 'oscillator_qubit'.
        horizon (int): how many past observations to return. Use 1 for RNNs 
            and >=1 for MLP policies. 
        clock_period (int): period for the one-hot encoded clock observation.
        train_episode_length (callable: int -> int): function that defines the 
            schedule for training episode durations. Takes as argument the int 
            epoch number and returns int episode duration for this epoch.
        eval_episode_length (int): duration of evaluation episodes.
        reward_mode (str): see GKP docs for more details.
        encoding (str): GKP encoding, either 'square' or 'hexagonal'
        quantum_circuit_type (str): see GKP docs for more details.
        action_script (str): name of action script, should be compatible with 
            this quantum_circuit_type. Action wrapper will select actions from
            this script if they are not to be learned.
        to_learn (dict, str:bool): dictionary mapping action dimensions to 
            bool flags. Specifies if the action should be learned or scripted.
        observations_whitelist (list, optional): specifies which observations
            should be kept by the FlattenObservationsWrapperTF. Defaults to
            None, meaning that all observations are preserved and flattened.
        actor_fc_layers (tuple): sizes of fully connected layers in actor net.
        value_fc_layers (tuple): sizes of fully connected layers in value net.
        use_rnn (bool): whether to use LSTM units in the neural net.
        actor_lstm_size (tuple): sizes of LSTM layers in actor net.
        value_lstm_size (tuple): sizes of LSTM layers in value net.
    """
    if root_dir is None:
        raise AttributeError('PPO requires a root_dir.')    
    tf.compat.v1.set_random_seed(random_seed)
    action_script = act_scripts.__getattribute__(action_script)
        
    # Create training env and wrap it
    train_env = gkp_init(simulate=simulate,                 
                    init='random', H=horizon, T=clock_period,
                    batch_size=train_batch_size, encoding=encoding,
                    reward_mode=reward_mode, 
                    quantum_circuit_type=quantum_circuit_type)
    
    train_env = wrappers.ActionWrapper(train_env, action_script, to_learn)
    train_env = wrappers.FlattenObservationsWrapperTF(train_env,
                                observations_whitelist=observations_whitelist)

    # Create evaluation env and wrap it
    eval_env = gkp_init(simulate=simulate,
                    init='random', H=horizon, T=clock_period,
                    batch_size=eval_batch_size, encoding=encoding,
                    episode_length=eval_episode_length,
                    reward_mode=reward_mode, 
                    quantum_circuit_type=quantum_circuit_type)
    
    eval_env = wrappers.ActionWrapper(eval_env, action_script, to_learn)
    eval_env = wrappers.FlattenObservationsWrapperTF(eval_env,
                                observations_whitelist=observations_whitelist)

    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    
    # Setup directories within 'root_dir'
    if not os.path.isdir(root_dir): os.mkdir(root_dir)
    policy_dir = os.path.join(root_dir, 'policy')
    checkpoint_dir = os.path.join(root_dir, 'checkpoint')
    logfile = os.path.join(root_dir,'log.hdf5')
    train_dir = os.path.join(root_dir, 'train_summaries')

    # Create tf summary writer
    train_summary_writer = tf.compat.v2.summary.create_file_writer(train_dir)
    train_summary_writer.set_as_default()
    summary_interval *= num_policy_epochs
    global_step = tf.compat.v1.train.get_or_create_global_step()
    with tf.compat.v2.summary.record_if(
            lambda: tf.math.equal(global_step % summary_interval, 0)):

        # Define action and observation specs
        observation_spec = train_env.observation_spec()
        action_spec = train_env.action_spec()
        
        # Define actor network and value network
        if use_rnn:
            actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
                input_tensor_spec = observation_spec,
                output_tensor_spec = action_spec,
                input_fc_layer_params = None,
                lstm_size = actor_lstm_size,
                output_fc_layer_params = actor_fc_layers)
    
            value_net = value_rnn_network.ValueRnnNetwork(
                input_tensor_spec = observation_spec,
                input_fc_layer_params = None,
                lstm_size = value_lstm_size,
                output_fc_layer_params = value_fc_layers)
        else:
            actor_net = actor_distribution_network.ActorDistributionNetwork(
                input_tensor_spec = observation_spec,
                output_tensor_spec = action_spec,
                fc_layer_params = actor_fc_layers)
        
            value_net = value_network.ValueNetwork(
                input_tensor_spec = observation_spec,
                fc_layer_params = value_fc_layers)
    
        # Create PPO agent
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
        tf_agent = ppo_agent.PPOAgent(
            time_step_spec = train_env.time_step_spec(),
            action_spec = action_spec,
            optimizer = optimizer,
            actor_net = actor_net,
            value_net = value_net,
            num_epochs = num_policy_epochs,
            train_step_counter = global_step,
            discount_factor = discount_factor,
            normalize_observations = normalize_observations,
            normalize_rewards = normalize_rewards,
            initial_adaptive_kl_beta = initial_adaptive_kl_beta,
            kl_cutoff_factor = kl_cutoff_factor,
            importance_ratio_clipping = importance_ratio_clipping,
            value_pred_loss_coef = value_pred_loss_coef,
            debug_summaries = True)
        
        tf_agent.initialize()
        eval_policy = tf_agent.policy
        collect_policy = tf_agent.collect_policy
    
        train_metrics = []
    
        # Create replay buffer and collection driver
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=tf_agent.collect_data_spec,
            batch_size=train_batch_size,
            max_length=replay_buffer_capacity)
    
        def train_step():
            experience = replay_buffer.gather_all()
            return tf_agent.train(experience)
    
        tf_agent.train = common.function(tf_agent.train)
    
        collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            train_env,
            collect_policy,
            observers=[replay_buffer.add_batch] + train_metrics,
            num_episodes=train_batch_size
            )
    
        # Create a checkpointer and load the saved agent 
        train_checkpointer = common.Checkpointer(
            ckpt_dir=checkpoint_dir,
            max_to_keep=1,
            agent=tf_agent,
            policy=tf_agent.policy,
            replay_buffer=replay_buffer,
            global_step=global_step,
            metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
        
        train_checkpointer.initialize_or_restore()
        global_step = tf.compat.v1.train.get_global_step()
    
        # Saver for the deterministic policy
        saved_model = policy_saver.PolicySaver(
            eval_policy, train_step=global_step)
    
        # Evaluate policy once before training
        avg_return = compute_avg_return(eval_env, tf_agent.policy)
        log = {
            'returns' : [avg_return],
            'epochs' : [global_step.numpy()/num_policy_epochs],
            'policy_steps' : [global_step.numpy()],
            'experience_time' : [0.0],
            'train_time' : [0.0]
            }
        
        # Save initial random policy
        global_step_val = global_step.numpy()
        path = os.path.join(policy_dir,('%d' % global_step_val).zfill(9))
        saved_model.save(path)
    
        # Training loop
        train_timer = timer.Timer()
        experience_timer = timer.Timer()
        for epoch in range(num_iterations):
            # Collect new experience
            experience_timer.start()
            train_env._env._env.episode_length = train_episode_length(epoch)
            collect_driver.run()
            experience_timer.stop()
            # Update the policy 
            train_timer.start()
            if lr_schedule: optimizer._lr = lr_schedule(epoch)
            train_loss = train_step()
            replay_buffer.clear()
            train_timer.stop()
            
            # Log and evaluate everything
            global_step_val = global_step.numpy()
            
            if global_step_val % (eval_interval*num_policy_epochs) == 0:
                # Evaluate the policy
                avg_return = compute_avg_return(eval_env, tf_agent.policy)
                # Print out and log all metrics
                print('-------------------')
                print('Epoch %d' %(global_step_val/num_policy_epochs))
                print('  Policy steps: %d' %global_step_val)
                print('  Experience time: %.2f mins' %(experience_timer.value()/60))
                print('  Policy train time: %.2f mins' %(train_timer.value()/60))
                print('  Average return: %.2f' %avg_return)
                log['epochs'].append(global_step_val/num_policy_epochs)
                log['policy_steps'].append(global_step_val)
                log['returns'].append(avg_return)
                log['experience_time'].append(experience_timer.value())
                log['train_time'].append(train_timer.value())
                
            if global_step_val % (save_interval*num_policy_epochs) == 0:
                # Save deterministic policy
                path = os.path.join(policy_dir,('%d' % global_step_val).zfill(9))
                saved_model.save(path)
                # Save updated log
                save_log(log, logfile, ('%d' % global_step_val).zfill(9))
            
            if checkpoint_interval is not None and \
                global_step_val % (checkpoint_interval*num_policy_epochs) == 0:
                    # Save training checkpoint
                    train_checkpointer.save(global_step)
                    