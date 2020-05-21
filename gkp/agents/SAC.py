# -*- coding: utf-8 -*-
"""
Created on Tue May 19 2020

@author: Vladimir Sivak
"""
import os
import tensorflow as tf

from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import policy_saver
from tf_agents.networks import actor_distribution_network,\
    actor_distribution_rnn_network
from tf_agents.agents.ddpg import critic_rnn_network, critic_network
from tf_agents.utils import common
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.eval import metric_utils
from tf_agents.agents.sac import sac_agent
from tf_agents.utils import timer
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.policies import greedy_policy
from tf_agents.metrics import tf_metrics

from gkp.gkp_tf_env import gkp_init
from gkp.gkp_tf_env import tf_env_wrappers as wrappers
from gkp.utils.rl_train_utils import compute_avg_return, save_log
import gkp.action_script as act_scripts

def train_eval(
        root_dir = None,
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
        td_errors_loss_fn = tf.math.squared_difference,
        reward_scale_factor = 0.1,
        gradient_clipping = None,
        discount_factor = 1.0,
        critic_learning_rate = 1e-4,
        actor_learning_rate = 1e-4,
        alpha_learning_rate = 1e4,
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
        reward_mode = 'pauli',
        quantum_circuit_type = 'v1',
        action_script = 'phase_estimation_4round',
        to_learn = {'alpha':True, 'beta':False, 'phi':False},
        # Actor and critic networks
        actor_fc_layers = (),
        actor_output_fc_layers = (),
        critic_obs_fc_layers = None,
        critic_action_fc_layers = None,
        critic_joint_fc_layers = (),
        critic_output_fc_layers = (),
        use_rnn = True,
        actor_lstm_size = (12,),
        critic_lstm_size = (12,)):
    """ A simple train and eval for SAC agent. """

    if root_dir is None:
        raise AttributeError('SAC requires a root_dir.')    
    tf.compat.v1.set_random_seed(random_seed)
    action_script = act_scripts.__getattribute__(action_script)
        
    # Create training env
    train_env = gkp_init(simulate=simulate,                 
                    init='random', H=horizon, batch_size=train_batch_size,
                    max_episode_length=max_episode_length,
                    reward_mode=reward_mode, 
                    quantum_circuit_type=quantum_circuit_type)
    
    train_env = wrappers.ActionWrapper(train_env, action_script, to_learn)
    train_env = wrappers.FlattenObservationsWrapperTF(train_env)

    # Create evaluation env    
    eval_env = gkp_init(simulate=simulate,
                    init='random', H=horizon, batch_size=eval_batch_size,
                    episode_length=eval_episode_length,
                    reward_mode=reward_mode, 
                    quantum_circuit_type=quantum_circuit_type)
    
    eval_env = wrappers.ActionWrapper(eval_env, action_script, to_learn)
    eval_env = wrappers.FlattenObservationsWrapperTF(eval_env)

    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    
    if not os.path.isdir(root_dir): os.mkdir(root_dir)
    policy_dir = os.path.join(root_dir, 'policy')
    checkpoint_dir = os.path.join(root_dir, 'checkpoint')
    logfile = os.path.join(root_dir,'log.hdf5')

    # Define action and observation specs
    observation_spec = train_env.observation_spec()
    action_spec = train_env.action_spec()
    
    # Define actor network and value network
    if use_rnn:
        actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
            input_tensor_spec = observation_spec,
            output_tensor_spec = action_spec,
            input_fc_layer_params = actor_fc_layers,
            lstm_size = actor_lstm_size,
            output_fc_layer_params = actor_output_fc_layers,
            continuous_projection_net = tanh_normal_projection_network
            .TanhNormalProjectionNetwork)

        critic_net = critic_rnn_network.CriticRnnNetwork(
            (observation_spec, action_spec),
            observation_fc_layer_params = critic_obs_fc_layers,
            action_fc_layer_params = critic_action_fc_layers,
            joint_fc_layer_params = critic_joint_fc_layers,
            lstm_size = critic_lstm_size,
            output_fc_layer_params = critic_output_fc_layers,
            kernel_initializer = 'glorot_uniform',
            last_kernel_initializer = 'glorot_uniform')
    else:
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            input_tensor_spec = observation_spec,
            output_tensor_spec = action_spec,
            fc_layer_params = actor_fc_layers,
            continuous_projection_net = tanh_normal_projection_network
            .TanhNormalProjectionNetwork)

        critic_net = critic_network.CriticNetwork(
            (observation_spec, action_spec),
            observation_fc_layer_params = critic_obs_fc_layers,
            action_fc_layer_params = critic_action_fc_layers,
            joint_fc_layer_params = critic_joint_fc_layers,
            kernel_initializer = 'glorot_uniform',
            last_kernel_initializer = 'glorot_uniform')

    # Create a SAC agent
    global_step = tf.compat.v1.train.get_or_create_global_step()
    tf_agent = sac_agent.SacAgent(
        time_step_spec = train_env.time_step_spec(),
        action_spec = action_spec,
        actor_network = actor_net,
        critic_network = critic_net,
        actor_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=actor_learning_rate),
        critic_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=critic_learning_rate),
        alpha_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=alpha_learning_rate),
        target_update_tau = target_update_tau,
        target_update_period = target_update_period,
        td_errors_loss_fn = td_errors_loss_fn,
        gamma = discount_factor,
        reward_scale_factor = reward_scale_factor,
        gradient_clipping = gradient_clipping,
        train_step_counter=global_step)
    
    tf_agent.initialize()
    eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
    collect_policy = tf_agent.collect_policy

    # Create replay buffer
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec = tf_agent.collect_data_spec,
        batch_size = train_batch_size,
        max_length = replay_buffer_capacity)


    env_steps = tf_metrics.EnvironmentSteps(prefix='Train')
    train_metrics = [env_steps]
    tf_agent.train = common.function(tf_agent.train)

    # Create a checkpointer and load the saved agent 
    train_checkpointer = common.Checkpointer(
        ckpt_dir = checkpoint_dir,
        max_to_keep = 1,
        agent = tf_agent,
        policy = tf_agent.policy,
        replay_buffer = replay_buffer,
        global_step = global_step,
        metrics = metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
    
    train_checkpointer.initialize_or_restore()
    global_step = tf.compat.v1.train.get_global_step()

    # Saver for the policy
    saved_model = policy_saver.PolicySaver(
        eval_policy, train_step=global_step)

    # Create the collection driver
    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        train_env,
        collect_policy,
        observers = [replay_buffer.add_batch] + train_metrics,
        num_episodes = train_batch_size)

    # Collect initial replay data
    if env_steps.result() == 0 or replay_buffer.num_frames() == 0:
        collect_driver.run()


    #! Not sure how this part works, I copied it from SAC example in tf-agents
    # Prepare replay buffer as dataset with invalid transitions filtered.
    # def _filter_invalid_transition(trajectories, unused_arg1):
    #     # Reduce filter_fn over full trajectory sampled. The sequence is kept
    #     # if all elements except for the last one pass the filter. This is to
    #     # allow training on terminal steps.
    #     return tf.reduce_all(~trajectories.is_boundary()[:-1])
    # dataset = replay_buffer.as_dataset(
    #     sample_batch_size=train_batch_size,
    #     num_steps=train_sequence_length+1).unbatch().filter(
    #         _filter_invalid_transition).batch(train_batch_size).prefetch(5)

    dataset = replay_buffer.as_dataset(#num_parallel_calls=3, 
        sample_batch_size=train_batch_size, num_steps=2).prefetch(3)
            
    # Dataset generates trajectories with shape [Bx2x...]
    iterator = iter(dataset)

    def train_step():
        experience, _ = next(iterator)
        return tf_agent.train(experience)

    # Evaluate policy once before training
    avg_return = compute_avg_return(eval_env, tf_agent.policy)
    log = {
        'returns' : [avg_return],
        'epochs' : [global_step.numpy()/train_steps_per_iteration],
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
    for _ in range(num_iterations):
        # Collect new experience
        start_env_steps = env_steps.result()
        experience_timer.start()
        collect_driver.run()
        experience_timer.stop()
        # Update the policy 
        episode_steps = env_steps.result() - start_env_steps
        train_timer.start()
        for _ in range(episode_steps):
          for _ in range(train_steps_per_iteration):
            train_loss = train_step()
        train_timer.stop()
        
        # Log and evaluate everything
        global_step_val = global_step.numpy()
        
        if global_step_val % (log_interval * train_steps_per_iteration) == 0:
            print('-------------------')
            print('Epoch %d' %(global_step_val / train_steps_per_iteration))
            print('  Loss: %.2f' %train_loss.loss)
            print('  Experience time: %.2f mins' %(experience_timer.value()/60))
            print('  Policy train time: %.2f mins' %(train_timer.value()/60))    
            
        if global_step_val % (eval_interval * train_steps_per_iteration) == 0:
            # Evaluate the policy
            avg_return = compute_avg_return(eval_env, eval_policy)
            print('  Policy steps: ' %global_step_val)
            print('  Average return: %.2f' %avg_return)
            # Cache all metrics
            log['epochs'].append(global_step_val / train_steps_per_iteration)
            log['policy_steps'].append(global_step_val)
            log['returns'].append(avg_return)
            log['experience_time'].append(experience_timer.value())
            log['train_time'].append(train_timer.value())
            
        if global_step_val % (save_interval * train_steps_per_iteration) == 0:
            # Save training
            train_checkpointer.save(global_step)
            # Save policy
            path = os.path.join(policy_dir,('%d' % global_step_val).zfill(9))
            saved_model.save(path)
            # Save all metrics
            save_log(log, logfile, ('%d' % global_step_val).zfill(9))

    
    