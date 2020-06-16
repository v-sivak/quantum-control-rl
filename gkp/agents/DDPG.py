# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:19:36 2020

@author: Vladimir Sivak
"""
import os
import tensorflow as tf

from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import policy_saver
from tf_agents.agents.ddpg import actor_network, critic_network
from tf_agents.utils import common
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.eval import metric_utils
from tf_agents.agents.ddpg import ddpg_agent
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
        replay_buffer_capacity = 100000,
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
        critic_learning_rate = 1e-4,
        actor_learning_rate = 1e-4,
        train_steps_per_iteration = 1,
        # Params for log, eval, save
        eval_batch_size = 200,
        eval_interval = 100,
        save_interval = 1000,
        log_interval = 100,
        # Params for environment
        simulate = 'oscillator',
        horizon = 4,
        max_episode_length = 24,
        eval_episode_length = 24,
        reward_mode = 'pauli',
        quantum_circuit_type = 'v2',
        action_script = 'phase_estimation_symmetric_with_trim_4round',
        to_learn = {'alpha':True, 'beta':True, 'phi':True},
        observations_whitelist = ['msmt','clock'],
        # Actor and critic networks
        actor_fc_layers = (200,100),
        actor_output_fc_layers = (),
        critic_obs_fc_layers = (200,),
        critic_action_fc_layers = (200,),
        critic_joint_fc_layers = (100,),
        critic_output_fc_layers = (),
        use_rnn = False,
        actor_lstm_size = (12,),
        critic_lstm_size = (12,)):
    """ A simple train and eval for DDPG agent. """

    if root_dir is None:
        raise AttributeError('DDPG requires a root_dir.')    
    tf.compat.v1.set_random_seed(random_seed)
    action_script = act_scripts.__getattribute__(action_script)
        
    # Create training env
    train_env = gkp_init(simulate=simulate,                 
                    init='random', H=horizon, batch_size=train_batch_size,
                    max_episode_length=max_episode_length,
                    reward_mode=reward_mode, 
                    quantum_circuit_type=quantum_circuit_type)
    
    train_env = wrappers.ActionWrapper(train_env, action_script, to_learn)
    train_env = wrappers.FlattenObservationsWrapperTF(train_env,
                                observations_whitelist=observations_whitelist)

    # Create evaluation env    
    eval_env = gkp_init(simulate=simulate,
                    init='random', H=horizon, batch_size=eval_batch_size,
                    episode_length=eval_episode_length,
                    reward_mode=reward_mode, 
                    quantum_circuit_type=quantum_circuit_type)
    
    eval_env = wrappers.ActionWrapper(eval_env, action_script, to_learn)
    eval_env = wrappers.FlattenObservationsWrapperTF(eval_env,
                                observations_whitelist=observations_whitelist)

    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    
    if not os.path.isdir(root_dir): os.mkdir(root_dir)
    policy_dir = os.path.join(root_dir, 'policy')
    checkpoint_dir = os.path.join(root_dir, 'checkpoint')
    logfile = os.path.join(root_dir,'log.hdf5')

    # Define action and observation specs
    observation_spec = train_env.observation_spec()
    action_spec = train_env.action_spec()
    
    # Define actor network and critic network
    if use_rnn:
        actor_net = actor_rnn_network.ActorRnnNetwork(
            input_tensor_spec = observation_spec,
            output_tensor_spec = action_spec,
            input_fc_layer_params = actor_fc_layers,
            lstm_size = actor_lstm_size,
            output_fc_layer_params = actor_output_fc_layers)

        critic_net = critic_rnn_network.CriticRnnNetwork(
            (observation_spec, action_spec),
            observation_fc_layer_params = critic_obs_fc_layers,
            action_fc_layer_params = critic_action_fc_layers,
            joint_fc_layer_params = critic_joint_fc_layers,
            lstm_size = critic_lstm_size,
            output_fc_layer_params = critic_output_fc_layers)
    else:
        actor_net = actor_network.ActorNetwork(
            input_tensor_spec = observation_spec,
            output_tensor_spec = action_spec,
            fc_layer_params = actor_fc_layers)

        critic_net = critic_network.CriticNetwork(
            (observation_spec, action_spec),
            observation_fc_layer_params = critic_obs_fc_layers,
            action_fc_layer_params = critic_action_fc_layers,
            joint_fc_layer_params = critic_joint_fc_layers)

    # Create a DDPG agent
    global_step = tf.compat.v1.train.get_or_create_global_step()

    tf_agent = ddpg_agent.DdpgAgent(
        time_step_spec = train_env.time_step_spec(),
        action_spec = action_spec,
        actor_network = actor_net,
        critic_network = critic_net,
        actor_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=actor_learning_rate),
        critic_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=critic_learning_rate),
        ou_stddev = ou_stddev,
        ou_damping = ou_damping,
        target_update_tau = target_update_tau,
        target_update_period = target_update_period,
        dqda_clipping = dqda_clipping,
        td_errors_loss_fn = tf.math.squared_difference,
        gamma = discount_factor,
        reward_scale_factor = reward_scale_factor,
        gradient_clipping = gradient_clipping,
        train_step_counter=global_step)
    
    tf_agent.initialize()
    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy

    train_metrics = []

    # Create replay buffer
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec = tf_agent.collect_data_spec,
        batch_size = train_batch_size,
        max_length = replay_buffer_capacity)

    tf_agent.train = common.function(tf_agent.train)

    # TODO: change this to dynamic step driver as in DDPG example???

    # Create the collection driver
    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver( 
        train_env,
        collect_policy,
        observers = [replay_buffer.add_batch] + train_metrics,
        num_episodes = train_batch_size)

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

    # Collect initial replay data
    if global_step.numpy() == 0 or replay_buffer.num_frames() == 0:
        collect_driver.run()
        
    # Dataset generates trajectories with shape [Bx2x...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, 
        sample_batch_size=train_batch_size, 
        num_steps=2).prefetch(3)
    iterator = iter(dataset)

    def train_step():
        experience, _ = next(iterator)
        return tf_agent.train(experience)

    # Evaluate policy once before training
    avg_return = compute_avg_return(eval_env, eval_policy)
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
        experience_timer.start()
        collect_driver.run()
        experience_timer.stop()
        # Update the policy 
        train_timer.start()
        for __ in range(train_steps_per_iteration):
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

    
    