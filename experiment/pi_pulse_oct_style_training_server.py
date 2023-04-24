import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# append parent 'gkp-rl' directory to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import tensorflow as tf
from tf_agents import specs
from rl_tools.agents import PPO
from tf_agents.networks import actor_distribution_network
from rl_tools.remote_env_tools import remote_env_tools as rmt

import numpy as np

root_dir = r'D:\rl_data\exp_training\pi_pulse_oct_style_1'

server_socket = rmt.Server()
# Vlad's old PC: 172.28.142.46
# DN new msmt PC: 172.28.143.183
# DN new msmt PC (local): 169.168.1.1
(host, port) = ('169.168.1.1', 5555)
server_socket.bind((host, port))
server_socket.connect_client()

# Params for environment
env_kwargs = eval_env_kwargs = {
  'T' : 1}

# Params for reward function
reward_kwargs = {
    'reward_mode' : 'remote',
    'server_socket' : server_socket,
    'epoch_type' : 'training',
    'N_msmt' : 500}

reward_kwargs_eval = {
    'reward_mode' : 'remote',
    'server_socket' : server_socket,
    'epoch_type' : 'evaluation',
    'N_msmt' : 5000}

# setting up initial pulse
n_array_vals = 40 # number of array values to optimize
sigma = 10 #initial guess for sigma of pulse
amp = 0.4 #initial guess for amp of pulse
ts = np.linspace(-n_array_vals/2, n_array_vals/2, n_array_vals)
P = np.exp(-ts**2 / (2.0 * sigma**2))
ofs = P[0]
init_pulse = list(amp*((P - ofs) / (1 - ofs)))

action_scale_array = list(np.ones(n_array_vals,dtype=float))

# Params for action wrapper
action_script = {
  'pulse_array' : [init_pulse] # shape=[n_array_vals]
  }

action_spec = {
  'pulse_array' : specs.TensorSpec(shape=[n_array_vals], dtype=tf.float32)
  }

action_scale = {
  'pulse_array':action_scale_array
  }

to_learn = {
  'pulse_array':True
  }

train_batch_size = 50
eval_batch_size = 1

learn_residuals = True

# Create drivers for data collection
from rl_tools.agents import dynamic_episode_driver_sim_env

collect_driver = dynamic_episode_driver_sim_env.DynamicEpisodeDriverSimEnv(
    env_kwargs, reward_kwargs, train_batch_size, action_script, action_scale,
    action_spec, to_learn, learn_residuals, remote=True)

eval_driver = dynamic_episode_driver_sim_env.DynamicEpisodeDriverSimEnv(
    eval_env_kwargs, reward_kwargs_eval, eval_batch_size, action_script, action_scale,
    action_spec, to_learn, learn_residuals, remote=True)

PPO.train_eval(
    root_dir = root_dir,
    random_seed = 0,
    num_epochs = 100,
    # Params for train
    normalize_observations = True,
    normalize_rewards = False,
    discount_factor = 1.0,
    lr = 2.5e-3,
    lr_schedule = None,
    num_policy_updates = 20,
    initial_adaptive_kl_beta = 0.0,
    kl_cutoff_factor = 0,
    importance_ratio_clipping = 0.1,
    value_pred_loss_coef = 0.005,
    gradient_clipping = 1.0,
    entropy_regularization = 0,
    log_prob_clipping = 0.0,
    # Params for log, eval, save
    eval_interval = 20,
    save_interval = 2,
    checkpoint_interval = None,
    summary_interval = 2,
    do_evaluation = True,
    # Params for data collection
    train_batch_size = train_batch_size,
    eval_batch_size = eval_batch_size,
    collect_driver = collect_driver,
    eval_driver = eval_driver,
    replay_buffer_capacity = 15000,
    # Policy and value networks
    ActorNet = actor_distribution_network.ActorDistributionNetwork,
    zero_means_kernel_initializer = False,
    init_action_stddev = 0.08,
    actor_fc_layers = (50,20),
    value_fc_layers = (),
    use_rnn = False,
    actor_lstm_size = (12,),
    value_lstm_size = (12,))
