# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 09:14:27 2021

@author: Vladimir Sivak
"""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# append parent 'gkp-rl' directory to path 
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import qutip as qt
from gkp.gkp_tf_env import gkp_init
from remote_env_tools.remote_env_tools import Client

# Create environment that will produce mock measurement outcomes
env = gkp_init(simulate='ECD_control', reward_kwargs={'reward_mode' : 'zero'},
               init='vac', T=8, batch_size=10, N=50, episode_length=8)

# connect to the agent
client_socket = Client()
(host, port) = '172.28.142.46', 5555
client_socket.connect((host, port))

# training loop
done = False
while not done:
    # receive action data from the agent
    message, done = client_socket.recv_data()
    if done: break
    action_batch = message['action_batch']
    mini_buffer = message['mini_buffer']
    N_msmt = message['N_msmt']
    
    # simulate a batch of episodes
    env.reset()
    for t in range(env.T):
        action = {a: action_batch[a][:,t,:] for a in action_batch.keys()}
        time_step = env.step(action)
    
    # collect reward measurements and send them back to the agent
    msmt = env.collect_tomography_measurements(
        'characteristic_fn', mini_buffer, N_msmt=N_msmt)
    client_socket.send_data(msmt)


