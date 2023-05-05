

# Author: Ben Brock 
# Created on May 02, 2023 

#%%
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
# add quantum-control-rl dir to path for subsequent imports
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import logging
import time
logger = logging.getLogger('RL')
logger.propagate = False
logger.handlers = []
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

from rl_tools.remote_env_tools.remote_env_tools import Client

from examples.pi_pulse_oct_style.pi_pulse_oct_style_sim_function import pi_pulse_oct_style_sim

client_socket = Client()
(host, port) = '127.0.0.1', 5555 # ip address of RL server, here it's hosted locally
client_socket.connect((host, port))

# training loop
done = False
while not done:

    # receive action data from the agent (see tf_env -> reward_remote())
    message, done = client_socket.recv_data()
    logger.info('Received message from RL agent server.')
    logger.info('Time stamp: %f' %time.time())

    if done:
        logger.info('Training finished.')
        break

    # parsing message (see tf_env -> reward_remote())
    action_batch = message['action_batch']
    batch_size = message['batch_size']
    epoch_type = message['epoch_type']
    epoch = message['epoch']

    # parsing action_batch and reshaping to get rid of nested
    # structure required by tensorflow
    # here env.T=1 so the shape is (batch_size,1,pulse_len)
    new_shape_list_real = list(action_batch['pulse_array_real'].shape)
    new_shape_list_imag = list(action_batch['pulse_array_imag'].shape)
    new_shape_list_real.pop(1)
    new_shape_list_imag.pop(1)
    real_pulses = action_batch['pulse_array_real'].reshape(new_shape_list_real)
    imag_pulses = action_batch['pulse_array_imag'].reshape(new_shape_list_imag)


    logger.info('Start %s epoch %d' %(epoch_type, epoch))

    # collecting rewards for each policy in the batch
    reward_data = np.zeros((batch_size))
    for ii in range(batch_size):

        # evaluating reward for ii'th element of the batch
        #   - can perform different operations depending on the epoch type
        #     for example, using more averaging for eval epochs
        if epoch_type == 'evaluation':
            reward_data[ii] = pi_pulse_oct_style_sim(real_pulses[ii],
                                                     imag_pulses[ii])
        elif epoch_type == 'training':
            reward_data[ii] = pi_pulse_oct_style_sim(real_pulses[ii],
                                                     imag_pulses[ii])
        
    # Print mean and stdev of reward for monitoring progress
    R = np.mean(reward_data) 
    std_R = np.std(reward_data)
    logger.info('Average reward %.3f' %R)
    logger.info('STDev reward %.3f' %std_R)
    
    # send reward data back to server (see tf_env -> reward_remote())
    logger.info('Sending message to RL agent server.')
    logger.info('Time stamp: %f' %time.time())
    client_socket.send_data(reward_data)
    


# %%
