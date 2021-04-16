# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 09:14:27 2021

@author: Vladimir Sivak
"""
import os
import numpy as np
# append parent 'gkp-rl' directory to path 
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from remote_env_tools.remote_env_tools import Client



# connect to the agent
client_socket = Client()
(host, port) = '172.28.142.46', 5555
client_socket.connect((host, port))

# training loop
done = False
while not done:
    # receive action data from the agent
    message, done = client_socket.recv_data()
    print('Received data.')
    if done: break
    action_batch = message['action_batch']
    mini_buffer = message['mini_buffer']
    print('Epoch %d' %message['epoch'])
    print(message['epoch_type'])
    print(message['action_batch']['beta'].shape)
    print('N_alpha=%d, N_msmt=%d' %(message['N_alpha'], message['N_msmt']))
    
    msmt = np.ones([2,message['batch_size'],message['N_alpha'],message['N_msmt']])
    client_socket.send_data(msmt)






# # Create environment that will produce mock measurement outcomes


# # connect to the agent
# client_socket = Client()
# (host, port) = '172.28.142.46', 5554
# client_socket.connect((host, port))

# # training loop
# done = False
# while not done:
#     # receive action data from the agent
#     message, done = client_socket.recv_data()
#     print('Received data.')
#     print(message)    
#     if done: break
#     action_batch = message['action_batch']
#     mini_buffer = message['mini_buffer']

#     msmt = np.ones([2,1,100,10])
#     client_socket.send_data(msmt)
