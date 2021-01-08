#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 19:22:28 2021

@author: qulab
"""

import socket
import time
from remote_env_tools import Client

def create_reward_data(i):
    time.sleep(3)
    return {'R' : [i,i+1]}

client_socket = Client()
(host, port) ='172.28.140.123', 5555
client_socket.connect((host, port))

for i in range(10):
    actions = client_socket.recv_data()
    print(actions)
    rewards = create_reward_data(i)
    client_socket.send_data(rewards)

client_socket.close()