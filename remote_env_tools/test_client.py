#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 19:22:28 2021

@author: qulab
"""
import time
from remote_env_tools import Client

def create_reward_data(i):
    time.sleep(1)
    return {'R' : [i,i+1]}

client_socket = Client()
(host, port) ='172.28.140.123', 5555
client_socket.connect((host, port))

done = False
i = 0
while not done:
    actions, done = client_socket.recv_data()
    if done: break
    print(actions)
    rewards = create_reward_data(i)
    client_socket.send_data(rewards)
    i += 1