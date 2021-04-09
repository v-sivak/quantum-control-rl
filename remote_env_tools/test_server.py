# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 15:35:33 2021

@author: Vladimir Sivak
"""
from remote_env_tools import Server

def create_action_data(i):
    return {'alpha': [i**2, i**3], 'beta': [i**4, i+1]}

server_socket = Server()
(host, port) = '172.28.142.46', 5555
server_socket.bind((host, port))
server_socket.connect_client()

for i in range(10):
    actions = create_action_data(i)
    server_socket.send_data(actions)
    rewards = server_socket.recv_data()
    print(rewards)

server_socket.disconnect_client()