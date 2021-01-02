#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 19:22:28 2021

@author: qulab
"""

import pickle
import socket
import time
from server import recv_data, send_data, HEADERSIZE

def test_create_data(i):
    time.sleep(3)
    return {'R' : [i,i+1]}

def client(N=10, create_data=test_create_data):
    host = socket.gethostname()
    port = 5000
    client_socket = socket.socket()
    client_socket.connect((host, port))
    for i in range(N):
        # receive action data from agent
        data = recv_data(client_socket)
        # send reward data to agent
        data = create_data(i)
        send_data(data, client_socket)
    client_socket.close()


if __name__ == '__main__':
    client()

