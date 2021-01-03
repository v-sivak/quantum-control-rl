#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 19:22:28 2021

@author: qulab
"""

import pickle
import socket
import time
from server import Client

def test_create_data(i):
    time.sleep(3)
    return {'R' : [i,i+1]}


if __name__ == '__main__':
    
    client_socket = Client()
    (host, port) = socket.gethostname(), 5000
    client_socket.connect((host, port))

    for i in range(10):
        # receive action data from agent
        data = client_socket.recv_data()
        # send reward data to agent
        data = test_create_data(i)
        client_socket.send_data(data)
    
    client_socket.close()