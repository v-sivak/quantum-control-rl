#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 19:22:28 2021

@author: Vladimir Sivak
"""
import pickle
import socket
import time

import logging
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')

HEADERSIZE = 10

def recv_data(connection):
    full_msg = b''
    new_msg = True
    msg_ended = False
    while not msg_ended:
        msg = connection.recv(16)
        if new_msg:
            msglen = int(msg[:HEADERSIZE])
            logging.info('New msg len: %d' % msglen)
            new_msg = False

        full_msg += msg

        if len(full_msg) - HEADERSIZE == msglen:
            logging.info('Full msg recieved')
            data = pickle.loads(full_msg[HEADERSIZE:])
            print(data)
            msg_ended = True
    return data

def send_data(data, connection):
    msg = pickle.dumps(data, protocol=2)
    msg = bytes(str(len(msg)).zfill(HEADERSIZE)) + msg
    connection.send(msg)


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

