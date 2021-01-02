#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 19:22:02 2021

@author: Vladimir Sivak
"""
import pickle
import socket

import logging
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')

HEADERSIZE = 10
pickle_protocol = 2

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
    msg = pickle.dumps(data, protocol=pickle_protocol)
    msg = bytes(str(len(msg)).zfill(HEADERSIZE), 'utf-8') + msg
    # msg = bytes(f'{len(msg):<{HEADERSIZE}}', 'utf-8') + msg
    connection.send(msg)

def test_create_data(i):
    return {'alpha': [i**2, i**3], 'beta': [i**4,i+1]}


def server(N=10, create_data=test_create_data):
    host = socket.gethostname()
    port = 5000
    server_socket = socket.socket()
    server_socket.bind((host, port))
    server_socket.listen(2)
    client_socket, address = server_socket.accept()
    logging.warning('Connection with: ' + str(address))
    for i in range(N):
        # send action data to fpga
        data = create_data(i)
        send_data(data, client_socket)
        # receive reward data from fpga
        data = recv_data(client_socket)
    client_socket.close()


if __name__ == '__main__':
    server()