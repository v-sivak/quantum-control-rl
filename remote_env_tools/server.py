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

class PickleSocket(socket.socket):
    """ This is a simple python socket to send pickled data over TCP/IP."""
    pickle_protocol = 2
    HEADERSIZE = 10
    
    def recv_data(self, connection):
        full_msg = b''
        new_msg = True
        msg_ended = False
        while not msg_ended:
            msg = connection.recv(16)
            if new_msg:
                msglen = int(msg[:self.HEADERSIZE])
                logging.info('New msg len: %d' % msglen)
                new_msg = False
            full_msg += msg
            if len(full_msg) - self.HEADERSIZE == msglen:
                logging.info('Full msg recieved')
                data = pickle.loads(full_msg[self.HEADERSIZE:])
                print(data)
                msg_ended = True
        return data

    def send_data(self, data, connection):
        msg = pickle.dumps(data, protocol=self.pickle_protocol)
        msg = bytes(str(len(msg)).zfill(self.HEADERSIZE), 'utf-8') + msg
        connection.send(msg)

    
class Server(PickleSocket):
    """ A server for a single client. 
    
    Intended use: server is on the RL agent side, client is on the environment 
    side. Server sends actions to client, client sends rewards back to server.
    """
    def send_data(self, data):
        super().send_data(data, self.client_socket)
    
    def recv_data(self):
        return super().recv_data(self.client_socket)
    
    def connect_client(self):
        self.listen(1)
        self.client_socket, self.client_address = server_socket.accept()
        logging.warning('Connection with: ' + str(self.client_address))

    def disconnect_client(self):
        self.client_socket.close()


class Client(PickleSocket):
    """ A simple client. 
    
    Intended use: server is on the RL agent side, client is on the environment 
    side. Server sends actions to client, client sends rewards back to server.
    """
    def send_data(self, data):
        super().send_data(data, self)
    
    def recv_data(self):
        return super().recv_data(self)


def test_create_data(i):
    return {'alpha': [i**2, i**3], 'beta': [i**4,i+1]}



if __name__ == '__main__':
    
    server_socket = Server()
    (host, port) = socket.gethostname(), 5000
    server_socket.bind((host, port))
    server_socket.connect_client()
    
    for i in range(10):
        # send action data to fpga
        data = test_create_data(i)
        server_socket.send_data(data)
        # receive reward data from fpga
        data = server_socket.recv_data()
    
    server_socket.disconnect_client()