#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 19:22:02 2021

@author: Vladimir Sivak
"""
import pickle
import socket
import sys

import logging
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')

class PickleSocket(socket.socket):
    """ This is a simple python socket to send pickled data over TCP/IP."""
    pickle_protocol = 2 # if client is in py2 environment 
    HEADERSIZE = 10
    
    def recv_data(self, connection):
        full_msg = b''
        new_msg = True
        msg_ended = False
        while not msg_ended:
            msg = connection.recv(16)
            if new_msg:
                if msg == b'': return  (None, True)
                msglen = int(msg[:self.HEADERSIZE])
                logging.info('New msg len: %d' % msglen)
                new_msg = False
            full_msg += msg
            if len(full_msg) - self.HEADERSIZE == msglen:
                logging.info('Full msg recieved')
                data = pickle.loads(full_msg[self.HEADERSIZE:])
                msg_ended = True
        return (data, False)

    def send_data(self, data, connection):
        msg = pickle.dumps(data, protocol=self.pickle_protocol)
        header_str = str(len(msg)).zfill(self.HEADERSIZE)
        # there is a slight difference for py2 vs py3
        header = header_str if sys.version[0]=='2' else bytes(header_str, 'utf-8')
        msg = header + msg
        connection.send(msg)

    
class Server(PickleSocket):
    """ A server for a single client. 
    
    Intended use: server is on the RL agent side, client is on the environment 
    side. Server sends actions to client, client sends rewards back to server.
    """
    def send_data(self, data):
        super(Server, self).send_data(data, self.client_socket)
    
    def recv_data(self):
        return super(Server, self).recv_data(self.client_socket)
    
    def connect_client(self):
        self.listen(1)
        self.client_socket, self.client_address = self.accept()
        logging.warning('Connection with: ' + str(self.client_address))

    def disconnect_client(self):
        self.client_socket.close()


class Client(PickleSocket):
    """ A simple client. 
    
    Intended use: server is on the RL agent side, client is on the environment 
    side. Server sends actions to client, client sends rewards back to server.
    """
    def send_data(self, data):
        super(Client, self).send_data(data, self)
    
    def recv_data(self):
        return super(Client, self).recv_data(self)
    
    
    