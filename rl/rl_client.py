# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 10:23:25 2021

@author: Vlad
"""
import config
import sys
from remote_env_tools import Client
from fpga_lib.scripting import wait_complete, get_last_results, connect_to_gui, get_gui

import logging
logger = logging.getLogger('RL')
logger.propagate = False
logger.handlers = []
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(config.formatter)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

__all__ = ['ReinforcementLearningExperiment']

class ReinforcementLearningExperiment():

    def update_exp_params(self, message):
        raise NotImplementedError

    def create_reward_data(self, results):
        raise NotImplementedError

    def training_loop(self, **kwargs):
        self.connect_to_RL_agent()
        connect_to_gui()
        get_gui()
        done = False
        while not done:
            message, done = self.recv()
            if done: break
            self.update_exp_params(message)
            self.exp.run()
            wait_complete(self.exp)
            logger.info('Finished collecting data.')
            results = get_last_results(self.exp)
            reward_data = self.create_reward_data(results)
            self.send(reward_data)

    def recv(self):
        message, done = self.client_socket.recv_data()
        logger.info('Received message from RL agent server.')
        if not done:
            self.batch_size = message['batch_size']
            self.epoch = message['epoch']
            self.epoch_type = message['epoch_type']
            logger.info('Start %s epoch %d' %(self.epoch_type, self.epoch))
            return (message, done)
        else:
            logger.info('Training finished.')
            return (None, done)

    def send(self, message):
        self.client_socket.send_data(message)
    
    def connect_to_RL_agent(self):
        self.client_socket = Client()
        (host, port) = '172.28.142.46', 5555
        self.client_socket.connect((host, port))
        
