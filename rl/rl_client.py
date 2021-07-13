# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 10:23:25 2021

@author: Vlad
"""
import config
import sys
import numpy as np
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

    def update_exp_params(self):
        raise NotImplementedError

    def create_reward_data(self):
        raise NotImplementedError

    def training_loop(self, **kwargs):
        if self.use_gui:
            connect_to_gui()
            get_gui()
        self.connect_to_RL_agent()
        done = False
        while not done:
            self.message, done = self.recv()
            if done: break
            self.split_batch()
            reward_data = [[]] * self.N_mini_batches
            for i in range(self.N_mini_batches):
                self.mini_batch_idx = i
                self.update_exp_params()
                self.exp.run()
                wait_complete(self.exp)
                logger.info('Finished collecting mini-batch %d.' %i)
                self.results = get_last_results(self.exp)
                reward_data[i] = self.create_reward_data()
            reward_data = np.concatenate(reward_data, axis=self.batch_axis)
            self.send(reward_data)

    def recv(self):
        message, done = self.client_socket.recv_data()
        logger.info('\nReceived message from RL agent server.')
        if not done:
            self.batch_size = message.pop('batch_size')
            self.epoch = message.pop('epoch')
            self.epoch_type = message.pop('epoch_type')
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
        
    def split_batch(self):
        N = self.batch_size // self.max_mini_batch_size
        mini_batches = [self.max_mini_batch_size] * N
        last_mini_batch = self.batch_size % self.max_mini_batch_size
        if last_mini_batch: mini_batches.append(last_mini_batch)
        self.mini_batches = mini_batches
        self.N_mini_batches = len(mini_batches)