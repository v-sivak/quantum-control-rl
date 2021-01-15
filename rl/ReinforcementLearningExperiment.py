# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 10:23:25 2021

@author: Vlad
"""
import numpy as np
import config
from fpga_lib.parameters import IntParameter
from fpga_lib import compiler
from fpga_lib.experiments.fpga import FPGAExperiment
import sys
from remote_env_tools import Client
import os
import matplotlib.pyplot as plt


import logging
logger = logging.getLogger('RL')
logger.propagate = False
logger.handlers = []
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(config.formatter)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

__all__ = ['ReinforcementLearningExperiment']

class ReinforcementLearningExperiment(FPGAExperiment):
    n_blocks = IntParameter(1)
    averages_per_block = IntParameter(1)
    save_raw = True # don't change, important for data shape consistency
    save_processed = False
    
    def __init__(self, name):
        super(ReinforcementLearningExperiment, self).__init__(name)
        
        # TODO: implement results parsing for arb n_blocks and avgs_per_block
        self.n_blocks = 1
        self.averages_per_block = 1
        
        self.trainable_pulses = {} # keys are fpga channels

    def sequence(self):
        """
        Make sure to set 'pulse_id' (in array_pulse) equal to index of the 
        trainable pulse in the batch. Channel and pulse_id are used to 
        identify the pulse in wave memory and update it on each epoch.
        """
        raise NotImplementedError


    def update_pulse_params(self):
        raise NotImplementedError


    def create_reward_data(self):
        raise NotImplementedError


    def training_loop(self, **kwargs):
        self.connect_to_RL_agent()
        done = False
        while not done:
            action, done = self.recv_action_data()
            if done: break
            self.update_pulse_params(action)
            if self.compile_flag:
                self.write_tables()
                logger.info('Updated tables.')
            else:
                self.write_wave_tables_only()
                logger.info('Updated wave tables.')
            self.run(compile=False)
            self.wait_complete()
            logger.info('Finished collecting data.')
            self.process_available_blocks()
            self.create_reward_data()
            self.send_reward_data()


    def recv_action_data(self):
        data, done = self.client_socket.recv_data()
        logger.info('Received data from RL agent server.')
        if not done:
            self.batch_size = data['batch_size']
            self.reps = data['reps']
            self.epoch = data['epoch']
            self.compile_flag = data['compile_flag']
            self.epoch_type = data['type']
            logger.info('Start %s epoch %d' %(data['type'], data['epoch']))
            return (data['action'], done)
        else:
            logger.info('Training finished.')
            return (None, done)


    def send_reward_data(self):
        self.client_socket.send_data(self.rewards)

    
    def connect_to_RL_agent(self):
        self.client_socket = Client()
        (host, port) = '172.28.140.123', 5555
        self.client_socket.connect((host, port))
        

    def write_wave_tables_only(self):
        """
        To avoid re-compilation, this method simply builds the IQTable by 
        appending pulses to WaveMemory in the same order as during the original
        compilation, but replacing the trainable pulses with new versions. This
        way they go to the same memory location, and there is no need to 
        re-compile the sequence of instructions.
        
        """
        waves_list = self.dsl_state.iq_table.waves_list
        for i, wave_tuple in enumerate(waves_list):
            chan, j = wave_tuple[0], wave_tuple[-1]
            if chan in self.trainable_pulses.keys() and j is not None:
                (i_wave, q_wave) = self.trainable_pulses[chan][j]
                i_wave = np.ascontiguousarray(i_wave)
                q_wave = np.ascontiguousarray(q_wave)
                waves_list[i] = (chan, i_wave, q_wave, [None, None], j)

        self.dsl_state.iq_table = compiler.IQTable(self.cards)
        for i, wave_tuple in enumerate(waves_list):
            self.dsl_state.iq_table.add_waves(*wave_tuple)

        for chan in self.trainable_pulses.keys():
            card = chan[0]
            tables_prefix = os.path.join(self.directory, 'Table_B%d' % card)
            self.dsl_state.iq_table.wms[card].export(tables_prefix)
    
    
    def plot_array_pulse(self, i_wave, q_wave):
        times = np.arange(len(i_wave))
        fig, axes = plt.subplots(2,1, sharex=True)
        axes[1].set_xlabel('Time (ns)')
        axes[0].set_ylabel('Re')
        axes[1].set_ylabel('Im')
        axes[0].plot(times, i_wave)
        axes[1].plot(times, q_wave)