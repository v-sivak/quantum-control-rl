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


    def training_loop(self, **kwargs):
        self.connect_to_RL_agent()
        done = False
        first_run = True
        while not done:
            action, done = self.recv_action_data()
            if done: break
            self.update_pulse_params(action)
            if first_run:
                # on the first run, compile and write all tables
                self.write_tables()
                self.map_wave_memory()
                first_run = False
                logger.info('Updated tables.')
            else:
                # on subsequent runs, only update wave tables
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
        

    def map_wave_memory(self):
        """
        Map out the locations of trainable waves in the waves_list after 
        compilation.
        """
        def find(chan, wave):
            locations = []
            for j, wave_tuple in enumerate(self.dsl_state.iq_table.waves_list):
                (channel, i_wave, q_wave, interp_step) = wave_tuple
                if channel == chan and len(i_wave) == len(wave[0]):
                    if np.allclose((i_wave, q_wave), wave):
                        locations.append(j)
            return locations
        
        self.add_waves_compiler_calls = {}
        for chan in self.trainable_pulses.keys():
            self.add_waves_compiler_calls[chan] = []
            for j in range(self.batch_size):
                wave = self.trainable_pulses[chan][j]
                locs = find(chan, wave)
                self.add_waves_compiler_calls[chan].append(locs)


    def write_wave_tables_only(self):
        """
        To avoid re-compilation, this method simply builds the IQTable by 
        appending waves to WaveMemory in the same order as during the original
        compilation, but replacing the trainable waves with new versions. This
        way they go to the same memory location, and there is no need to 
        re-compile the sequence of instructions.
        """
        waves_list = self.dsl_state.iq_table.waves_list
        
        for chan in self.trainable_pulses.keys():
            for j in range(self.batch_size):
                locs = self.add_waves_compiler_calls[chan][j]
                (i_wave, q_wave) = self.trainable_pulses[chan][j]
                i_wave = np.ascontiguousarray(i_wave)
                q_wave = np.ascontiguousarray(q_wave)
                for i in locs:
                    waves_list[i] = (chan, i_wave, q_wave, [None, None])

        self.dsl_state.iq_table = compiler.IQTable(self.cards)

        for i, wave_tuple in enumerate(waves_list):
            (channel, i_wave, q_wave, interp_step) = wave_tuple
            self.dsl_state.iq_table.add_waves(channel, i_wave, q_wave, interp_step)

        for i in range(len(config.cards)):
            tables_prefix = os.path.join(self.directory, 'Table_B%d' % i)
            self.dsl_state.iq_table.wms[i].export(tables_prefix)
