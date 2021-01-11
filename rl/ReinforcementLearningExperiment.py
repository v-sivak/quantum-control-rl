# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 10:23:25 2021

@author: Vlad
"""
import numpy as np
from fpga_lib.parameters import IntParameter, FloatParameter
from fpga_lib import compiler as fc
import sys
from init_script import *
from gkp_exp.CD_gate.conditional_displacement_compiler import ConditionalDisplacementCompiler
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
    beta = FloatParameter(1.)
    tau = FloatParameter(100.)
    loop_delay = FloatParameter(4e6)
    n_blocks = IntParameter(1)
    averages_per_block = IntParameter(1)

    save_raw = True # don't change, important for data shape consistency
    save_processed = False
    
    def __init__(self, name):
        super(ReinforcementLearningExperiment, self).__init__(name)
        self.beta = 2.5
        self.tau = 100.
        self.loop_delay = 4e6
        
        # TODO: implement results parsing for arb n_blocks and avgs_per_block
        self.n_blocks = 1               
        self.averages_per_block = 1


    def sequence(self):
        
        @subroutine
        def init_circuit(init_state):
            if init_state=='e': qubit.flip()
            sync()

        @subroutine
        def reward_circuit(init_state):
            return_phase = np.pi/2.0 if init_state=='g' else -np.pi/2.0
            cavity.displace(self.beta/2.0, phase=return_phase)
            sync()
            qubit.flip(selective=True)
            sync()
            readout(**{init_state:'se'})
            sync()
        
        def action_circuit(i):
            sync()
            qubit.array_pulse(*self.qubit_pulse[i])
            cavity.array_pulse(*self.cavity_pulse[i])
            sync()

        with Repeat(self.reps, plot_label='reps'):
            for i in range(self.batch_size):
                init_circuit('g')
                action_circuit(i)
                reward_circuit('g')
                delay(self.loop_delay)
                
                init_circuit('e')
                action_circuit(i)
                reward_circuit('e')
                delay(self.loop_delay)
    

    def update_pulse_params(self, action):
        """
        Create lists of qubit and cavity array pulses to use in action circuit. 
        
        Args:
            action (dict): dictionary of parametrizations for action circuit.
        """
        C = ConditionalDisplacementCompiler()
        self.cavity_pulse, self.qubit_pulse = [], []
        for i in range(self.batch_size):
            c_pulse, q_pulse = C.make_pulse(self.tau, 
                                action['alpha'][i]*20.0,
                                action['phi_g'][i]*np.pi/10.0,
                                action['phi_e'][i]*np.pi/10.0)
            self.cavity_pulse.append(c_pulse)
            self.qubit_pulse.append(q_pulse)
        logger.info('Compiled pulses.')


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


    def create_reward_data(self):
        # data shape is [epoch*n_blocks*averages_per_block, reps, batch_size]
        assert self.n_blocks == self.averages_per_block == 1
        sigmaz = {'g' : 1 - 2 * self.results['g'][-1,:].threshold().data,
                  'e' : 1 - 2 * self.results['e'][-1,:].threshold().data}
        self.rewards = np.mean([sigmaz['g'], -sigmaz['e']], axis=0)
        self.rewards = np.mean(self.rewards, axis=0) # average over Repeat axis
        logger.info('Average reward %.3f' %np.mean(self.rewards))


    def send_reward_data(self):
        self.client_socket.send_data(self.rewards)

    
    def connect_to_RL_agent(self):
        self.client_socket = Client()
        (host, port) = '172.28.140.123', 5555
        self.client_socket.connect((host, port))
        

    def map_wave_memory(self):
        
        def find(chan, wave):
            locations = []
            for j, wave_tuple in enumerate(self.dsl_state.iq_table.waves_list):
                (channel, i_wave, q_wave, interp_step) = wave_tuple
                if channel == chan and len(i_wave) == len(wave[0]):
                    if np.allclose((i_wave, q_wave), wave):
                        locations.append(j)
            return locations
        
        # TODO: so far this is only for cavity channel, update to all channels
        chan = cavity.chan        
        self.add_waves_compiler_calls = {}
        self.add_waves_compiler_calls[chan] = []
        for j in range(self.batch_size):
            wave = self.cavity_pulse[j]
            locs = find(chan, wave)
            self.add_waves_compiler_calls[chan].append(locs)


    def write_wave_tables_only(self):
        chan = cavity.chan
        waves_list = self.dsl_state.iq_table.waves_list
        for j in range(self.batch_size):
            locs = self.add_waves_compiler_calls[chan][j]
            (i_wave, q_wave) = self.cavity_pulse[j]
            i_wave, q_wave = np.ascontiguousarray(i_wave), np.ascontiguousarray(q_wave)
            for i in locs:
                waves_list[i] = (chan, i_wave, q_wave, [None, None])

        self.dsl_state.iq_table = fc.IQTable(self.cards)

        for i, wave_tuple in enumerate(waves_list):
            (channel, i_wave, q_wave, interp_step) = wave_tuple
            self.dsl_state.iq_table.add_waves(channel, i_wave, q_wave, interp_step)

        for i in range(len(config.cards)):
            tables_prefix = os.path.join(self.directory, 'Table_B%d' % i)
            self.dsl_state.iq_table.wms[i].export(tables_prefix)
