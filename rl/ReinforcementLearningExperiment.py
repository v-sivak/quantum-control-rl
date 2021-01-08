# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 10:23:25 2021

@author: V
"""
import numpy as np
from fpga_lib.parameters import IntParameter
import sys
from init_script import *
from gkp_exp.CD_gate.conditional_displacement_compiler import ConditionalDisplacementCompiler
from remote_env_tools import Client

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
    batch_size = IntParameter(200)
    EPOCHS = IntParameter(10)
    n_blocks = IntParameter(1)
    averages_per_block = IntParameter(1)

    save_raw = False
    save_processed = False
    
    def __init__(self, name):
        super(ReinforcementLearningExperiment, self).__init__(name)
        self.beta = 2.5
        self.tau = 100.
        self.loop_delay = 4e6
        self.batch_size = 200
        self.EPOCHS = 10
        self.n_blocks = 1
        self.averages_per_block = 1

    def sequence(self):
        
        def init_circuit(init_state):
            if init_state=='e': qubit.flip()
            sync()
#        
#        def action_circuit(i):
#            sync()
#            qubit.array_pulse(*self.qubit_pulse[i])
#            cavity.array_pulse(*self.cavity_pulse[i])
#            sync()            

        def reward_circuit(init_state):
            return_phase = np.pi/2.0 if init_state=='g' else -np.pi/2.0
            cavity.displace(self.beta/2.0, phase=return_phase)
            sync()
            qubit.flip(selective=True)
            sync()
            readout(**{init_state:'se'})
            sync()
#        
#        for i in range(self.batch_size):
#            init_circuit('g')
#            action_circuit(i)
#            reward_circuit('g')
#            delay(self.loop_delay)
#            
#            init_circuit('e')
#            action_circuit(i)
#            reward_circuit('e')
#            delay(self.loop_delay)
        
        @subroutine
        def update_displacement(amp_reg):
            cx = FloatRegister()
            sx = FloatRegister()
            cx <<= amp_reg
            sx <<= 0
            
            sync()
            DynamicMixer[0][0] <<= cx
            DynamicMixer[1][1] <<= cx
            DynamicMixer[0][1] <<= sx
            DynamicMixer[1][0] <<= -sx
            sync()
            delay(2000)
            cavity.load_mixer()
            delay(2000)

        def CD():
            cavity.displace(amp='dynamic')
            sync()
            delay(self.tau)
            sync()
            cavity.displace(amp='dynamic', phase=-np.pi)
            sync()
            qubit.flip()
            sync()
            cavity.displace(amp='dynamic', phase=-np.pi)
            sync()
            delay(self.tau)
            sync()
            cavity.displace(amp='dynamic')
            sync()

        
        def action_circuit():
            sync()
            CD()
            sync()

        unit_amp = cavity.displace.unit_amp
        dac_amps = unit_amp * self.action['alpha']
        dac_amps_arr = Array(dac_amps, float)
        amp_reg = FloatRegister(0.0)
        i_range = (0, self.batch_size-1, self.batch_size)

        with scan_register(*i_range) as i:
            amp_reg <<= dac_amps_arr[i]
            delay(1000)
            update_displacement(amp_reg)
            
            init_circuit('g')
            action_circuit()
            reward_circuit('g')
            delay(self.loop_delay)
            
            init_circuit('e')
            action_circuit()
            reward_circuit('e')
            delay(self.loop_delay)

        

    
    # create a client server and connect to the agent server
    def update_pulse_params(self, action):
        """
        This creates 'self.action', 'self.cavity_pulse' and 'self.qubit_pulse'.
        
        """
#        C = ConditionalDisplacementCompiler()
#        alpha, phi_g, phi_e = C.CD_params(self.beta, self.tau)
#        self.action = {
#                'alpha' : alpha*np.ones(self.batch_size),
#                'phi_g' : phi_g*np.ones(self.batch_size),
#                'phi_e' : phi_e*np.ones(self.batch_size),
#                'delay' : -10.*np.ones(self.batch_size)}
#        
#        self.cavity_pulse, self.qubit_pulse = [], []
#        for i in range(self.batch_size):
#            C.qubit_pulse_shift = int(round(self.action['delay'][i]))
#            c_pulse, q_pulse = C.make_pulse(self.tau, 
#                                self.action['alpha'][i],
#                                self.action['phi_g'][i],
#                                self.action['phi_e'][i])
#            c_pulse = (c_pulse[0]*(1+0.01*i), c_pulse[1]*(1+0.01*i))
#            self.cavity_pulse.append(c_pulse)
#            self.qubit_pulse.append(q_pulse)
        
        self.action = {'alpha' : action['alpha']*20.0}
        logger.info('Compiled pulses.')
    

    def training_loop(self, **kwargs):
        self.connect_to_RL_agent()
        done = False
        while not done:
            action, done = self.recv_action_data()
            if done: break
            self.update_pulse_params(action)
            self.write_tables()
            logger.info('Updated tables.')
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
            logger.info('Start %s epoch %d' %(data['type'], data['epoch']))
            return data['action'], done
        else:
            logger.info('Training finished.')
            return None, done
        
    
    def create_reward_data(self):
        sigmaz_g = 1 - 2 * self.results['g'].thresh_mean().data
        sigmaz_e = 1 - 2 * self.results['e'].thresh_mean().data
        self.rewards = np.mean([sigmaz_g, -sigmaz_e], axis=0)
        logger.info('Average reward %.3f' %np.mean(self.rewards))
    
    def send_reward_data(self):
        self.client_socket.send_data(self.rewards)
    
    def connect_to_RL_agent(self):
        self.client_socket = Client()
        (host, port) = '172.28.140.123', 5555
        self.client_socket.connect((host, port))
