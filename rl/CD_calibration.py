# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:30:49 2021

@author: qulab
"""
from init_script import *
import numpy as np
from ReinforcementLearningExperiment import ReinforcementLearningExperiment
from gkp_exp.CD_gate.conditional_displacement_compiler import ConditionalDisplacementCompiler

import logging
logger = logging.getLogger('RL')

class CD_calibration(ReinforcementLearningExperiment):
    """
    This is a simple version of the CD gate calibration experiment.
    Only the displacement amplitude 'alpha' is learned.
    """
    beta = FloatParameter(1.)
    tau = FloatParameter(100.)
    loop_delay = FloatParameter(4e6)
    
    def __init__(self, name):
        super(CD_calibration, self).__init__(name)
        self.beta = 2.5
        self.tau = 100.
        self.loop_delay = 4e6

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
        w = {'alpha' : [-20.0, 20.0]} # params window

        action_scaled = {s : action[s]*(w[s][1]-w[s][0])/2+(w[s][1]+w[s][0])/2
                         for s in action.keys()}
        
        C = ConditionalDisplacementCompiler()
        self.cavity_pulse, self.qubit_pulse = [], []
        for i in range(self.batch_size):
            c_pulse, q_pulse = C.make_pulse(
                    self.tau, action_scaled['alpha'][i],0,0)
            self.cavity_pulse.append(c_pulse)
            self.qubit_pulse.append(q_pulse)
        self.trainable_pulses[(cavity.chan)] = self.cavity_pulse
        logger.info('Compiled pulses.')



    def create_reward_data(self):
        # data shape is [epoch*n_blocks*averages_per_block, reps, batch_size]
        assert self.n_blocks == self.averages_per_block == 1
        sigmaz = {'g' : 1 - 2 * self.results['g'][-1,:].threshold().data,
                  'e' : 1 - 2 * self.results['e'][-1,:].threshold().data}
        self.rewards = np.mean([sigmaz['g'], -sigmaz['e']], axis=0)
        self.rewards = np.mean(self.rewards, axis=0) # average over Repeat axis
        logger.info('Average reward %.3f' %np.mean(self.rewards))
