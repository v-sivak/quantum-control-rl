# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 13:37:53 2021

@author: qulab
"""
from init_script import *
import numpy as np

class state_prep_fock_reward_fpga(FPGAExperiment):
    """ State preparation with Fock reward."""
    loop_delay = FloatParameter(4e6)
    batch_size = IntParameter(10)
    opt_file = StringParameter('')
    fock = IntParameter(4)

    self.qubit = qubit_ef

    def sequence(self):
        # load pulse sequences and phase space points from file
        params = np.load(self.opt_file)
        self.qubit_pulses = [(p.real, p.imag) for p in params['qubit_pulses']]
        self.cavity_pulses = [(p.real, p.imag) for p in params['cavity_pulses']]

        @subroutine
        def reward_circuit():
            readout(m1='se')
            sync()
            self.qubit.flip(selective=True, detune=-self.fock*cavity.chi)
            sync()
            readout(m2='se')
            delay(self.loop_delay)

        @subroutine
        def init_circuit():
            readout(m0='se')

        def control_circuit(i):
            sync()
            qubit.array_pulse(*self.qubit_pulses[i])
            cavity.array_pulse(*self.cavity_pulses[i])
            sync()

        for i in range(self.batch_size):
            # initialize, run control circuit and collect reward
            init_circuit()
            control_circuit(i)
            reward_circuit()
