# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 13:12:29 2021

@author: qulab
"""
from init_script import *

class T1_with_registers_test(FPGAExperiment):

    fit_func = {'m1' : 'exp_decay'}

    pauli = StringParameter('X')
    init_steps = IntParameter(15)
    steps = RangeParameter((15,50,36))
    step_duration = IntParameter(1e3)

    loop_delay = IntParameter(5e5)

    def sequence(self):

        def control_circuit(s):
            delay(self.step_duration)

        def init_circuit(s, res_name):
            qubit.flip()
            readout(**{res_name : 'se'})

        def reward_circuit(s, res_name):
            readout(**{res_name : 'se'})

        R0 = Register(self.init_steps)
        R1 = Register()
        C = Register()

        with scan_register(*self.steps, reg=R1):
            C <<= 0

            label_next('stabilization')


            label_next('x_round')
            if_then(C==R0, 'init')
            if_then(C==R1, 'reward')
            control_circuit('x')
            C += 1
            goto('p_round')


            label_next('p_round')
            if_then(C==R0, 'init')
            if_then(C==R1, 'reward')
            control_circuit('p')
            C += 1
            goto('x_round')


            label_next('init')
            init_circuit(self.pauli, res_name='m0')
            C += 1
            goto('stabilization')


            label_next('reward')
            reward_circuit(self.pauli, res_name='m1')
            goto('end')


            label_next('end')
            delay(self.loop_delay)

    def process_data(self):
        m0 = self.results['m0'].threshold()
        m1 = self.results['m1'].threshold()

        result = m1.postselect(m0, [0])[0]
        self.results['postselect_g'] = result.thresh_mean().data
        self.results['postselect_g'].ax_data = self.results['m0'].ax_data[1:]
        self.results['postselect_g'].labels = self.results['m0'].labels[1:]

        result = m1.postselect(m0, [1])[0]
        self.results['postselect_e'] = result.thresh_mean().data
        self.results['postselect_e'].ax_data = self.results['m0'].ax_data[1:]
        self.results['postselect_e'].labels = self.results['m0'].labels[1:]
