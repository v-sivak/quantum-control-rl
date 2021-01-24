# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 13:49:54 2021

@author: qulab
"""
from init_script import *

class Rabi_time(FPGAExperiment):
    amp = FloatParameter(0.5)
    time_range = RangeParameter((0,200,11))
    loop_delay=0.5e6
    fit_func = 'sine'
    fixed_fit_params = {'phi': -np.pi/2}

    def sequence(self):
        with scan_length(*self.time_range) as dynlen:
            sync()
            system.cool_qubit()
            sync()
            qubit.constant_pulse(dynlen, self.amp)
            sync()
            readout()
            delay(self.loop_delay)

    def process_fit_data(self):
        self.fit_params['rabi_frequency_MHz'] = self.fit_params['f0']*1e9 * 1e-6
