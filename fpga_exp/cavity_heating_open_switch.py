# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 17:56:53 2021

@author: qulab
"""
from init_script import *

class cavity_heating_open_switch(FPGAExperiment):
    detune_cavity = IntParameter(180e6)
    loop_delay = IntParameter(4e6)
    time_range = RangeParameter((0,100e3,101))
    
    def sequence(self):
        cavity.set_detune(self.detune_cavity)        
        with scan_length(*self.time_range) as dynlen:
            sync()
            cavity.constant_pulse(dynlen, amp=0.0001)
            sync()
            qubit.flip(selective=True)
            sync()
            readout()
            sync()
            delay(self.loop_delay)
