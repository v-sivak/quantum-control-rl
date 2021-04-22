# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 20:51:11 2021

@author: qulab
"""


from init_script import *

class qubit_number_split_spec(FPGAExperiment):
    detune_range = RangeParameter((-20e6,20e6,41))
    selective = BoolParameter(True)
    loop_delay = FloatParameter(4e6)
    alpha = FloatParameter(3.0)

    def sequence(self):
        qssb = -50e6
        start = self.detune_range[0] + qssb
        stop = self.detune_range[1] + qssb
        points = self.detune_range[-1]


        with scan_register(start/1e3, stop/1e3, points) as ssbreg:
            qubit.set_ssb(ssbreg)
            delay(2000)

            sync()
            cavity.displace(amp=self.alpha)
            sync()
            qubit.flip(selective=self.selective)
            sync()
#            cavity.displace(amp=self.alpha, phase=np.pi)
            sync()
            readout()
            delay(self.loop_delay)
