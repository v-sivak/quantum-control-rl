# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 11:18:30 2021

@author: qulab
"""
from init_script import *

class pi_pulse_unselectivity(FPGAExperiment):
    disp_range = RangeParameter((0,10,101))
    detune_range = RangeParameter((-20,0,11))
    loop_delay = IntParameter(4e6)
    flip_qubit = BoolParameter(False)
    return_cavity = BoolParameter(True)

    def sequence(self):

        with qubit.scan_detune(*self.detune_range):
            with cavity.displace.scan_amplitude(*self.disp_range):
                sync()
                if self.flip_qubit:
                    qubit.flip()
                sync()
                cavity.displace(amp='dynamic')
                sync()
                qubit.flip()
                sync()
                if self.return_cavity:
                    cavity.displace(amp='dynamic', phase=np.pi)
                delay(24)
                sync()
                readout()
                delay(self.loop_delay)
