# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 13:43:48 2021

@author: qulab
"""
from init_script import *

class qubit_cooling_time_sweep(FPGAExperiment):
    readout_detune_MHz = FloatParameter(50.4)
    qubit_detune_MHz = FloatParameter(8.0)
    readout_amp = FloatParameter(0.22)
    qubit_amp = FloatParameter(0.2)
    wait_time = IntParameter(100)
    time_range = RangeParameter((0,1000,21))
    loop_delay = IntParameter(500e3)
    flip_qubit = BoolParameter(False)

       
    def sequence(self):        
        ssb0 = -50e6
        ssb_q = ssb0 + self.qubit_detune_MHz*1e6
        ssb_r = ssb0 + self.readout_detune_MHz*1e6

        for tau in np.linspace(*self.time_range):
            sync()
            qubit.set_ssb(ssb0)
            delay(24)
            sync()
            readout_aux.set_ssb(ssb_r)
            delay(24)
            sync()
            if self.flip_qubit:
                qubit.flip()
            sync()
            qubit.set_ssb(ssb_q)
            delay(24)
            sync()
            duration = 4 * (tau.astype(int) // 4)
            readout_aux.constant_pulse(duration, amp=self.readout_amp)
            qubit.constant_pulse(duration, amp=self.qubit_amp)
            sync()
            qubit.pi2_pulse(phase=-np.pi/2)
            sync()
            delay(self.wait_time)
            sync()
            readout()
            sync()
            delay(self.loop_delay)
            