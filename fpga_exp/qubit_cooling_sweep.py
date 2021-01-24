# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 14:21:40 2021

@author: qulab
"""
from init_script import *

class qubit_cooling_sweep(FPGAExperiment):
    readout_detune_MHz = FloatParameter(30.4)
    qubit_amp = FloatParameter(0.06)
    pump_time = IntParameter(200)
    wait_time = IntParameter(1000)
    readout_amp_range = RangeParameter((0, 0.1, 21))
    qubit_detune_range = RangeParameter((-10e6, 10e6, 21))
    loop_delay = IntParameter(500e3)
    flip_qubit = BoolParameter(False)

    def sequence(self):

        ssb0 = -50e6
        qubit_detune_range = ((ssb0 + self.qubit_detune_range[0]) / 1e3,
                              (ssb0 + self.qubit_detune_range[1]) / 1e3,
                              self.qubit_detune_range[2])

        with scan_register(*qubit_detune_range) as ssbreg:
            with scan_amplitude(readout.chan, *self.readout_amp_range):
                if self.flip_qubit:
                    qubit.flip()
                sync()
                qubit.set_ssb(ssbreg)
                delay(500)
                sync()
                readout.constant_pulse(self.pump_time, amp='dynamic', 
                                       detune=self.readout_detune_MHz*1e6)
                qubit.constant_pulse(self.pump_time, amp=self.qubit_amp)
                sync()
                qubit.pi2_pulse(phase=-np.pi/2)
                sync()
                delay(self.wait_time)
                sync()
                readout()
                sync()
                delay(self.loop_delay)
