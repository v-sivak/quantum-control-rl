# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 11:10:10 2021

@author: qulab
"""


# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 13:43:48 2021

@author: qulab
"""
from init_script import *

class qubit_cooling_rotation_sweep(FPGAExperiment):
    readout_detune_MHz = FloatParameter(50.4)
    qubit_detune_MHz = FloatParameter(8.0)
    readout_amp = FloatParameter(0.22)
    qubit_amp = FloatParameter(0.2)
    wait_time = IntParameter(100)
    smooth_sigma_t = IntParameter(40)
    loop_delay = IntParameter(500e3)
    flip_qubit = BoolParameter(False)
    tau = IntParameter(1000)
    phase_range = RangeParameter((-2., 2., 201))
    
    fit_func = 'gaussian'

       
    def sequence(self):        
        ssb0 = -50e6
        ssb_q = ssb0 + self.qubit_detune_MHz*1e6
        ssb_r = ssb0 + self.readout_detune_MHz*1e6
        
        with scan_phase(qubit.chan, -0.5*qubit.pulse.unit_amp, *self.phase_range):
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
            readout_aux.smoothed_constant_pulse(self.tau, amp=self.readout_amp,
                                                sigma_t=self.smooth_sigma_t)
            qubit.smoothed_constant_pulse(self.tau, amp=self.qubit_amp,
                                          sigma_t=self.smooth_sigma_t)
            sync()
            qubit.pulse(amp='dynamic')
            sync()
            delay(self.wait_time)
            sync()
            readout()
            sync()
            delay(self.loop_delay)
            