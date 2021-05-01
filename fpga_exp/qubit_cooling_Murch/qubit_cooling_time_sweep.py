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
    smooth_sigma_t = IntParameter(40)
    time_range = RangeParameter((0,1000,21))
    loop_delay = IntParameter(500e3)
    flip_qubit = BoolParameter(False)

    def sequence(self):
        ssb0 = -50e6
        ssb_q = ssb0 + self.qubit_detune_MHz*1e6
        ssb_r = ssb0 + self.readout_detune_MHz*1e6

        self.qubit_detuned = qubit_ef
        self.qubit = qubit

        self.readout_detuned = readout_aux
        self.readout = readout

        sync()
        self.readout_detuned.set_ssb(ssb_r)
        self.qubit_detuned.set_ssb(ssb_q)
        sync()
        
        def cool_Murch(duration):
            sync()
            self.readout_detuned.smoothed_constant_pulse(
                    duration, amp=self.readout_amp, sigma_t=self.smooth_sigma_t)
            self.qubit_detuned.smoothed_constant_pulse(
                    duration, amp=self.qubit_amp, sigma_t=self.smooth_sigma_t)
            sync()            
            
        for tau in np.linspace(*self.time_range):
            
            duration = 4 * (tau.astype(int) // 4)
            sync()
            if self.flip_qubit:
#                self.qubit.flip()
                self.qubit.pi2_pulse()
            
            cool_Murch(duration)

            delay(self.wait_time)
            sync()
            self.readout()
            sync()
            delay(self.loop_delay)


    def process_data(self):
        
        durations = []
        for tau in np.linspace(*self.time_range):
            durations.append(4 * (tau.astype(int) // 4))

        if self.blocks_seen > 0:
            self.results['default'].ax_data[1] = np.array(durations)
            self.results['default'].labels[1] = 'Time (ns)'