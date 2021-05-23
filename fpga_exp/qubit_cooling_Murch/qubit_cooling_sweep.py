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
    qubit_ramp_t = IntParameter(200)
    readout_ramp_t = IntParameter(50)
    readout_amp_range = RangeParameter((0, 0.1, 21))
    qubit_detune_range = RangeParameter((-10e6, 10e6, 21))
    loop_delay = IntParameter(500e3)

    def sequence(self):

        self.qubit_detuned = qubit_ef
        self.qubit = qubit

        readout_pump_time = self.pump_time+2*self.qubit_ramp_t-2*self.readout_ramp_t

        def exp(res_name, flip_qubit):

            with self.qubit_detuned.scan_detune(*self.qubit_detune_range):
                with scan_amplitude(readout.chan, *self.readout_amp_range):
                    sync()
                    if flip_qubit:
                        self.qubit.flip()
                    else:
                        delay(24)
                    sync()
                    readout.smoothed_constant_pulse(
                            readout_pump_time, amp='dynamic',
                            detune=self.readout_detune_MHz*1e6,
                            sigma_t=self.readout_ramp_t)
                    self.qubit_detuned.smoothed_constant_pulse(
                            self.pump_time, amp=self.qubit_amp,
                            sigma_t=self.qubit_ramp_t)
                    sync()
                    delay(24)
                    readout(**{res_name:'se'})
                    sync()
                    delay(self.loop_delay)

        exp('g', False)
        exp('e', True)


    def process_data(self):

        g_sz = 1 - 2*self.results['g'].threshold()
        e_sz = 1 - 2*self.results['e'].threshold()

        self.results['avg_sz'] = (g_sz + e_sz) / 2.0
        self.results['avg_sz'].ax_data = self.results['g'].ax_data
        self.results['avg_sz'].labels = self.results['g'].labels
        #self.results['avg_sz'].vmin = -1
        #self.results['avg_sz'].vmax = +1
