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
    smooth_sigma_t = IntParameter(40)
    readout_amp_range = RangeParameter((0, 0.1, 21))
    qubit_detune_range = RangeParameter((-10e6, 10e6, 21))
    loop_delay = IntParameter(500e3)
    flip_qubit = BoolParameter(False)

    def sequence(self):

        ssb0 = -50e6
        qubit_detune_range = ((ssb0 + self.qubit_detune_range[0]) / 1e3,
                              (ssb0 + self.qubit_detune_range[1]) / 1e3,
                              self.qubit_detune_range[2])

        self.qubit_detuned = qubit_ef
        self.qubit = qubit
        
        def exp(res_name, flip_qubit):
            with scan_register(*qubit_detune_range, 
                               plot_label='ssb frequency (kHz)') as ssbreg:
                with scan_amplitude(readout.chan, *self.readout_amp_range):
                    if flip_qubit:
                        self.qubit.flip()
                    sync()
                    self.qubit_detuned.set_ssb(ssbreg)
                    delay(500)
                    sync()
                    readout.smoothed_constant_pulse(self.pump_time, amp='dynamic',
                                           detune=self.readout_detune_MHz*1e6,
                                           sigma_t=self.smooth_sigma_t)
                    self.qubit_detuned.smoothed_constant_pulse(self.pump_time, amp=self.qubit_amp,
                                            sigma_t=self.smooth_sigma_t)
#                    sync()
#                    self.qubit.pi2_pulse(phase=np.pi/2)
                    sync()
                    delay(self.wait_time)
                    sync()
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
