# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 12:50:20 2021

@author: qulab
"""
from init_script import *
from gkp_exp.CD_gate.conditional_displacement_compiler import ConditionalDisplacementCompiler
import numpy as np


class CF_tomography_test(FPGAExperiment):
    qubit_pulse_pad = IntParameter(4)
    tau_ns = IntParameter(50)
    cal_dir = StringParameter(r'D:\DATA\exp\2021-05-13_cooldown\CD_fixed_time_amp_cal')
    beta_range = RangeParameter((-2,2,21))
    loop_delay = IntParameter(4e6)
    
    
    def sequence(self):
        
        
        C = ConditionalDisplacementCompiler(qubit_pulse_pad=self.qubit_pulse_pad)
        CD_params = C.CD_params_fixed_tau_from_cal(1.0, self.tau_ns, self.cal_dir)
        cavity_pulse, qubit_pulse = C.make_pulse(*CD_params)
        cavity_pulse = [cavity_pulse[s]/0.2 for s in [0,1]]
        self.cavity_pulse = cavity_pulse
        
        def CD_tomo(s):
            phase = dict(Re=-np.pi/2.0, Im=0.0)
            sync()
            qubit.pi2_pulse(phase=np.pi/2.0)
            sync()
            cavity.array_pulse(*cavity_pulse, amp='dynamic')
            qubit.array_pulse(*qubit_pulse)
            sync()
            qubit.pi2_pulse(phase=phase[s])
            sync()
            delay(24)
            readout(**{s+'_m':'se'})
            sync()
            delay(self.loop_delay) 

            
        with scan_amplitude(cavity.chan, *self.beta_range, scale=0.2):
            CD_tomo('Re')
            CD_tomo('Im')
            
    
    def process_data(self):
        for s in ['Re', 'Im']:
            self.results[s+'_CF'] = 1 - 2*self.results[s+'_m'].thresh_mean().data
            self.results[s+'_CF'].ax_data = [self.results[s+'_m'].ax_data[1]]
            self.results[s+'_CF'].labels = [self.results[s+'_m'].labels[1]]  
    