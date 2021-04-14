# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:50:18 2021

@author: qulab
"""
from init_script import *
import numpy as np

class state_prep_wigner_reward_fpga(FPGAExperiment):
    """ State preparation with Wigner reward."""
    loop_delay = FloatParameter(4e6)
    batch_size = IntParameter(10)
    N_alpha = IntParameter(100)
    opt_file = StringParameter('')

    def sequence(self):
        # load pulse sequences and phase space points from file
        params = np.load(self.opt_file)
        self.alphas, self.targets = params['alphas'], params['targets']
        self.qubit_pulses = [(p.real, p.imag) for p in params['qubit_pulses']]
        self.cavity_pulses = [(p.real, p.imag) for p in params['cavity_pulses']]        
        
        @arbitrary_function(float, float)
        def cos(x):
            return np.cos(np.pi * x)
    
        @arbitrary_function(float, float)
        def sin(x):
            return np.sin(np.pi * x)

        @subroutine
        def update_dynamic_mixer(amp_reg, phase_reg):
            cx = FloatRegister()
            sx = FloatRegister()
            cx <<= amp_reg * cos(phase_reg)
            sx <<= amp_reg * sin(phase_reg)
            sync()
            DynamicMixer[0][0] <<= cx
            DynamicMixer[1][1] <<= cx
            DynamicMixer[0][1] <<= sx
            DynamicMixer[1][0] <<= -sx
            sync()
            delay(2000)
            cavity.load_mixer()
            delay(2000)

        amp_reg = FloatRegister(0.0)
        phase_reg = FloatRegister(0.0)        
        unit_amp = cavity.displace.unit_amp
        dac_amps = unit_amp * np.abs(self.alphas)
        dac_amps_arr = Array(dac_amps, float)
        phase_arr = Array(np.angle(self.alphas) / np.pi, float)

        @subroutine
        def measure_parity():
            sync()
            qubit.pi2_pulse(phase=np.pi/2)
            delay(0.5 / cavity.chi * 1e9, round=True)
            qubit.pi2_pulse(phase=-np.pi/2)
            sync()
            readout(m2='se')
    
        @subroutine
        def reward_circuit():
            delay(2000)
            readout(m1='se')
            cavity.displace('dynamic', phase=np.pi)
            measure_parity()
            delay(self.loop_delay)

        @subroutine
        def init_circuit():
            readout(m0='se')
        
        def control_circuit(i):
            sync()
            qubit.array_pulse(*self.qubit_pulses[i])
            cavity.array_pulse(*self.cavity_pulses[i])
            sync()

        j_range = (0, self.N_alpha-1, self.N_alpha)
        
        for i in range(self.batch_size):
            with scan_register(*j_range) as j:
                # update dynamic mixer for tomography
                amp_reg <<= dac_amps_arr[j]
                phase_reg <<= phase_arr[j]
                update_dynamic_mixer(amp_reg, phase_reg)
                
                # initialize, run control circuit and collect reward
                init_circuit()
                control_circuit(i)
                reward_circuit()
    

    def plot(self, fig, data):
        ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
        
        # Plot scattered phase space points and IDEAL Wigner values
        ax1.scatter(self.alphas.real, self.alphas.imag, marker="o", s=12, 
           c=self.targets, cmap='seismic', vmin=-1, vmax=+1)
        ax1.set_title('Ideal')
        ax1.set_aspect('equal')
        
        # Plot scattered phase space points and MEASURED Wigner values
        m2 = 1. - 2*np.squeeze(self.results['m2'].threshold().data)
        avg_values = np.mean(m2, axis=(0,1))
        
        ax2.scatter(self.alphas.real, self.alphas.imag, marker="o", s=12, 
           c=avg_values, cmap='seismic', vmin=-1, vmax=+1)
        ax2.set_title('Measured')
        ax2.set_aspect('equal')