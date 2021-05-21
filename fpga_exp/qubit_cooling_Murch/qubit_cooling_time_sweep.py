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
    smooth_sigma_t = IntParameter(40)
    time_range = RangeParameter((0,1000,21))
    loop_delay = IntParameter(500e3)
    flip_qubit = BoolParameter(False)

    def sequence(self):
        self.qubit_detuned = qubit_ef
        self.qubit = qubit

        self.readout_detuned = readout_aux
        self.readout = readout

        sync()
        self.readout_detuned.set_detune(self.readout_detune_MHz*1e6)
        self.qubit_detuned.set_detune(self.qubit_detune_MHz*1e6)
        sync()

        def cool_Murch(duration):
            sync()
            self.readout_detuned.smoothed_constant_pulse(
                    duration, amp=self.readout_amp, sigma_t=self.smooth_sigma_t)
            self.qubit_detuned.smoothed_constant_pulse(
                    duration, amp=self.qubit_amp, sigma_t=self.smooth_sigma_t)
            sync()


        def exp(state):

            sync()
            self.readout_detuned.set_detune(self.readout_detune_MHz*1e6)
            self.qubit_detuned.set_detune(self.qubit_detune_MHz*1e6)
            self.qubit.set_detune(0)
            sync()
            
#            for tau in np.linspace(*self.time_range):
            with scan_length(*self.time_range) as tau:
                sync()
                if self.flip_qubit:
                    self.qubit.flip()
                else:
                    delay(24)
                sync()
                cool_Murch(tau)
                sync()
                if state=='z':
                    delay(24)
                if state=='x':
                    self.qubit.pi2_pulse(phase=-np.pi/2)
                if state=='y':
                    self.qubit.pi2_pulse()
                sync()
                self.readout(**{state:'se'})
                sync()
                delay(self.loop_delay)

        exp('x')
        exp('y')
        exp('z')


    def plot(self, fig, data):
        sigma = {}
        for state in ['x','y','z']:
            sigma[state] = 1 - 2 * self.results[state].threshold().mean(axis=0).data

        purity = sum([s**2 for k,s in sigma.items()])

        times = self.results['x'].ax_data[1]
        times_label = self.results['x'].labels[1]

        ax = fig.add_subplot(111)
        ax.set_xlabel(times_label)
        ax.set_ylim(-1.1, 1.1)
        for state in ['x','y','z']:
            ax.plot(times, sigma[state], label=state, marker='.')
        ax.plot(times, np.sqrt(purity), color='k', marker='.', label='sqrt(purity)')
        ax.plot(times, np.ones_like(times), linestyle='--')
        ax.legend()
