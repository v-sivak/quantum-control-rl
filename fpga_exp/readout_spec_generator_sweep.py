# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:37:54 2020

@author: Vlad
"""

from init_script import *
from fpga_lib import dsl
import numpy as np
import matplotlib.pyplot as plt
from fpga_lib.get_instruments import get_instruments

class readout_spec_generator_sweep(FPGAExperiment):
    """
    Fast readout spectroscopy with generator triggering.
    Can modify <subsequence>, <initialize> and <finalize> if needed.

    Before running:
        1) To use the generator in sweep mode, make sure the driver has
           modulation option and it is enabled.
        2) FPGA 2V marker output needs to be shifted to 5V for triggering
           Agilent generator.
    """
    delay = IntParameter(1e6)
    loop_delay= IntParameter(1e6)
    card = IntParameter(0)
    marker = IntParameter(1)
    averages = IntParameter(1e3)
    marker_len = FloatParameter(1e3)
    start_f = FloatParameter(9.32e9)
    stop_f = FloatParameter(9.42e9)

    def sequence(self):
        with dsl.Repeat(self.averages):
            self.subsequence()
        sync()
        marker_pulse((self.card, self.marker), self.marker_len)
        delay(self.loop_delay)

    def subsequence(self):
        qubit.flip()
        readout(re_e='se0', im_e='se1')
        delay(self.delay)
        readout(re_g='se0', im_g='se1')
        delay(self.delay)


    def run(self, **kwargs):
        self.averages_per_block = 1
        self.initialize()
        super(readout_spec_generator_sweep, self).run(**kwargs)

    def finish_run(self):
        self.finalize()
        super(readout_spec_generator_sweep, self).finish_run()

    def initialize(self):
        instruments = get_instruments()
        ag = instruments['readout_LO']
        self.freq =  ag.get_frequency()
        ag.set_sweep_frequency_mode('SWE')
        ag.set_sweep_start_frequency(self.start_f + 50e6)
        ag.set_sweep_stop_frequency(self.stop_f + 50e6)
        ag.set_sweep_n_points(self.n_blocks)
        ag.set_sweep_trigger('IMM')
        ag.set_sweep_point_trigger('EXT')

        self.results.create('freq', np.linspace(self.start_f, self.stop_f, self.n_blocks))

    def finalize(self):
        instruments = get_instruments()
        ag = instruments['readout_LO']
        ag.set_sweep_frequency_mode('CW')
        ag.set_frequency(self.freq)

    def process_data(self):
        for s in ['g', 'e']:
            data = self.results['re_'+s].data+1j*self.results['im_'+s].data
            data = data.mean(axis=1)
            self.results.create('phase_'+s, np.angle(data))
            self.results.create('logmag_'+s, 20*np.log10(np.abs(data)))

    def plot(self, fig, data):
        freqs = np.linspace(self.start_f,self.stop_f,self.n_blocks)*1e-6
        ax_phase = fig.add_subplot(211)
        ax_logmag = fig.add_subplot(212, sharex = ax_phase)
        ax_phase.set_ylabel('Phase (unwrapped)')
        ax_logmag.set_ylabel('Log magnitude (dB)')
        ax_logmag.set_xlabel('Frequency (MHz)')
        for s in ['g', 'e']:
            phase = np.unwrap(data['phase_'+s].data)
            logmag = data['logmag_'+s].data
            ax_phase.plot(freqs[:len(phase)], phase, label=s)
            ax_logmag.plot(freqs[:len(logmag)], logmag, label=s)
        ax_phase.legend()
        ax_logmag.legend()
