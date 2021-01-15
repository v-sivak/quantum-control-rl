# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 17:32:45 2021

@author: qulab
"""
from init_script import *

class Rabi_pi2_pulse(FPGAExperiment):
    amp_range = RangeParameter((0, 1.5, 41))
    along_x = BoolParameter(True)
    fit_func = 'sine'
    fixed_fit_params = {'phi': -np.pi/2}
    loop_delay = IntParameter(1e6)


    def sequence(self):
        pulse = qubit.pi2_pulse
        with pulse.scan_amplitude(*self.amp_range, in_phase=self.along_x):
            sync()
            readout(init_state='se')
            pulse(amp='dynamic')
            sync()
            pulse(amp='dynamic')
            sync()
            delay(24)
            readout()
            delay(self.loop_delay)

    def update(self):
        scale = 1 / (self.fit_params['f0'] * 2)
        p = qubit.pi2_pulse
        old_amp = self.run_calib_params['qubit'][p.name.split('.')[-1]]['unit_amp']
        new_amp = old_amp * scale
        self.logger.info('Setting %s amp from %s to %s', p.name, old_amp, new_amp)
        p.unit_amp = new_amp

    def process_data(self):
        init_state = self.results['init_state'].threshold()
        self.results['postselected'] = self.results['default'].postselect(init_state, [0])[0]
        self.results['postselected'].ax_data = self.results['init_state'].ax_data
        self.results['postselected'].labels = self.results['init_state'].labels
