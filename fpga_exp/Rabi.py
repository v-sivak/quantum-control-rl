from init_script import *

class Rabi(FPGAExperiment):
    amp_range = RangeParameter((0, 1.5, 41))
    disp = FloatParameter(0)
    detune = FloatParameter(0)
    selective = BoolParameter(False)
    along_x = BoolParameter(True)
    fit_func = 'sine'
    fixed_fit_params = {'phi': -np.pi/2}
    loop_delay = IntParameter(1e6)


    def sequence(self):
        pulse = qubit.selective_pulse if self.selective else qubit.pulse
        with pulse.scan_amplitude(*self.amp_range, in_phase=self.along_x):
            sync()
            readout(init_state='se')
            pulse(amp='dynamic')
            sync()
            delay(24)
            readout()
            delay(self.loop_delay)

    def update(self):
        scale = 1 / (self.fit_params['f0'] * 2)
        p = qubit.selective_pulse if self.run_params['selective'] else qubit.pulse
        old_amp = self.run_calib_params['qubit'][p.name.split('.')[-1]]['unit_amp']
        new_amp = old_amp * scale
        self.logger.info('Setting %s amp from %s to %s', p.name, old_amp, new_amp)
        p.unit_amp = new_amp

    def process_data(self):
        init_state = self.results['init_state'].threshold()
        self.results['postselected'] = self.results['default'].postselect(init_state, [0])[0]
        self.results['postselected'].ax_data = self.results['init_state'].ax_data
        self.results['postselected'].labels = self.results['init_state'].labels
