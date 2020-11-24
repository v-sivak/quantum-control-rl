from init_script import *

class displacement(FPGAExperiment):
    start = FloatParameter(0)
    stop = FloatParameter(1.5)
    steps = IntParameter(41)
    selective = BoolParameter(False)
    fit_func = 'displacement_cal'
    fixed_fit_params = {'n': 0}
    loop_delay = IntParameter(2e6)
    
    def sequence(self):
        pulse = cavity.selective_pulse if self.selective else cavity.pulse
        with pulse.scan_amplitude(self.start, self.stop, self.steps):
            pulse(amp='dynamic')
            sync()
            delay(100)
            qubit.flip(selective=True)
            readout()
            delay(self.loop_delay)
            #system.relax(self.stop)

    def update(self):
        p = cavity.selective_pulse if self.run_params['selective'] else cavity.pulse
        old_amp = self.run_calib_params['cavity'][p.name.split('.')[-1]]['unit_amp']
        new_amp = old_amp / self.fit_params['dispscale']
        self.logger.info('Setting %s amp from %s to %s', p.name, old_amp, new_amp)
        p.unit_amp = new_amp
