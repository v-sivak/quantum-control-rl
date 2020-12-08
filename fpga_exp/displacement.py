from init_script import *

class displacement(FPGAExperiment):
    amp_range = RangeParameter((0, 1.5, 41))
    fit_func = 'displacement_cal'
    fixed_fit_params = {'n': 0}
    loop_delay = IntParameter(2e6)

    def sequence(self):
        pulse = cavity.displace
        with pulse.scan_amplitude(*self.amp_range):
            pulse(amp='dynamic')
            sync()
            delay(100)
            qubit.flip(selective=True)
            sync()
            readout()
            delay(self.loop_delay)
            #system.relax(self.stop)

    def update(self):
        p = cavity.displace
        old_amp = self.run_calib_params['cavity'][p.name.split('.')[-1]]['unit_amp']
        new_amp = old_amp / self.fit_params['dispscale']
        self.logger.info('Setting %s amp from %s to %s', p.name, old_amp, new_amp)
        p.unit_amp = new_amp
