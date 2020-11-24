from init_script import *

class cavity_spec(FPGAExperiment):
    start_detune = FloatParameter(-20e6)
    stop_detune = FloatParameter(20e6)
    selective = BoolParameter(True)
    pi_pulse = BoolParameter(False)
    points = IntParameter(41)
    loop_delay = IntParameter(1e6)
    displacement_alpha = FloatParameter(1.0)
    fit_func = 'gaussian'

    def sequence(self):
        start = self.start_detune - 50e6
        stop = self.stop_detune - 50e6
        with cavity.scan_ssb(start, stop, self.points):
            #system.prepare()
            sync()
            if self.pi_pulse:
                qubit.flip()
            #constant_pulse((1,0), 200e3)
            #cavity.displace(1, selective=self.selective)
            cavity.displace(self.displacement_alpha)
            sync()
            qubit.flip(selective=True)
            readout()
            #system.relax(1)
            delay(self.loop_delay)
