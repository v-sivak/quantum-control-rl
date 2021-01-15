from init_script import *

class cavity_spec(FPGAExperiment):
    start_detune = FloatParameter(-20e6)
    stop_detune = FloatParameter(20e6)
    selective = BoolParameter(True)
    points = IntParameter(41)
    loop_delay = IntParameter(1e6)
    fit_func = 'gaussian'

    def sequence(self):
        start = self.start_detune - 50e6
        stop = self.stop_detune - 50e6
        with cavity.scan_ssb(start, stop, self.points):
            sync()
            delay(2000)
            sync()
            constant_pulse(cavity.chan, 500e3, amp=0.00005)
            sync()
            delay(100)
            qubit.flip(selective=False)
            sync()
            readout()
            delay(self.loop_delay)
