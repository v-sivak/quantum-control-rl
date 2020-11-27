from init_script import *

class qubit_spec(FPGAExperiment):
    start_detune = FloatParameter(-20e6)
    stop_detune = FloatParameter(20e6)
    points = IntParameter(41)
    selective = BoolParameter(False)
    fit_func = 'gaussian'
    loop_delay = FloatParameter(1e6)

    def sequence(self):
        qssb = -50e6
        start = self.start_detune + qssb
        stop = self.stop_detune + qssb
        with scan_register(start/1e3, stop/1e3, self.points) as ssbreg:
            qubit.set_ssb(ssbreg)
            delay(2000)
            sync()
            qubit.flip(selective=self.selective)
            readout()
            delay(self.loop_delay)
