from init_script import *

class qubit_stark_shift(FPGAExperiment):
    start_detune = FloatParameter(-20e6)
    stop_detune = FloatParameter(20e6)
    points = IntParameter(41)
    alpha_range = RangeParameter((0, 2, 41))
    selective = BoolParameter(False)
    loop_delay = FloatParameter(1e6)
    software_detune = FloatParameter(0)

    def sequence(self):
        qssb = -50e6
        start = self.start_detune + qssb
        stop = self.stop_detune + qssb

        with cavity.displace.scan_amplitude(*self.alpha_range):
            with scan_register(start/1e3, stop/1e3, self.points) as ssbreg:
                qubit.set_ssb(ssbreg)
                delay(2000)
                sync()
                cavity.displace(amp='dynamic')
                sync()
                qubit.flip(selective=self.selective, detune=self.software_detune)
                sync()
                cavity.displace(amp='dynamic', phase=pi)
                sync()
                readout()
                delay(self.loop_delay)
