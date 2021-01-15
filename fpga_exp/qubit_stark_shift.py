from init_script import *

class qubit_stark_shift(FPGAExperiment):
    detune_range = RangeParameter((-20e6,20e6,41))
    alpha_range = RangeParameter((0, 2, 41))
    selective = BoolParameter(False)
    loop_delay = FloatParameter(1e6)
    software_detune = FloatParameter(0)

    def sequence(self):
        qssb = -50e6
        start = self.detune_range[0] + qssb
        stop = self.detune_range[1] + qssb
        points = self.detune_range[-1]

        with cavity.displace.scan_amplitude(*self.alpha_range):
            with scan_register(start/1e3, stop/1e3, points) as ssbreg:
                qubit.set_ssb(ssbreg)
                delay(2000)
                sync()
                cavity.displace(amp='dynamic')
                sync()
                qubit.flip(selective=self.selective, detune=self.software_detune)
                sync()
                cavity.displace(amp='dynamic', phase=np.pi)
                sync()
                readout()
                delay(self.loop_delay)
