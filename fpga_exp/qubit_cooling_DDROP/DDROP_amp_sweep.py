from init_script import *

class DDROP_amp_sweep(FPGAExperiment):
    readout_amp_range = RangeParameter((0, 0.02, 61))
    qubit_amp_range = RangeParameter((0, 0.05, 101))
    loop_delay = FloatParameter(500e3)
    detune = IntParameter(463e3)
    tau = IntParameter(4e3)
    flip_qubit = BoolParameter(False)

    def sequence(self):
        readout_aux.set_detune(self.detune)
        with scan_amplitude(readout_aux.chan, *self.readout_amp_range):
            with scan_amplitude(qubit.chan, *self.qubit_amp_range):
                if self.flip_qubit:
                    qubit.flip()
                sync()
                readout_aux.smoothed_constant_pulse(self.tau, amp='dynamic', sigma_t=100)
                qubit.smoothed_constant_pulse(self.tau, amp='dynamic', sigma_t=100)
                sync()
                delay(2e3)
                readout()
                delay(self.loop_delay)
