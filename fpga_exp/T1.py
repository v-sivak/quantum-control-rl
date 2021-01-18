from init_script import *
from fpga_lib.analysis import fit

class T1(FPGAExperiment):
    points = IntParameter(101)
    t_max = FloatParameter(1e5)
    ro_drive_amp = FloatParameter(0)
    fit_func = 'exp_decay'
    fit_fmt = {'tau': ('%.1f us', 1e3)}
    loop_delay = IntParameter(1e6)
    n_repeat = IntParameter(1)
    bin_size = IntParameter(0)

    def sequence(self):
        with scan_length(0, self.t_max, self.points, axis_scale=self.n_repeat) as dynlen:
            sync()
            system.cool_qubit()
            sync()
            qubit.flip()
            sync()
            if self.ro_drive_amp:
                for _ in range(self.n_repeat):
                    readout.constant_pulse(dynlen, self.ro_drive_amp, sigma_t=100)
                delay(1000)
            else:
                for _ in range(self.n_repeat):
                    delay(dynlen)
            readout()
            delay(self.loop_delay)

    def update(self):
        qubit.t1 = self.fit_params['tau']
