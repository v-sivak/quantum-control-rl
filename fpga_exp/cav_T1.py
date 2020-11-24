from init_script import *

class cav_T1(FPGAExperiment):
    points = IntParameter(41)
    t_max = FloatParameter(1e5)
    init_disp = FloatParameter(1)
    loop_delay=IntParameter(2e6)
    delay_repeats = IntParameter(2)
    fit_func = 'cohstate_decay'
    fixed_fit_params = {'n': 0}
    fit_fmt = {'tau': ('%.1f us', 1e3)}

    def sequence(self):
        with scan_length(0, self.t_max, self.points, 
                         axis_scale=float(self.delay_repeats)) as dynlen:
            delay(24)
            cavity.displace(self.init_disp)
            sync()
            for i in range(self.delay_repeats):
                delay(dynlen)
            qubit.flip(selective=True)
            readout()
            delay(self.loop_delay)

    def update(self):
        cavity.t1 = self.fit_params['tau']