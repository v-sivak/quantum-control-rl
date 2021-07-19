from init_script import *

class qubit_spec(FPGAExperiment):
    start_detune = FloatParameter(-20e6)
    stop_detune = FloatParameter(20e6)
    points = IntParameter(41)
    selective = BoolParameter(False)
    fit_func = 'gaussian'
    loop_delay = FloatParameter(1e6)
    qubit_generator = StringParameter('qubit_LO')

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

    def update(self):
        gen_name = self.run_params['qubit_generator']
        delta_f = self.fit_params['x0']*1e3 - qubit.get_ssb0()
        new_f = self.run_inst_params[gen_name]['frequency'] + delta_f
        new_f = np.around(new_f, -3) #rounds to kHz level for generator
        self.instruments[gen_name].set_frequency(new_f)
        self.logger.info('Moving %s freq by %s kHz to %s GHz', gen_name, delta_f*1e-3, new_f*1e-9)
