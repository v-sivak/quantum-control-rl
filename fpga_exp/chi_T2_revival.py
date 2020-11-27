from init_script import *
import numpy as np


class chi_T2_revival(FPGAExperiment):
    points = IntParameter(41)
    t_max = FloatParameter(1e5)
    echo = BoolParameter(False)
    detune = FloatParameter(200e3)
    phi = FloatParameter(45.0)
    #negate_update_detune = BoolParameter(False)
    qubit_generator = StringParameter('qubit_LO')
    delay_time = FloatParameter(1e6)
    displacement_alpha = FloatParameter(1.0)

    fit_func = {'rotated' : 'exp_decay_sine',
                'default' : 'exp_decay_sine'}
    fit_fmt = {'f0' : ('%.3f kHz', 1e-6),
               'tau' : ('%.2f us', 1e3)}

    def sequence(self):
        scale = 2 if self.echo else 1
        max_delay = self.t_max / scale
        phase_scale = 0.5*self.detune*self.t_max*1e-9
        delta_phase = -2.0/(self.points-1)
        phase_reg = FloatRegister()
        phase_reg <<= 0.0

        @arbitrary_function(float, float)
        def cos_2(x):
            return qubit.pulse.unit_amp/2 * np.cos(2*np.pi*phase_scale*x)

        @arbitrary_function(float, float)
        def sin_2(x):
            return qubit.pulse.unit_amp/2 * np.sin(2*np.pi*phase_scale*x)

        delay(100)
        sync()
        with scan_length(0, max_delay, self.points, axis_scale=scale) as dynlen:
            DynamicMixer[0][0] <<= cos_2(phase_reg)
            DynamicMixer[1][0] <<= sin_2(phase_reg)
            DynamicMixer[0][1] <<= 0.0-sin_2(phase_reg)
            DynamicMixer[1][1] <<= cos_2(phase_reg)
            sync()
            qubit.load_mixer()
            sync()
            delay(400)
            readout(init_state='se')
            sync()
            delay(100)
            cavity.displace(self.displacement_alpha)
            sync()
            qubit.rotate_y(pi/2)
            sync()
            delay(dynlen)
            if self.echo:
                qubit.rotate_x(pi)
                sync()
                delay(dynlen)
            qubit.rotate_y('dynamic')
            sync()
            readout()
            delay(self.delay_time)
            phase_reg += delta_phase


    def process_data(self):
        #postselect on initial measurement
        init_state = self.results['init_state'].threshold()
        self.results['postselected'] = self.results['default'].postselect(
                init_state, [0])[0].thresh_mean().data
        self.results['postselected'].ax_data = self.results['init_state'].ax_data[1:]
        self.results['postselected'].labels = self.results['init_state'].labels[1:]

        self.results['im'] = np.imag(np.average(self.results['default'].data, axis=0))
        self.results['im'].ax_data = self.results['default'].ax_data[1:]
        self.results['im'].labels = self.results['default'].labels[1:]
        self.results['re'] = np.real(np.average(self.results['default'].data, axis=0))
        self.results['re'].ax_data = self.results['default'].ax_data[1:]
        self.results['re'].labels = self.results['default'].labels[1:]
        self.results['rotated'] = np.imag(np.exp(1.0j*self.phi*np.pi/180.0)*np.average(self.results['default'].data, axis=0))
        self.results['rotated'].ax_data = self.results['default'].ax_data[1:]
        self.results['rotated'].labels = self.results['default'].labels[1:]


    def update(self):
        gen_name = self.run_params['qubit_generator']
        delta_f = self.fit_params['rotated:f0']*1e9 - self.run_params['detune']
        new_f = self.run_inst_params[gen_name]['frequency'] + delta_f
        new_f = np.around(new_f, -3) #rounds to kHz level for generator
        self.instruments[gen_name].set_frequency(new_f)
        self.logger.info('Moving %s freq by %s MHz to %s GHz', gen_name, delta_f*1e-6, new_f*1e-9)
