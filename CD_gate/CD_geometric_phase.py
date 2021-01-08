from init_script import *
import numpy as np

class CD_geometric_phase(FPGAExperiment):
    beta_CD = FloatParameter(1.0)
    alpha_CD = FloatParameter(5.0)
    displacement_range = RangeParameter((0,5,101))
    delay_time = FloatParameter(4e6)

    fit_func = {'sx_postselected' : 'sine',
                'sy_postselected' : 'sine'}


    def sequence(self):

        self.cavity = cavity_1

        def CD(beta):
            #note: beta = 2*alpha0*sin(chi*tw)
            chi = 2*pi* self.cavity.chi
            self.tw = np.abs(np.arcsin(beta / (2*self.alpha_CD)) / chi) * 1e9
            self.tw -= 24 # padding of fast displacement pulses
            phase = np.angle(beta) - np.pi/2.0
            ratio = np.cos(chi/2.0*self.tw*1e-9)
            ratio2 = np.cos(chi*self.tw*1e-9)

            sync()
            self.cavity.displace(self.alpha_CD, phase=phase)
            sync()
            delay(self.tw, round=True)
            sync()
            self.cavity.displace(ratio*self.alpha_CD, phase=phase+pi)
            sync()
            qubit.flip()
            sync()
            self.cavity.displace(ratio*self.alpha_CD,  phase=phase+pi)
            sync()
            delay(self.tw, round=True)
            sync()
            self.cavity.displace(ratio2*self.alpha_CD,  phase=phase)

        def myexp(op='sx'):
            readout(**{op+'_init_state':'se'})
            sync()
            qubit.rotate(angle=np.pi/2.0, phase = np.pi/2.0)
            sync()
            CD(1j*self.beta_CD)
            sync()
            self.cavity.displace(amp='dynamic')
            sync()
            CD(1j*self.beta_CD)
            sync()
            self.cavity.displace(amp='dynamic', phase=np.pi)
            sync()
            if op == 'sx':
                qubit.rotate(angle=np.pi/2.0, phase = -np.pi/2.0)
            elif op == 'sy':
                qubit.rotate(angle=np.pi/2.0, phase = 0.0)
            sync()
            delay(24)
            readout(**{op:'se'})

        with self.cavity.displace.scan_amplitude(*self.displacement_range):
            sync()
            myexp(op='sx')
            sync()
            delay(self.delay_time)

            sync()
            myexp(op='sy')
            sync()
            delay(self.delay_time)


    def expected_sx_sy(self, alpha):
        phase = 2*alpha*self.beta_CD
        sx = np.cos(phase)
        sy = -np.sin(phase)
        return sx, sy

    def process_data(self):

        for op in ['sx','sy']:
            init_state = self.results[op+'_init_state'].threshold()
            self.results[op + '_postselected'] = Result()
            self.results[op + '_postselected'] = self.results[op].postselect(init_state, [0])[0]
            self.results[op + '_postselected'].ax_data = self.results[op].ax_data
            self.results[op + '_postselected'].labels = self.results[op].labels

            self.results[op + '_full'] = Result()
            self.results[op + '_full'].ax_data = self.results[op].ax_data[1:]
            self.results[op + '_full'].labels = ['alpha','<' + op + '>']
            self.results[op + '_full'].data = 1 - 2*self.results[op + '_postselected'].thresh_mean().data

        self.results['sqrt(sx^2 + sy^2)'] = Result()
        self.results['sqrt(sx^2 + sy^2)'].ax_data = self.results['sx_full'].ax_data
        self.results['sqrt(sx^2 + sy^2)'].labels = self.results['sx_full'].labels
        self.results['sqrt(sx^2 + sy^2)'].data = np.sqrt(self.results['sx_full'].data**2 + self.results['sy_full'].data**2)

    def plot(self, fig, data):
        title = '%s (%s)' % (self.name, self.run_name)
        ax1 = fig.add_subplot(111)
        sx = self.results['sx_full'].data
        sy = self.results['sy_full'].data
        alphas = self.results['sx_full'].ax_data[0]
        alphas_interp = np.linspace(alphas[0], alphas[-1], 401)
        expected_sx, expected_sy = self.expected_sx_sy(alphas_interp)
        ax1.plot(alphas, sx, 'o', label='sx')
        ax1.plot(alphas, sy, 'o', label='sy')
        ax1.plot(alphas_interp, expected_sx, label='sx_expected')
        ax1.plot(alphas_interp, expected_sy, label='sy_expected')
        ax1.set_title(title)
        ax1.grid()
        ax1.legend()

    def update(self):
        x = np.sqrt(self.fit_params['sx_postselected:f0']/(self.beta_CD/np.pi))
        p = self.cavity.displace
        old_amp = self.run_calib_params['cavity'][p.name.split('.')[-1]]['unit_amp']
        new_amp = old_amp / x
        self.logger.info('Setting %s amp from %s to %s', p.name, old_amp, new_amp)
        p.unit_amp = new_amp
