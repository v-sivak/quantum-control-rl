from init_script import *
from fpga_lib.analysis import fit
from scipy.optimize import curve_fit

class CD_geometric_phase(FPGAExperiment):
    beta_CD = FloatParameter(1.0)
    alpha_CD = FloatParameter(5.0)
    displacement_range = RangeParameter((0,5,101))
    delay_time = FloatParameter(4e6)
    buffer_time = IntParameter(4)


    def sequence(self):
        
        def CD(beta):
            #note: beta = 2*alpha0*sin(chi*tw)
            def chi(alpha):
                chi = cavity.chi + alpha**2 * cavity.chi_prime
                return 2*np.pi*chi
            corrected_chi = chi(self.alpha_CD)
            self.tw = np.abs(np.arcsin(beta / (2*self.alpha_CD))/corrected_chi)*1e9
            phase = np.angle(beta) + np.pi/2.0
            ratio = np.cos((corrected_chi/2.0)*self.tw*1e-9)
            ratio2 = np.cos(corrected_chi*self.tw*1e-9)

            sync()
            cavity.displace(self.alpha_CD, phase=phase)
            sync()
            delay(self.tw, round=True)
            sync()
            cavity.displace(ratio*self.alpha_CD, phase=phase+pi)
            sync()
#            delay(self.buffer_time)
            qubit.flip()
#            delay(self.buffer_time)
            sync()
            cavity.displace(ratio*self.alpha_CD,  phase=phase+pi)
            sync()
            delay(self.tw, round=True)
            sync()
            cavity.displace(ratio2*self.alpha_CD,  phase=phase)

        def myexp(op='sx'):
#            sync()
#            self.mysystem.prepare()
#            sync()
#            marker_pulse((1,0),48)
            sync()
            qubit.rotate(angle=pi/2.0, phase = np.pi/2.0)
            sync()
            CD(1j*self.beta_CD)
            sync()
            cavity.displace(amp='dynamic')
            sync()
            CD(1j*self.beta_CD)
            sync()
            cavity.displace(amp='dynamic', phase=pi)
            sync()
            if op == 'sx':
                qubit.rotate(angle=pi/2.0, phase = -pi/2.0)
            elif op == 'sy':
                qubit.rotate(angle=pi/2.0, phase = 0.0)
            sync()
            delay(24)
            readout(**{op:'se'})

        with cavity.displace.scan_amplitude(*self.displacement_range):
            sync()
            myexp(op='sx')
            sync()
            delay(self.delay_time)

            sync()
            myexp(op='sy')
            sync()
            delay(self.delay_time)

    #todo: include dephasing
    def expected_sx_sy(self, alpha):
        phase = (2*alpha*self.beta_CD)
        sx = np.cos(phase)
        sy = np.sin(phase)
        return sx, sy

    def process_data(self):
        for op in ['sx','sy']:
            self.results[op + '_full'] = Result()
            self.results[op + '_full'].ax_data = self.results[op].ax_data[1:]
            self.results[op + '_full'].labels = ['alpha','<' + op + '>']
            self.results[op + '_full'].data = 1 - 2*self.results[op].thresh_mean().data

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
