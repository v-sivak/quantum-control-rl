from init_script import *
from fpga_lib.analysis import fit
from scipy.optimize import curve_fit

class out_and_back_Kerr_cancel(FPGAExperiment):
#    static_tau = FloatParameter(160)
    delay_time = FloatParameter(4e6)
    do_g = BoolParameter(True)
    do_e = BoolParameter(True)
    phase_range_deg = RangeParameter((-10, 20, 51))
    nbar_range = RangeParameter((10,800,41))
    postselect = BoolParameter(True)
    
    detune_Kerr_pump_MHz = FloatParameter(20)
    Kerr_pump_time_ns = IntParameter(200)
    Kerr_pump_amp = FloatParameter(0.05)
    Kerr_pump_ramp_ns = IntParameter(100)

#    fit_func = {'chi_kHz':'linear','detuning_kHz':'linear'}

    def sequence(self):

        self.cavity = cavity
        self.cavity_1 = cavity_1
        
        # setup qubit mode for Kerr cancelling drive
        self.qubit_detuned = qubit_ef
        self.qubit_detuned.set_detune(self.detune_Kerr_pump_MHz*1e6)

        #need to dynamicaly update the phase of the return pulse
        @arbitrary_function(float, float)
        def cos(x):
            return np.cos(np.pi * x)

        @arbitrary_function(float, float)
        def sin(x):
            return np.sin(np.pi * x)

        # here, will assume cavity is pulse out and cavity_2 is pulse back
        @subroutine
        def update_amp_phase_out_and_return(amp_reg, phase_reg):
            cx = FloatRegister()
            sx = FloatRegister()
            cx <<= amp_reg
            sx <<= 0
            sync()
            DynamicMixer[0][0] <<= cx
            DynamicMixer[1][1] <<= cx
            DynamicMixer[0][1] <<= sx
            DynamicMixer[1][0] <<= -sx
            sync()
            delay(2000)
            self.cavity.load_mixer()
            delay(2000)
            cx <<= amp_reg*cos(phase_reg)
            sx <<= amp_reg*sin(phase_reg)
            sync()
            DynamicMixer[0][0] <<= cx
            DynamicMixer[1][1] <<= cx
            DynamicMixer[0][1] <<= sx
            DynamicMixer[1][0] <<= -sx
            sync()
            delay(2000)
            self.cavity_1.load_mixer()
            delay(2000)

        def myexp(start='g'):
            if self.postselect:
                readout(**{'sz_' + start + '_init_measurement':'se'})
                sync()
            if start == 'e':
                qubit.flip()
            sync()
            self.cavity.displace(amp='dynamic') # displace out
            sync()
            self.qubit_detuned.smoothed_constant_pulse(
                    self.Kerr_pump_time_ns, amp=self.Kerr_pump_amp, 
                    sigma_t=self.Kerr_pump_ramp_ns)
            sync()
            self.cavity_1.displace(amp='dynamic') # return to origin
            sync()
            delay(24)

            sync()
            if self.postselect:
                readout(**{'sz_' + start + '_middle_measurement':'se'})
                sync()
            qubit.flip(selective=True)
            sync()
            readout(**{'sz_' + start:'se'})

        #scan phase is in units of pi
        phase_start = (self.phase_range_deg[0] - 180.0)/180.0
        phase_stop = (self.phase_range_deg[1] - 180.0)/180.0
        phase_points = self.phase_range_deg[2]

        self.nbars = np.linspace(*self.nbar_range)
        self.alphas = np.sqrt(self.nbars)

        unit_amp = self.cavity.displace.unit_amp
        dac_amps = unit_amp*self.alphas
        f_arr = Array(dac_amps, float)
        amp_reg = FloatRegister(0.0)
        with scan_array(f_arr, amp_reg, axis_scale=1/unit_amp, plot_label='alpha'):
            if self.do_g:
                with scan_float_register(phase_start, phase_stop, phase_points,
                        axis_scale = 180.0, plot_label='return phase, deg') as phase_reg:
                    update_amp_phase_out_and_return(amp_reg,phase_reg)
                    sync()
                    myexp(start='g')
                    sync()
                    delay(self.delay_time)
            if self.do_e:
                with scan_float_register(-1*phase_start, -1*phase_stop, phase_points,
                        axis_scale = 180.0, plot_label='return phase, deg') as phase_reg:
                    update_amp_phase_out_and_return(amp_reg,phase_reg)
                    sync()
                    myexp(start='e')
                    sync()
                    delay(self.delay_time)


    def fit_gaussian(self, xs, x0, sig, ofs, amp):
        return ofs + amp * np.exp(-(xs - x0)**2 / (2 * sig**2))


    def process_data(self):
        ops = []

        if self.do_g:
            ops.append('sz_g')
            if self.blocks_processed == 1:
                self.results['sz_g'].ax_data[2] = self.results['sz_g'].ax_data[2] + 180.0
        if self.do_e:
            ops.append('sz_e')
            if self.blocks_processed == 1:
                self.results['sz_e'].ax_data[2] = self.results['sz_e'].ax_data[2] - 180.0

        for op in ops:
            if self.postselect:
                init_result = self.results[op + '_init_measurement'].threshold()
                middle_result = self.results[op + '_middle_measurement'].threshold()
                postselected_init = self.results[op].postselect(init_result,[0])[0]
                self.results[op + '_postselected_g'] = postselected_init.postselect(middle_result,[0])[0]
                self.results[op + '_postselected_e'] = postselected_init.postselect(middle_result,[1])[0]
                #replace the rest of the analysis with the postselected version
                op = op + '_postselected_g' if op == 'sz_g' else op + '_postselected_e'

            gaussian_peaks = []
            alphas = self.results[op].ax_data[1]
            for i, alpha in enumerate(alphas):
                phases = self.results[op].ax_data[2]
                return_phase = self.results[op].thresh_mean().data[i]
                if op == 'sz_g' or op == 'sz_g_postselected_g':
                    guess = phases[np.argmax(return_phase)]
                    p0 = (guess, 5., 0, 1.0)
                if op == 'sz_e' or op == 'sz_e_postselected_e':
                    guess = phases[np.argmin(return_phase)]
                    p0 = (guess, 5., 1.0, -1.0)
                popt, pcov = curve_fit(self.fit_gaussian, phases, return_phase, p0=p0)
                gaussian_peaks.append(popt[0])

            self.results[op + '_fit_gaussian_results'] = np.array(gaussian_peaks)
            self.results[op + '_fit_gaussian_results'].ax_data = [self.nbars]


#    def plot(self, fig, data):
#        title = '%s (%s)' % (self.name, self.run_name)
#        nbars = self.nbars
#        ax1 = fig.add_subplot(411)
#        sz_g_phase = self.results['sz_g_postselected_g_fit_gaussian_results'].data
#        g_sweep_phase = self.results['sz_g_postselected_g'].ax_data[2]
#        dx = nbars[1] - nbars[0]
#        dy = g_sweep_phase[1] - g_sweep_phase[0]
#        nbars_plot = np.concatenate([nbars, [nbars[-1] + dx]])
#        g_sweep_phase= np.concatenate([g_sweep_phase, [g_sweep_phase[-1] + dy]])
#        ax1.pcolormesh(nbars_plot - dx/2.0, g_sweep_phase - dy/2.0, self.results['sz_g_postselected_g'].thresh_mean().data.T, vmin=0, vmax=+1, cmap='seismic')
#        ax1.plot(nbars, sz_g_phase, '.',label='data', color='lime')
#        ax1.set_title(title)
#
#        ax2 = fig.add_subplot(412, sharex=ax1)
#        sz_e_phase = self.results['sz_e_postselected_e_fit_gaussian_results'].data
#        e_sweep_phase = self.results['sz_e_postselected_e'].ax_data[2]
#        dy = e_sweep_phase[1] - e_sweep_phase[0]
#        e_sweep_phase= np.concatenate([e_sweep_phase, [e_sweep_phase[-1] + dy]])
#        ax2.pcolormesh(nbars_plot - dx/2.0, e_sweep_phase - dy/2.0, self.results['sz_e_postselected_e'].thresh_mean().data.T, vmin=0, vmax=+1, cmap='seismic')
#        ax2.plot(nbars, sz_e_phase, '.',label='data', color='lime')
#        ax2.set_xlabel('alpha')
#        #ax2.grid()
#
#        ax3 = fig.add_subplot(413, sharex=ax2)
#        ax3.plot(nbars, self.results['chi_kHz'].data, 'o', label='chi kHz')
#        chi_kHz = self.fit_params['chi_kHz:b']
#        chi_prime_kHz = self.fit_params['chi_kHz:a']
#        predicted_chi = chi_kHz + chi_prime_kHz*nbars
#        ax3.plot(nbars, predicted_chi, '-', label='linear fit')
#        ax3.plot([],[], ' ', label = 'chi_kHz = %.3f ' % (chi_kHz))
#        ax3.plot([],[], ' ', label = 'chi_prime_Hz = %.3f' % (chi_prime_kHz*1e3))
#        ax3.grid()
#        ax3.legend(prop={'size': 6})
#
#        ax4 = fig.add_subplot(414, sharex=ax3)
#        ax4.plot(nbars, self.results['detuning_kHz'].data, 'o', label='detuning kHz')
#        detuning_kHz = self.fit_params['detuning_kHz:b']
#        kerr_kHz = self.fit_params['detuning_kHz:a']
#        predicted_detuning = detuning_kHz + kerr_kHz*nbars
#        ax4.plot(nbars, predicted_detuning, '-', label='linear fit')
#        ax4.plot([],[], ' ', label = 'detuning_kHz = %.3f' % (detuning_kHz))
#        ax4.plot([],[], ' ', label = 'kerr_Hz = %.3f' % (kerr_kHz*1e3))
#        ax4.grid()
#        ax4.legend(prop={'size': 6})
