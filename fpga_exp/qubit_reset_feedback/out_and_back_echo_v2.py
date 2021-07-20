# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:05:00 2021

@author: qulab
"""

from init_script import *
from fpga_lib.analysis import fit
from scipy.optimize import curve_fit

class out_and_back_echo_v2(FPGAExperiment):
    """
    In this version the echo pulse is AFTER the readout, and we sweep the duration
    of the feedback delay (roughly should be equal to the duration of the readout).
    """
    phase_range_deg = RangeParameter((-10, 20, 51))
    loop_delay = IntParameter(4e6)
    alpha = FloatParameter(1.5)
    reps = IntParameter(4)

    feedback_delay_range = RangeParameter((600,1200,41))
    final_delay = IntParameter(0)


    def sequence(self):

        self.cavity = cavity
        self.cavity_1 = cavity_1

        @subroutine
        def reset_with_echo(feedback_delay):
            sync()
            readout(wait_result=True, log=False, sync_at_beginning=False)
            sync()
            qubit.flip() # echo pulse
            delay(feedback_delay, round=True)
            if_then_else(qubit.measured_low(), 'flip', 'wait')
            label_next('flip')
            qubit.flip()
            goto('continue')
            label_next('wait')
            delay(qubit.pulse.length)
            label_next('continue')
            delay(self.final_delay)
            sync()

        #need to dynamicaly update the phase of the return pulse
        @arbitrary_function(float, float)
        def cos(x):
            return np.cos(np.pi * x)

        @arbitrary_function(float, float)
        def sin(x):
            return np.sin(np.pi * x)

        # here, will assume cavity is pulse out and cavity_1 is pulse back
        @subroutine
        def update_phase_out_and_return(phase_reg):
            dac_amp = self.alpha * self.cavity.displace.unit_amp
            cx = FloatRegister()
            sx = FloatRegister()
            cx <<= dac_amp
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
            cx <<= dac_amp*cos(phase_reg)
            sx <<= dac_amp*sin(phase_reg)
            sync()
            DynamicMixer[0][0] <<= cx
            DynamicMixer[1][1] <<= cx
            DynamicMixer[0][1] <<= sx
            DynamicMixer[1][0] <<= -sx
            sync()
            delay(2000)
            self.cavity_1.load_mixer()
            delay(2000)


        def exp(s, feedback_delay):
            if s=='e':
                qubit.flip()
            else:
                delay(qubit.pulse.length)
            sync()
            self.cavity.displace(amp='dynamic') # displace out
            sync()
            reset_with_echo(feedback_delay)
            sync()
            self.cavity_1.displace(amp='dynamic') # return to origin
            sync()


        #scan phase is in units of pi
        phase_start = (self.phase_range_deg[0] - 180.0)/180.0
        phase_stop = (self.phase_range_deg[1] - 180.0)/180.0
        phase_points = self.phase_range_deg[2]

        self.feedback_delays = np.linspace(*self.feedback_delay_range)
        for s in ['g', 'e']:
            # There is some weird bug when doing this with scan_length
            for feedback_delay in self.feedback_delays:
                with scan_float_register(phase_start, phase_stop, phase_points,
                        axis_scale = 180.0, plot_label='return phase, deg') as phase_reg:
                    update_phase_out_and_return(phase_reg)
                    sync()
                    with Repeat(self.reps):
                        sync()
                        exp(s, feedback_delay)
                    sync()
                    qubit.flip(selective=True)
                    sync()
                    readout(**{s:'se'})
                    delay(self.loop_delay)


    def fit_gaussian(self, xs, x0, sig, ofs, amp):
        return ofs + amp * np.exp(-(xs - x0)**2 / (2 * sig**2))


    def process_data(self):

        if self.blocks_processed == 1:
            for s in ['g','e']:
                self.results[s].ax_data[2] += 180.0
                self.results[s].ax_data[1] = self.feedback_delays
                self.results[s].labels[1] = 'feedback delay [ns]'

        for s in ['g','e']:
            gaussian_peaks = []
            for i in range(self.feedback_delay_range[-1]):
                phases = self.results[s].ax_data[2]
                data = self.results[s].thresh_mean().data[i]
                guess = phases[np.argmax(data)]
                p0 = (guess, 5., 0, 1.0)

                popt, pcov = curve_fit(self.fit_gaussian, phases, data, p0=p0)
                gaussian_peaks.append(popt[0])

            self.results[s + '_fit_gaussian'] = np.array(gaussian_peaks)
            self.results[s + '_fit_gaussian'].ax_data = [self.feedback_delays]
            self.results[s + '_fit_gaussian'].labels = ['feedback delay [ns]']


    def plot(self, fig, data):
        ax = fig.add_subplot(111)
        ax.set_xlabel('feedback delay [ns]')
        ax.set_ylabel('Phase [deg]')
        for s in ['g','e']:
            ax.plot(self.feedback_delays, self.results[s + '_fit_gaussian'].data,
                    marker='.')
