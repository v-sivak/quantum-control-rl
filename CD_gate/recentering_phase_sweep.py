# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 14:19:22 2020

@author: qulab
"""
from init_script import *
from scipy.optimize import curve_fit
import numpy as np

def linear(time, freq, offset):
    return -360 * freq * time*1e-9 + offset

class recentering_phase_sweep(FPGAExperiment):
    """
    This will displace the cavity to a given amplitude 'alpha', let it evolve
    for some time, and then attemp to displace back with some phase to recenter.

    Thi allows to find the optimal phase for the backward displacement.

    Should be used in combination with another script which will sweep alpha
    outside fpga gui. THis is done in such way in order to track the expected
    location of the rotated blob and only sweep the angle in some small range
    around that value. If the expected rotaion is small, better use a script
    from Alec which dynamically sweeps amplitude and phase and works faster.

    """
    alpha = FloatParameter(15.0)
    loop_delay = FloatParameter(4e6)
    phase_range_deg = RangeParameter((-30.0, 30.0, 21))
    time_range_ns = RangeParameter((0.0, 800, 11))
    freq_guess = FloatParameter(13e3)
    offset_guess = FloatParameter(-180)
    flip_qubit = BoolParameter(False)



    def sequence(self):

        t_points = self.time_range_ns[2]
        t_start = self.time_range_ns[0]
        delta_t = (self.time_range_ns[1] - self.time_range_ns[0]) / (t_points - 1)

        self.fit_func = {'selective'+str(i) :'gaussian' for i in range(t_points)}

        for i in range(t_points):

            t = t_start + i * delta_t
            phase_expected = linear(t, self.freq_guess, self.offset_guess)

            # scan phase in units of pi
            phase_start = (phase_expected + self.phase_range_deg[0]) / 180.0
            phase_stop = (phase_expected + self.phase_range_deg[1]) / 180.0
            phase_points = self.phase_range_deg[2]

            # sweep phase around the expected value
            with scan_phase(cavity.chan, self.alpha*cavity.displace.unit_amp,
                            phase_start, phase_stop, phase_points,
                            axis_scale=180.0, plot_label='return phase (deg)'):
                sync()
                if self.flip_qubit:
                    qubit.flip()
                    sync()
                self.displace_and_recenter(self.alpha, t)
                sync()
                qubit.flip(selective=True)
                sync()
                readout(**{'selective' + str(i) :'se'})
                delay(self.loop_delay)


    def displace_and_recenter(self, alpha, tau):
#        readout()
#        sync()
        cavity.displace(amp=alpha)
        sync()
        delay(tau)
        sync()
        cavity.displace(amp='dynamic')


    def process_fit_data(self):

        T = self.time_range_ns[2]
        times = np.linspace(*self.time_range_ns)

        mean_phase = [self.fit_params['selective'+str(i)+':x0'] for i in range(T)]
        std_phase = [self.fit_params['selective'+str(i)+':sig'] for i in range(T)]

        self.results['mean_phase'] = np.array(mean_phase)
        self.results['mean_phase'].ax_data = [times]
        self.results['mean_phase'].labels = ['time (ns)']

        self.results['std_phase'] = np.array(std_phase)
        self.results['std_phase'].ax_data = [times]
        self.results['std_phase'].labels = ['time (ns)']

        # fit rotation frequency
        p_guess = [self.freq_guess, self.offset_guess]
        popt, pcov = curve_fit(linear, times, mean_phase, p0=p_guess)
        self.fit_params.update(dict(f0=popt[0], offset=popt[1]))

    def plot(self, fig, data):

        times = self.results['mean_phase'].ax_data[0]
        mean_phase = self.results['mean_phase'].data
        std_phase = self.results['std_phase'].data

        # plot linear phase accumulation
        ax = fig.add_subplot(111)
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Phase (deg)')
        ax.errorbar(times, mean_phase, std_phase, linestyle='None', marker='^')
        ax.plot(times, linear(times, self.fit_params['f0'], self.fit_params['offset']),
                linestyle='-', marker=None)
