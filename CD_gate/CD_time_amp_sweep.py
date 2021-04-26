# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 09:32:29 2021

@author: qulab
"""

import numpy as np
from init_script import *
from conditional_displacement_compiler import ConditionalDisplacementCompiler
from scipy.optimize import curve_fit

class CD_time_amp_sweep(FPGAExperiment):
    """
    This experiment sweeps the amplitude of displacement in CD gate for
    each gate duration. The CD gate is simple with no amplitude or phase
    corrections. After CD gate the cavity is returned to vacuum with a
    displacement, and selective qubit pulses is used to verify this.

    """
    beta = FloatParameter(1.)
    cal_dir = StringParameter(r'C:\code\gkp_exp\CD_gate\storage_rotation_freq_vs_nbar')
    tau_range_ns = RangeParameter((400,50,21))
    loop_delay = FloatParameter(4e6)
    flip_qubit = BoolParameter(False)
    amp_range = RangeParameter((1,20,11))
    reps = IntParameter(1)

    def sequence(self):
        tau_ns = np.linspace(*self.tau_range_ns)
        return_phase = np.pi/2.0 if not self.flip_qubit else -np.pi/2.0
        C = ConditionalDisplacementCompiler(cal_dir=self.cal_dir)
        self.alpha_const_chi, self.alpha_from_cal = [], []
        for tau in tau_ns:
            # predicted optimal displacement amplitudes
            self.alpha_const_chi.append(np.abs(C.CD_params_fixed_tau(self.beta, tau)[1]))
#            self.alpha_from_cal.append(C.CD_params_improved(self.beta, tau)[0])
            # create CD pulse whose displacement amplitude will be scanned
            cavity_pulse, qubit_pulse = C.make_pulse(tau, 1., 0., 0.)
            cavity_pulse = [cavity_pulse[s]/cavity.displace.unit_amp for s in [0,1]]
            extra_phase = 0.
            with scan_amplitude(cavity.chan, *self.amp_range, scale=cavity.displace.unit_amp):
                readout(init_state='se')
                sync()
                if self.flip_qubit:
                    qubit.flip()
                for j in range(self.reps): #with Repeat(self.reps):
                    sync()
                    qubit.array_pulse(*qubit_pulse)
                    cavity.array_pulse(*cavity_pulse, amp='dynamic')
                    sync()
                    cavity.displace(self.beta/2.0, phase=return_phase-extra_phase)
                    sync()
                    qubit.flip()
                qubit.flip(selective=True)
                sync()
                readout()
                delay(self.loop_delay)

    def fit_gaussian(self, xs, x0, sig, ofs, amp):
        return ofs + amp * np.exp(-(xs - x0)**2 / (2 * sig**2))

    def process_data(self):
        # change axis label
        if self.blocks_processed >= 1:
            self.results['default'].ax_data[1] = np.linspace(*self.tau_range_ns)
            self.results['default'].labels[1] = 'Delay (ns)'

        # postselect on initial msmt
        self.results['postselected'] = Result()
        init_state = self.results['init_state'].threshold()
        self.results['postselected'] = self.results['default'].postselect(
                init_state, [0])[0].thresh_mean().data
        self.results['postselected'].ax_data = self.results['default'].ax_data[1:]
        self.results['postselected'].labels = self.results['default'].labels[1:]

        # create slices for each gate time
        times = np.linspace(*self.tau_range_ns)
        for i, t in enumerate(times):
            name = 'time_slice_' + str(t) + '_ns'
            self.results[name] = Result()
            self.results[name].data = self.results['default'].data[:,i,:].mean(axis=0)
            self.results[name].ax_data = [self.results['default'].ax_data[2]]
            self.results[name].labels = [self.results['default'].labels[2]]

        # fit best amplitude for each gate time
        # NOTE: fitting to gaussian is not correct, but taking argmax is noisy
        optimal_amps = np.zeros_like(times)
        for i, t in enumerate(times):
            try:
                name = 'time_slice_' + str(t) + '_ns'
                p0 = (10., 10., 5., 0.01 if self.flip_qubit else -0.01)
                popt, pcov = curve_fit(self.fit_gaussian,
                                       self.results[name].ax_data[0],
                                       self.results[name].data.real,p0=p0)
                optimal_amps[i] = popt[0]
            except: 
                optimal_amps[i] = None
        self.results['optimal_amp'] = Result()
        self.results['optimal_amp'].data = optimal_amps
        self.results['optimal_amp'].ax_data = [self.results['default'].ax_data[1]]
        self.results['optimal_amp'].labels = [self.results['default'].labels[1]]



    def plot(self, fig, data):
        times = np.linspace(*self.tau_range_ns)
        amps = np.linspace(*self.amp_range)
        ax = fig.add_subplot(111)
        ax.set_xlabel('Delay (ns)')
        ax.set_ylabel('Amplitude')
        ax.pcolormesh(times, amps, np.transpose(self.results['postselected'].data))
        # plot predictions of optimal amplitude
#        ax.plot(times, self.results['optimal_amp'].data, marker='o', label='exp optimal', color='black')
        ax.plot(times, self.alpha_const_chi, label='const chi', color='black')
#        ax.plot(times, self.alpha_from_cal, label='from chi cal')
        ax.set_ylim(amps.min(),amps.max())
        ax.legend()
