# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 09:32:29 2021

@author: qulab
"""

import numpy as np
from init_script import *
from conditional_displacement_compiler import ConditionalDisplacementCompiler
from scipy.optimize import curve_fit

class CD_fixed_time_amp_cal(FPGAExperiment):
    """
    This experiment sweeps the amplitudes of displacement in CD gate (alpha)
    and of return displacement (beta/2) to bring the blob to origin.

    The CD gate is simple with no amplitude or phase corrections.
    After CD gate the cavity is returned to vacuum with a displacement,
    and selective qubit pulses is used to verify this.

    """
    tau = IntParameter(48)
    loop_delay = FloatParameter(4e6)
    flip_qubit = BoolParameter(False)
    alpha_range = RangeParameter((1,20,11))
    beta_range = RangeParameter((0,3,11))
    reps = IntParameter(1)
    cal_dir = StringParameter(r'C:\code\gkp_exp\CD_gate\storage_rotation_freq_vs_nbar')

    def sequence(self):
        return_phase = np.pi/2.0 if not self.flip_qubit else -np.pi/2.0
        C = ConditionalDisplacementCompiler(cal_dir=self.cal_dir)
        # create CD pulse whose displacement amplitude will be scanned
        cavity_pulse, qubit_pulse = C.make_pulse(self.tau, 1., 0., 0.)
        cavity_pulse = [cavity_pulse[s]/cavity.displace.unit_amp for s in [0,1]]


        self.alpha_const_chi, self.alpha_from_cal = [], []
        for beta in np.linspace(*self.beta_range):
            # predicted optimal displacement amplitudes
            self.alpha_const_chi.append(np.abs(C.CD_params_fixed_tau(beta, self.tau)[1]))
#            self.alpha_from_cal.append(C.CD_params_improved(beta, self.tau)[0])


        with scan_amplitude(cavity.chan, *self.alpha_range,
                            scale=cavity.displace.unit_amp,
                            plot_label='alpha'):
            with scan_amplitude(cavity_1.chan, *self.beta_range,
                                scale=cavity_1.displace.unit_amp / 2.0,
                                plot_label='beta'):
                readout(init_state='se')
                sync()
                if self.flip_qubit:
                    qubit.flip()
                for j in range(self.reps): #with Repeat(self.reps):
                    sync()
                    qubit.array_pulse(*qubit_pulse)
                    cavity.array_pulse(*cavity_pulse, amp='dynamic')
                    sync()
                    cavity_1.displace(amp='dynamic', phase=return_phase)  # TODO: needs to be beta/2
                    sync()
                    qubit.flip()
                sync()
                qubit.flip(selective=True)
                sync()
                readout()
                delay(self.loop_delay)

    def fit_gaussian(self, xs, x0, sig, ofs, amp):
        return ofs + amp * np.exp(-(xs - x0)**2 / (2 * sig**2))

    def process_data(self):
        # postselect on initial msmt
        self.results['postselected'] = Result()
        init_state = self.results['init_state'].threshold()
        self.results['postselected'] = self.results['default'].postselect(
                init_state, [0])[0].thresh_mean().data
        self.results['postselected'].ax_data = self.results['default'].ax_data[1:]
        self.results['postselected'].labels = self.results['default'].labels[1:]

        # create slices for each beta
        betas = np.linspace(*self.beta_range)
        for i, beta in enumerate(betas):
            name = 'beta_slice_' + '%.2f' %beta
            self.results[name] = Result()
            self.results[name].data = self.results['default'].data[:,:,i].mean(axis=0)
            self.results[name].ax_data = [self.results['default'].ax_data[1]]
            self.results[name].labels = [self.results['default'].labels[1]]

        # fit best amplitude for each beta
        # NOTE: fitting to gaussian is not correct, but taking argmax is noisy
        optimal_amps = np.zeros_like(betas)
        for i, beta in enumerate(betas):
            name = 'beta_slice_' + '%.2f' %beta
            try:
                this_data = self.results[name].data.real

                # guess for thie fit
                x0_guess = self.results[name].ax_data[0][np.argmax(this_data)]
                offset_guess = np.min(this_data)
                amp_guess = np.max(this_data)-np.min(this_data)
                p0 = (x0_guess, 5., offset_guess, amp_guess)
                popt, pcov = curve_fit(self.fit_gaussian,
                                       self.results[name].ax_data[0],
                                       self.results[name].data.real, p0=p0)
                optimal_amps[i] = popt[0]
            except:
                optimal_amps[i] = None
        self.results['optimal_amp'] = Result()
        self.results['optimal_amp'].data = optimal_amps
        self.results['optimal_amp'].ax_data = [self.results['default'].ax_data[2]]
        self.results['optimal_amp'].labels = [self.results['default'].labels[2]]



    def plot(self, fig, data):
        betas = np.linspace(*self.beta_range)
        alphas = np.linspace(*self.alpha_range)
        ax = fig.add_subplot(111)
        ax.set_xlabel('beta')
        ax.set_ylabel('alpha')
        ax.pcolormesh(betas, alphas, self.results['postselected'].data)
        # plot predictions of optimal amplitude
        ax.plot(betas, self.results['optimal_amp'].data, marker='o', label='exp optimal', color='black')
        ax.plot(betas, self.alpha_const_chi, label='const chi', color='green')
#        ax.plot(betas, self.alpha_from_cal, label='from chi cal')
        ax.set_ylim(alphas.min(),alphas.max())
        ax.legend(loc='lower right')
