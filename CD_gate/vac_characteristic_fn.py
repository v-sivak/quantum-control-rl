# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 14:19:22 2020

@author: qulab
"""
from init_script import *
from scipy.optimize import curve_fit

def Cs_vac(x, a, b, c):
    """
    Fit function for characteristic function of vacuum.
    Here x/b is the translation amplitude.
    """
    return a * np.exp(-(x/b)**2 / 4) + c

class vac_characteristic_fn(FPGAExperiment):
    """
    Here the qubit sigma_z measurement is such that:
    <sigma_z> = <Re D[beta]> * cos(phi) + <Im D[beta]> * sin(phi)
    """
    phi = FloatParameter(0)
    alpha_range = RangeParameter((-20,20,101))
    free_time = FloatParameter(100)
    loop_delay = FloatParameter(4e6)
    chi = 184e3

    def sequence(self):

        angle = 2*pi * self.chi * self.free_time*1e-9 / 2.0

        with cavity.displace.scan_amplitude(*self.alpha_range):
            readout(init_state='se')
            sync()
            qubit.rotate_y(pi/2)
            sync()
            self.simple_CD(alpha='dynamic', tau=self.free_time, angle=angle)
            sync()
            qubit.rotate_y(pi/2-self.phi)
            readout()
            delay(self.loop_delay)

    def simple_CD(self, alpha, tau, angle):
        """
        Simple conditional displacement with adjustment for rotation.

        Args:
            alpha: large cavity displacement amplitude
            tau: wait time in the displaced state (ns)
            angle: correction angle for displacements (should be 2*pi*chi*tau)

        """
        cavity.displace(amp=alpha)
        sync()
        delay(tau)
        sync()
        cavity.displace(amp=alpha, static_amp=cos(angle), phase=pi)
        sync()
        qubit.flip()
        sync()
        cavity.displace(amp=alpha, static_amp=cos(angle), phase=pi)
        sync()
        delay(tau)
        sync()
        cavity.displace(amp=alpha, static_amp=cos(2*angle))

    def process_data(self):
        init_state = self.results['init_state'].threshold()
        self.results['postselected'] = self.results['default'].postselect(
                init_state, [0])[0].thresh_mean().data
        self.results['postselected'].ax_data = self.results['init_state'].ax_data[1:]
        self.results['postselected'].labels = self.results['init_state'].labels[1:]

        alpha = self.results['default'].ax_data[1]
        sigma_z = self.results['postselected'].data * 2 - 1
        popt, pcov = curve_fit(Cs_vac, alpha, sigma_z)
        a, b, c = popt

        self.fit_params['a'] = a
        self.fit_params['b'] = b
        self.fit_params['c'] = c

        self.results['<sigma_z>'] = sigma_z
        self.results['<sigma_z>'].ax_data = [alpha / b]
        self.results['<sigma_z>'].labels = ['beta']

    def plot(self, fig, data):

        beta = self.results['<sigma_z>'].ax_data[0]
        sigma_z = self.results['<sigma_z>'].data

        # Plot mean e trajectory after postselection
        ax = fig.add_subplot(111)
        ax.set_title('Characteristic function of vacuum')
        ax.set_xlabel(r'Translation amplitude ($\beta$)')
        ax.set_ylabel(r'Re $C_s(\beta)$')
        ax.plot(beta, sigma_z, linestyle='none', marker='o')
        ax.plot(beta, Cs_vac(beta, self.fit_params['a'], 1, self.fit_params['c']),
                linestyle='-')
