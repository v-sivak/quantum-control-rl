# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 22:53:11 2021

@author: qulab
"""
import os
import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import CubicSpline
from init_script import cavity, qubit

class ConditionalDisplacementCompiler():
    
    def __init__(self, cal_dir, qubit_pulse_shift=-12):
        self.cal_dir = cal_dir
        self.qubit_pulse_shift = qubit_pulse_shift
    
    def CD_params(self, beta, tau_ns):
        alpha = beta / 2. / np.sin(2*np.pi*cavity.chi*tau_ns*1e-9)
        phi_g = np.zeros_like(alpha)
        phi_e = np.zeros_like(alpha)
        return (alpha, phi_g, phi_e)

    def CD_params_improved(self, beta, tau, interpolation='quartic_fit'):
        # load cavity rotation frequency vs nbar data
        nbar = np.load(os.path.join(self.cal_dir, 'nbar.npy'))
        freq_e_exp = np.load(os.path.join(self.cal_dir, 'freq_e.npy'))
        freq_g_exp = np.load(os.path.join(self.cal_dir, 'freq_g.npy'))

        if interpolation == 'quartic_fit':
            # fit experimental data to 4-th order polynomial in nbar
            def quartic_fit(n, c0, c1, c2, c3, c4):
                return c0 + c1*n + c2*n**2 + c3*n**3 + c4*n**4
            popt_g, _ = curve_fit(quartic_fit, nbar, freq_g_exp)
            popt_e, _ = curve_fit(quartic_fit, nbar, freq_e_exp)
    
            freq_g = lambda a: quartic_fit(a**2, *popt_g)
            freq_e = lambda a: quartic_fit(a**2, *popt_e)
        
        if interpolation == 'cubic_spline':
            freq_g = lambda a: CubicSpline(nbar, freq_g_exp)(a**2)
            freq_e = lambda a: CubicSpline(nbar, freq_e_exp)(a**2)

        # use Nelder-Mead algo to find optimal alpha for each tau
        def cost_fn(a):
            phi = 2*np.pi*(freq_g(a)-freq_e(a))*tau*1e-9
            return (beta - 2*a*np.sin(phi))**2

        def find_alpha(beta, tau):
            alpha_guess = beta / 2. / np.sin(2*np.pi*cavity.chi*tau*1e-9)
            res = minimize(cost_fn, x0=alpha_guess, method='Nelder-Mead')
            return res.x

        alpha = find_alpha(beta, tau)
        phi_g = 2*np.pi * freq_g(alpha) * tau*1e-9
        phi_e = 2*np.pi * freq_e(alpha) * tau*1e-9

        return (alpha, phi_g, phi_e)
    
    def get_calibrated_pulse(self, pulse, zero_pad=False):
        i, q = pulse.make_wave(zero_pad=zero_pad)
        f = pulse.detune
        t_offset = (24 - len(i)) / 2
        t = (np.arange(len(i)) + t_offset)*1e-9
        i_prime = np.cos(2*np.pi*f*t)*i + np.sin(2*np.pi*f*t)*q
        q_prime = np.cos(2*np.pi*f*t)*q - np.sin(2*np.pi*f*t)*i
        A_complex = i_prime + 1j * q_prime
        A_complex *= pulse.unit_amp
        return A_complex
    
    def make_pulse(self, tau, alpha, phi_g, phi_e):
        # calculate parameters for the pulse        
        phi_diff = phi_g - phi_e
        phi_sum = phi_g + phi_e

        r1 = np.cos(phi_diff/2.)
        r2 = np.cos(phi_diff)
        phase1 = np.pi - phi_sum/2.
        phase2 = -phi_sum
        
        Q_complex = self.get_calibrated_pulse(qubit.pulse)
        D_complex = self.get_calibrated_pulse(cavity.displace)

        delay_tau = np.zeros(int(round(tau)))

        #----- first build cavity pulse ------------
        C_pulse = D_complex * alpha
        C_pulse = np.concatenate([C_pulse, delay_tau])
        D1 = D_complex * alpha * r1 * np.exp(1j*phase1)
        C_pulse = np.concatenate([C_pulse, D1])
        C_pulse = np.concatenate([C_pulse, np.zeros(len(Q_complex))])
        C_pulse = np.concatenate([C_pulse, D1])
        C_pulse = np.concatenate([C_pulse, delay_tau])
        D2 = D_complex * alpha * r2 * np.exp(1j*phase2)
        C_pulse = np.concatenate([C_pulse, D2])
        #----------------------------------------

        #----- now build qubit pulse ------------
        flip_delay = (len(C_pulse) - len(Q_complex)) / 2
        Q_pulse = np.zeros(flip_delay)
        Q_pulse = np.concatenate([Q_pulse, Q_complex])
        Q_pulse = np.concatenate([Q_pulse, np.zeros(flip_delay)])
        #----------------------------------------
        
        # make sure instruction length is multiple of 4 ns
        zero_pad = np.zeros(4 - (len(C_pulse) % 4))
        C_pulse = np.concatenate([C_pulse, zero_pad])
        Q_pulse = np.concatenate([Q_pulse, zero_pad])
        
        # shift the qubit pulse to compensate electrical delay
        Q_pulse = np.roll(Q_pulse, self.qubit_pulse_shift)

        return (C_pulse.real, C_pulse.imag), (Q_pulse.real, Q_pulse.imag)