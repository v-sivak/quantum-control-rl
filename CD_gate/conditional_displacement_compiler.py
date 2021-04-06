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
    
    def __init__(self, cal_dir=None, qubit_pulse_shift=0, qubit_pulse_pad=0,
                 pad_clock_cycle=True):
        self.cal_dir = cal_dir
        self.qubit_pulse_shift = qubit_pulse_shift
        self.qubit_pulse_pad = qubit_pulse_pad
        self.pad_clock_cycle = pad_clock_cycle
    
    def CD_params(self, beta, tau_ns):
        """Find parameters for CD gate based on simple constant chi model."""
        alpha = beta / 2. / np.sin(2*np.pi*cavity.chi*tau_ns*1e-9)
        phi_g = np.zeros_like(alpha)
        phi_e = np.zeros_like(alpha)
        return (alpha, phi_g, phi_e)

    def CD_params_improved(self, beta, tau, interpolation='quartic_fit'):
        """Find parameters for CD gate based on calibration of cavity rotation
        frequency vs nbar."""
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
        """ Get calibrated pulse with correct dac amp, detuning etc."""
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
        """ Build cavity and qubit sequences for CD gate."""
        # calculate parameters for the pulse        
        phi_diff = phi_g - phi_e
        phi_sum = phi_g + phi_e

        r1 = np.cos(phi_diff/2.)
        r2 = np.cos(phi_diff)
        phase1 = np.pi - phi_sum/2.
        phase2 = -phi_sum
        
        Q_complex = self.get_calibrated_pulse(qubit.pulse)
        Q_complex = np.concatenate([np.zeros(self.qubit_pulse_pad), 
                                    Q_complex, np.zeros(self.qubit_pulse_pad)])
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
        
        # shift the qubit pulse to compensate electrical delay
        Q_pulse = np.roll(Q_pulse, self.qubit_pulse_shift)

        # make sure instruction length is multiple of 4 ns
        if self.pad_clock_cycle:
            zero_pad = np.zeros(4 - (len(C_pulse) % 4))
            C_pulse = np.concatenate([C_pulse, zero_pad])
            Q_pulse = np.concatenate([Q_pulse, zero_pad])

        return (C_pulse.real, C_pulse.imag), (Q_pulse.real, Q_pulse.imag)
    
    
class sBs_compiler():
    
    def __init__(self, tau_small, tau_big, cal_dir=None):
        self.C = ConditionalDisplacementCompiler(cal_dir=cal_dir, 
                                                 pad_clock_cycle=False)
        self.tau_small = tau_small
        self.tau_big = tau_big
    
    def make_pulse(self, eps1, eps2, beta):
        phase = 0.0 # Keep track of the global oscillator rotation
        CD_params_func = self.C.CD_params # will predict (alpha, phi_g, phi_e)
        
        # 1) Start in |g>, put the qubit in |+>
        X90 = self.C.get_calibrated_pulse(qubit.pi2_pulse)
        sbs_qb = X90 * np.exp(1j*np.pi/2.0)
        sbs_cav = np.zeros_like(X90)

        # 2a) apply 1st "small" CD gate
        (alpha, phi_g, phi_e) = CD_params_func(eps1, self.tau_small)
        cav_CD, qb_CD = self.C.make_pulse(self.tau_small, alpha, phi_g, phi_e)
        qb_pulse_complex = qb_CD[0] + 1j * qb_CD[1]
        cav_pulse_complex = (cav_CD[0] + 1j * cav_CD[1]) * np.exp(1j*phase)
        phase += phi_g + phi_e
        sbs_qb = np.concatenate([sbs_qb, qb_pulse_complex])
        sbs_cav = np.concatenate([sbs_cav, cav_pulse_complex])
        
        # 3) qubit X180 + X90 rotation (combined in single rotation)
        sbs_qb = np.concatenate([sbs_qb, X90*np.exp(1j*np.pi)])
        sbs_cav = np.concatenate([sbs_cav, np.zeros_like(X90)])
        
        # 4a) apply "big" CD gate (with stabilizer amplitude)
        (alpha, phi_g, phi_e) = CD_params_func(beta, self.tau_big)
        alpha = complex(alpha) * np.exp(-1j*np.pi/2.0)
        cav_CD, qb_CD = self.C.make_pulse(self.tau_big, alpha, phi_g, phi_e)
        qb_pulse_complex = qb_CD[0] + 1j * qb_CD[1]
        cav_pulse_complex = (cav_CD[0] + 1j * cav_CD[1]) * np.exp(1j*phase)
        phase += phi_g + phi_e
        sbs_qb = np.concatenate([sbs_qb, qb_pulse_complex])
        sbs_cav = np.concatenate([sbs_cav, cav_pulse_complex])

        # 5) qubit X180 - X90 rotation (combined in single rotation)
        sbs_qb = np.concatenate([sbs_qb, X90])
        sbs_cav = np.concatenate([sbs_cav, np.zeros_like(X90)])

        # 6a) apply 2nd "small" CD gate
        (alpha, phi_g, phi_e) = CD_params_func(eps2, self.tau_small)
        cav_CD, qb_CD = self.C.make_pulse(self.tau_small, alpha, phi_g, phi_e)
        qb_pulse_complex = qb_CD[0] + 1j * qb_CD[1]
        cav_pulse_complex = (cav_CD[0] + 1j * cav_CD[1]) * np.exp(1j*phase)
        phase += phi_g + phi_e
        sbs_qb = np.concatenate([sbs_qb, qb_pulse_complex])
        sbs_cav = np.concatenate([sbs_cav, cav_pulse_complex])

        # 7) Qubit Y180 - Y90 rotation (combined in single rotation)
        # this rotates the qubit to be preferentially in |g>
        sbs_qb = np.concatenate([sbs_qb, X90 * np.exp(1j*np.pi/2.0)])
        sbs_cav = np.concatenate([sbs_cav, np.zeros_like(X90)])

        zero_pad = np.zeros(4 - (len(sbs_cav) % 4))
        sbs_cav = np.concatenate([sbs_cav, zero_pad])
        sbs_qb = np.concatenate([sbs_qb, zero_pad])
        
        return (sbs_cav.real, sbs_cav.imag), (sbs_qb.real, sbs_qb.imag)