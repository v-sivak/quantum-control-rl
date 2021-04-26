# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 22:53:11 2021

@author: qulab
"""
import os
import numpy as np
import math
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import CubicSpline
from init_script import cavity, qubit

class ConditionalDisplacementCompiler():
    """
    This uses the definition CD(beta) = D(sigma_z*beta/2).
    """
    def __init__(self, cal_dir=None, qubit_pulse_shift=0, qubit_pulse_pad=0,
                 pad_clock_cycle=True):
        """
        Args:
            cal_dir (str): directory with cavity rotation frequency calibrations. 
                Should contain .npy files named 'nbar', 'freq_e' and 'freq_g'.
            qubit_pulse_shift (int): by how much to advance or delay the qubit 
                pulse relative to the cavity pulse (in nanoseconds).
            qubit_pulse_pad (int): by how much to pad the qubit pulse on each
                side with zeros (in nanoseconds).
            pad_clock_cycle (bool): flag to pad the gate to be multiple of 4 ns
        """
        self.cal_dir = cal_dir
        self.qubit_pulse_shift = qubit_pulse_shift
        self.qubit_pulse_pad = qubit_pulse_pad
        self.pad_clock_cycle = pad_clock_cycle
    
    def CD_params_fixed_tau(self, beta, tau_ns):
        """
        Find parameters for CD gate based on simple constant chi model.
        The displacements can be complex-valued here.
        Angles are taken in the counter-clockwise direction, so normally
        phi_e will be positive and phi_g will be negative.
        """
        alpha = beta / 2. / np.sin(2*np.pi*cavity.chi*tau_ns*1e-9) * 1j
        if np.abs(alpha)*cavity.displace.unit_amp > 0.95:
            alpha *= (0.95/cavity.displace.unit_amp) / np.abs(alpha)
        phi_g = np.zeros_like(alpha)
        phi_e = np.zeros_like(alpha)
        return (tau_ns, alpha, phi_g, phi_e)

    def CD_params_fixed_alpha(self, beta, alpha_abs):
        tau = 1. / (2*np.pi*cavity.chi) * np.arcsin(np.abs(beta/alpha_abs/2.))
        tau_ns = int(math.ceil(tau*1e9))
        alpha_abs = np.abs(beta) / 2. / np.sin(2*np.pi*cavity.chi*tau_ns*1e-9)
        alpha_phase = np.angle(beta) + np.pi/2
        alpha = alpha_abs * np.exp(1j*alpha_phase)
        phi_g = np.zeros_like(alpha)
        phi_e = np.zeros_like(alpha)
        return (tau_ns, alpha, phi_g, phi_e)

    def CD_params_fixed_tau_from_cal(self, beta, tau_ns):
        """
        Find parameters for CD gate based on simple calibration.
        Calibration gives a linear fit "alpha = a * beta + b" for a fixed tau,
        this can be obtained from 'CD_fixed_time_amp_cal' experiment.
        The displacements can be complex-valued here.
        Angles are taken in the counter-clockwise direction, so normally
        phi_e will be positive and phi_g will be negative.
        """
        data = np.load(os.path.join(self.cal_dir, 'linear_fit.npz'))
        a, b = data['a'], data['b']
        assert tau_ns == data['tau_ns']
        alpha_abs = a * np.abs(beta) + b
        alpha_abs_max = 0.95/cavity.displace.unit_amp
        alpha_abs = np.min([alpha_abs, alpha_abs_max])
        alpha = alpha_abs * np.exp(1j*(np.pi/2 + np.angle(beta)))
        phi_g = np.zeros_like(alpha)
        phi_e = np.zeros_like(alpha)
        return (tau_ns, alpha, phi_g, phi_e)

    
    def get_calibrated_pulse(self, pulse, zero_pad=False):
        """ 
        Get calibrated pulse with correct dac amplitude, detuning etc.
        For qubit this is a pi-pulse, for cavity this is a unit displacement.
        If the pulse is shorter than 24 ns, fpga_lib would pad it to 24, and
        this function strips that padding.
        """
        i, q = pulse.make_wave(zero_pad=zero_pad)
        f = pulse.detune
        t_offset = (24 - len(i)) / 2 if len(i)<24 else 0
        t = (np.arange(len(i)) + t_offset)*1e-9
        i_prime = np.cos(2*np.pi*f*t)*i + np.sin(2*np.pi*f*t)*q
        q_prime = np.cos(2*np.pi*f*t)*q - np.sin(2*np.pi*f*t)*i
        A_complex = i_prime + 1j * q_prime
        A_complex *= pulse.unit_amp
        return A_complex
    
    def make_pulse(self, tau, alpha, phi_g, phi_e):
        """ Build cavity and qubit sequences for CD gate."""
        # calculate parameters for the pulse        
        phi_diff = phi_e - phi_g
        phi_sum = phi_e + phi_g

        r1 = np.cos(phi_diff/2.)
        r2 = np.cos(phi_diff)
        phase1 = np.pi + phi_sum/2.
        phase2 = phi_sum
        
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

        # make sure pulse length is multiple of 4 ns
        if self.pad_clock_cycle:
            zero_pad = np.zeros(4 - (len(C_pulse) % 4))
            C_pulse = np.concatenate([C_pulse, zero_pad])
            Q_pulse = np.concatenate([Q_pulse, zero_pad])

        return (C_pulse.real, C_pulse.imag), (Q_pulse.real, Q_pulse.imag)
        

# TODO: make sure all conventions for direction of rotation are consistent
class ECD_control_simple_compiler():
    
    def __init__(self, tau_ns=None, alpha_abs=None):
        self.CD = ConditionalDisplacementCompiler(pad_clock_cycle=False)
        self.pi_pulse = self.CD.get_calibrated_pulse(qubit.pulse)
        
        if tau_ns is not None:
            self.CD_params_func = lambda beta: self.CD.CD_params_fixed_tau(beta, tau_ns)
        elif alpha_abs is not None:
            self.CD_params_func = lambda beta: self.CD.CD_params_fixed_alpha(beta, alpha_abs)

    def make_pulse(self, beta, phi):
        """
        Args:
            beta (array([T,2], flaot32)):
            phi (array([T2, float32]))
        """
        T = beta.shape[0] # protocol duration (number of steps)
        C_pulse, Q_pulse = np.array([]), np.array([])
        
        for t in range(T):
            # First create the qubit rotation gate
            phase_t, angle_t = phi[t,0], phi[t,1]
            qb_rotation = self.pi_pulse * angle_t / np.pi * np.exp(1j*phase_t)
            C_pulse = np.concatenate([C_pulse, np.zeros_like(qb_rotation)])
            Q_pulse = np.concatenate([Q_pulse, qb_rotation])
            
            # Then create the CD gate
            beta_t = beta[t,0] + 1j*beta[t,1]
            (tau, alpha, phi_g, phi_e) = self.CD_params_func(beta_t)
            cav_CD, qb_CD = self.CD.make_pulse(tau, alpha, phi_g, phi_e)
            C_pulse = np.concatenate([C_pulse, cav_CD[0] + 1j*cav_CD[1]])
            Q_pulse = np.concatenate([Q_pulse, qb_CD[0] + 1j*qb_CD[1]])
        
        # This is to make sure we end up in 'g'
        # TODO: to avoid this, you can implement last CD as a simple displacement
        C_pulse = np.concatenate([C_pulse, np.zeros_like(self.pi_pulse)])
        Q_pulse = np.concatenate([Q_pulse, self.pi_pulse])
        
        # make sure pulse length is multiple of 4 ns
        zero_pad = np.zeros(4 - (len(C_pulse) % 4))
        C_pulse = np.concatenate([C_pulse, zero_pad])
        Q_pulse = np.concatenate([Q_pulse, zero_pad])
            
        return C_pulse, Q_pulse


# TODO: this needs to be re-written after updating the CD compiler functions.
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