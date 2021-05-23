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

def get_calibrated_pulse(pulse, zero_pad=False):
    """ 
    Get calibrated pulse with correct dac amplitude, detuning etc.
    For qubit this is a pi-pulse, for cavity this is a unit displacement.
    If the pulse is shorter than 24 ns, fpga_lib will pad it to 24, and
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

class ConditionalDisplacementCompiler():
    """
    This uses the definition CD(beta) = D(sigma_z*beta/2).
    """
    def __init__(self, qubit_pulse_shift=0, qubit_pulse_pad=0,
                 pad_clock_cycle=True):
        """
        Args:
            qubit_pulse_shift (int): by how much to advance or delay the qubit 
                pulse relative to the cavity pulse (in nanoseconds).
            qubit_pulse_pad (int): by how much to pad the qubit pulse on each
                side with zeros (in nanoseconds).
            pad_clock_cycle (bool): flag to pad the gate to be multiple of 4 ns
        """
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

    def CD_params_fixed_tau_from_cal(self, beta, tau_ns, cal_dir):
        """
        Find parameters for CD gate based on simple calibration.
        Calibration gives a linear fit "alpha = a * beta + b" for a fixed tau,
        this can be obtained from 'CD_fixed_time_amp_cal' experiment.
        The displacements can be complex-valued here.
        Angles are taken in the counter-clockwise direction, so normally
        phi_e will be positive and phi_g will be negative.
        """
        data = np.load(os.path.join(cal_dir, 'linear_fit.npz'))
        a, b = data['a'], data['b']
        assert tau_ns == data['tau_ns']
        alpha_abs = a * np.abs(beta) + b
        alpha_abs_max = 0.95/cavity.displace.unit_amp
        alpha_abs = np.min([alpha_abs, alpha_abs_max])
        alpha = alpha_abs * np.exp(1j*(np.pi/2 + np.angle(beta)))
        phi_g = np.zeros_like(alpha)
        phi_e = np.zeros_like(alpha)
        return (tau_ns, alpha, phi_g, phi_e)
    
    def make_pulse(self, tau, alpha, phi_g, phi_e):
        """ Build cavity and qubit sequences for CD gate."""
        # calculate parameters for the pulse        
        phi_diff = phi_e - phi_g
        phi_sum = phi_e + phi_g

        r1 = np.cos(phi_diff/2.)
        r2 = np.cos(phi_diff)
        phase1 = np.pi + phi_sum/2.
        phase2 = phi_sum
        
        Q_complex = get_calibrated_pulse(qubit.pulse)
        Q_complex = np.concatenate([np.zeros(self.qubit_pulse_pad), 
                                    Q_complex, np.zeros(self.qubit_pulse_pad)])
        D_complex = get_calibrated_pulse(cavity.displace)

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
        

class ECD_control_simple_compiler():
    
    def __init__(self, CD_compiler_kwargs={}, CD_params_func_kwargs={}):
        """
        Args:
            CD_compiler_kwargs (dict): to be used on initialization of the
                'ConditionalDisplacementCompiler' instance.
            CD_params_func_kwargs (dict): should contain 'name' of the method
                used for CD_params_func; the rest of the keys will be passed
                to that method as arguments. 
        """
        CD_compiler_kwargs['pad_clock_cycle'] = False
        self.CD = ConditionalDisplacementCompiler(**CD_compiler_kwargs)
        self.pi_pulse = get_calibrated_pulse(qubit.pulse)
        
        func = getattr(self.CD, CD_params_func_kwargs.pop('name'))
        self.CD_params_func = lambda beta: func(beta, **CD_params_func_kwargs)

    def make_pulse(self, beta, phi):
        """
        Args:
            beta (array([T,2], flaot32))
            phi  (array([T,2], float32))
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
            if t < T-1:
                (tau, alpha, phi_g, phi_e) = self.CD_params_func(beta_t)
                cav_CD, qb_CD = self.CD.make_pulse(tau, alpha, phi_g, phi_e)
                C_pulse = np.concatenate([C_pulse, cav_CD[0] + 1j*cav_CD[1]])
                Q_pulse = np.concatenate([Q_pulse, qb_CD[0] + 1j*qb_CD[1]])
            elif t == T-1:
                # Implement last CD as simple displacement
                D = get_calibrated_pulse(cavity.displace)
                C_pulse = np.concatenate([C_pulse, D * beta_t/2.0])
                Q_pulse = np.concatenate([Q_pulse, np.zeros_like(D)])
        
        # make sure pulse length is multiple of 4 ns
        zero_pad = np.zeros(4 - (len(C_pulse) % 4))
        C_pulse = np.concatenate([C_pulse, zero_pad])
        Q_pulse = np.concatenate([Q_pulse, zero_pad])
            
        return C_pulse, Q_pulse



class SBS_simple_compiler():
    
    def __init__(self, CD_compiler_kwargs={}, 
                 s_CD_params_func_kwargs={}, b_CD_params_func_kwargs={}):
        """
        Args:
            CD_compiler_kwargs (dict): to be used on initialization of the
                'ConditionalDisplacementCompiler' instance.
            CD_params_func_kwargs (dict): should contain 'name' of the method
                used for CD_params_func; the rest of the keys will be passed
                to that method as arguments. 
        """
        self.pi_pulse = get_calibrated_pulse(qubit.pulse)
        
        # compiler for small conditional isplacements
        CD_compiler_kwargs['pad_clock_cycle'] = False
        self.CD = ConditionalDisplacementCompiler(**CD_compiler_kwargs)
        
        func = getattr(self.CD, s_CD_params_func_kwargs.pop('name'))
        self.s_CD_params_func = lambda beta: func(beta, **s_CD_params_func_kwargs)

        func = getattr(self.CD, b_CD_params_func_kwargs.pop('name'))
        self.b_CD_params_func = lambda beta: func(beta, **b_CD_params_func_kwargs)
    
    def make_pulse(self, eps1, eps2, beta):
        phase = 0.0 # Keep track of the global oscillator rotation (anti-clockwise)
        
        # 1) Start in |g>, put the qubit in |+> with Y90
        X90 = get_calibrated_pulse(qubit.pi2_pulse)
        Q_pulse = X90 * np.exp(1j*np.pi/2.0)
        C_pulse = np.zeros_like(X90)

        # 2) apply 1st "small" CD gate
        (tau, alpha, phi_g, phi_e) = self.s_CD_params_func(eps1)
        cav_CD, qb_CD = self.CD.make_pulse(tau, alpha, phi_g, phi_e)
        C_pulse = np.concatenate([C_pulse, cav_CD[0] + 1j*cav_CD[1]])
        Q_pulse = np.concatenate([Q_pulse, qb_CD[0] + 1j*qb_CD[1]])
        phase += phi_g + phi_e
        
        # 3) qubit X180 + X90 rotation (combined in single rotation)
        Q_pulse = np.concatenate([Q_pulse, X90*np.exp(1j*np.pi)])
        C_pulse = np.concatenate([C_pulse, np.zeros_like(X90)])
        
        # 4) apply "big" CD gate (with stabilizer amplitude)
        (tau, alpha, phi_g, phi_e) = self.b_CD_params_func(beta)
        cav_CD, qb_CD = self.CD.make_pulse(tau, alpha, phi_g, phi_e)
        C_pulse = np.concatenate([C_pulse, cav_CD[0] + 1j*cav_CD[1]])
        Q_pulse = np.concatenate([Q_pulse, qb_CD[0] + 1j*qb_CD[1]])
        phase += phi_g + phi_e

        # 5) qubit X180 - X90 rotation (combined in single rotation)
        Q_pulse = np.concatenate([Q_pulse, X90])
        C_pulse = np.concatenate([C_pulse, np.zeros_like(X90)])

        # 6) apply 2nd "small" CD gate
        (tau, alpha, phi_g, phi_e) = self.s_CD_params_func(eps2)
        cav_CD, qb_CD = self.CD.make_pulse(tau, alpha, phi_g, phi_e)
        C_pulse = np.concatenate([C_pulse, cav_CD[0] + 1j*cav_CD[1]])
        Q_pulse = np.concatenate([Q_pulse, qb_CD[0] + 1j*qb_CD[1]])
        phase += phi_g + phi_e

        # 7) Qubit Y180 - Y90 rotation (combined in single rotation)
        # this rotates the qubit to be preferentially in |g>
        Q_pulse = np.concatenate([Q_pulse, X90 * np.exp(1j*np.pi/2.0)])
        C_pulse = np.concatenate([C_pulse, np.zeros_like(X90)])

        zero_pad = np.zeros(4 - (len(C_pulse) % 4))
        C_pulse = np.concatenate([C_pulse, zero_pad])
        Q_pulse = np.concatenate([Q_pulse, zero_pad])
        
        return (C_pulse.real, C_pulse.imag), (Q_pulse.real, Q_pulse.imag)