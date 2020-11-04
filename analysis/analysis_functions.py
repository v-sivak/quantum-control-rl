# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 16:32:11 2020

@author: qulab
"""
import numpy as np
from scipy.optimize import leastsq


def complex_a_out(f, f_0, kc, ki, a_in, T):
    """
    Complex amplitude of the reflected wave.
    
    Args:
        f (float): frequency of the signal (Hz)
        f_0 (float): resonance frequency (Hz)
        kc (float): coupling rate (Hz)
        ki (float): interal loss rate (Hz)
        a_in (float): complex amplitude of the incoming wave (a.u.)
        T (float): electrical delay (seconds)
    """
    D = f - f_0
    num = - 1j*D + (kc - ki)/2
    den = 1j*D + (kc+ki)/2
    if kc>0 and ki>0 and f_0>0:
        return num / den * a_in * np.exp(1j*D*T)
    else:
        return np.Inf

def fit_complex_a_out(f, a_out, f_0=7e9, kc=0.5e6, ki=0.5e6, a_in=None, T=70e-9,
                      multistart=10):
    """
    Fit complex reflection coefficient over a range of frequencies to extract
    parameters of the resonance.
    
    Args:
        f (float): array of probe frequencies (Hz)
        a_out (complex): array of complex amplitudes of the reflected wave
        f_0 (float): guess resonance frequency (Hz)
        kc (float): guess coupling rate (Hz)
        ki (float): guess interal loss rate (Hz)
        a_in (complex): array of complex amplitudes of the incoming wave (a.u.)
        T (s): electrical delay (seconds)
        multistart (int): number of guess 'f_0' values uniformly selected from
            the array of 'f' to restart the optimizer. If 1, will only use the
            provided guess 'f_0'. 

    """
    def complex_a_out_wrapper(f, f_0, kc, ki, re_a_in, im_a_in, T): 
        """ Wrapper to make sure all arguments are real. Everything after 
        'f' will be fitted. """
        return complex_a_out(f, f_0, kc, ki, re_a_in + 1j*im_a_in, T)
    
    def residuals(params, x, y):
        """ Residuals for least squares optimizer. """
        diff = complex_a_out_wrapper(x, *params) - y
        flatDiff = np.zeros(diff.size * 2, dtype=np.float64)
        flatDiff[0:flatDiff.size:2] = diff.real
        flatDiff[1:flatDiff.size:2] = diff.imag
        return flatDiff

    if a_in is None: a_in = a_out[0]
    p_guess = [f_0, kc, ki, np.real(a_in), np.imag(a_in), T]

    if multistart == 1:
        # use the provided guess f_0 for resonance frequency
        popt, pcov = leastsq(residuals, p_guess, args=(f, a_out))
    else:
        # iterate guess f_0 in the range of measured frequencies
        perr_best = np.Inf
        for i in range(multistart):
            p_guess[0] = f[i*len(f)/multistart] # new f_0 value
            popt_new, pcov, infodict, mesg, ier = leastsq(
                    residuals, p_guess, args=(f, a_out), full_output=True)
            perr = np.sqrt(np.diag(pcov))[0]
            if perr < perr_best:
                perr_best = perr
                popt = popt_new
    
    (f_0, kc, ki, re_a_in, im_a_in, T) = popt
    return (f_0, kc, ki, re_a_in+1j*im_a_in, T)