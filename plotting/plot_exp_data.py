# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 10:19:35 2020

@author: qulab
"""
import numpy as np
import matplotlib.pyplot as plt
import plot_config
from scipy.optimize import curve_fit

# load and plot experimental data
dac_cal_coeff = 1
nbar = np.load(r'Z:\tmp\for Vlad\from_vlad\nbar.npy') / dac_cal_coeff
freq_g = np.load(r'Z:\tmp\for Vlad\from_vlad\freq_g.npy')
freq_e = np.load(r'Z:\tmp\for Vlad\from_vlad\freq_e.npy')

fig, ax = plt.subplots(1,1,figsize=(3.375,2), dpi=200)
ax.set_xlabel('nbar')
ax.set_ylabel('Rotation freq (kHz)')
ax.plot(nbar, freq_g*1e-3, marker='.', linestyle='none', label='g')
ax.plot(nbar, freq_e*1e-3, marker='.', linestyle='none', label='e')
ax.legend(loc='lower right')
plt.tight_layout()


# Plot average rotation frequency and fit Kerr and detuning
avg_freq = (freq_g + freq_e)/2.0
def avg_freq_fit_func(n, Kerr, Delta):
    return  Delta + Kerr * n
fit_pts = 21
popt, pcov = curve_fit(avg_freq_fit_func, nbar[:fit_pts], avg_freq[:fit_pts])
Kerr, Delta = popt

fig, ax = plt.subplots(1,1,figsize=(3.375,2.2), dpi=200)
ax.set_xlabel('nbar')
ax.set_ylabel('Avg rotation freq (kHz)')
ax.plot(nbar, avg_freq*1e-3, marker='.',linestyle='none')
ax.plot(nbar[:fit_pts], avg_freq_fit_func(nbar[:fit_pts], *popt)*1e-3,
        label=r'K=%.0f Hz, $\Delta$= %.1f kHz' %(Kerr,Delta*1e-3))
ax.legend()
plt.tight_layout()


# Plot difference rotation frequency and fit chi and chi_prime
diff_freq = freq_g - freq_e
def diff_freq_fit_func(n, chi, chi_prime):
    return  chi + chi_prime * n
popt, pcov = curve_fit(diff_freq_fit_func, nbar[:fit_pts], diff_freq[:fit_pts])
chi, chi_prime = popt

fig, ax = plt.subplots(1,1,figsize=(3.375,2.2), dpi=200)
ax.set_xlabel('nbar')
ax.set_ylabel('Diff rotation freq (kHz)')
ax.plot(nbar, diff_freq*1e-3, marker='.',linestyle='none')
ax.plot(nbar[:fit_pts], diff_freq_fit_func(nbar[:fit_pts], *popt)*1e-3,
        label='chi=%.0f kHz, chi_prime= %.0f Hz' %(chi*1e-3,chi_prime))
ax.legend()
plt.tight_layout()


# Plot quartic fits to 'e' and 'g' rotation frequency
def quartic_fit(n, c0, c1, c2, c3, c4):
    return c0 + c1*n + c2*n**2 + c3*n**3 + c4*n**4
popt_g, _ = curve_fit(quartic_fit, nbar, freq_g)
popt_e, _ = curve_fit(quartic_fit, nbar, freq_e)

fig, ax = plt.subplots(1,1,figsize=(3.375,2), dpi=200)
ax.set_xlabel('nbar')
ax.set_ylabel('Rotation freq (kHz)')
ax.plot(nbar, freq_g*1e-3, marker='.', linestyle='none', label='g')
ax.plot(nbar, freq_e*1e-3, marker='.', linestyle='none', label='e')
ax.plot(nbar, quartic_fit(nbar, *popt_g)*1e-3)
ax.plot(nbar, quartic_fit(nbar, *popt_e)*1e-3)
ax.legend(loc='lower right')
plt.tight_layout()