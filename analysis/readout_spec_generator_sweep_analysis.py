# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 17:49:33 2020

@author: qulab
"""

import h5py
import numpy as np
import analysis_functions as af
import matplotlib.pyplot as plt


filename = r'D:\DATA\exp\gkp_exp.fpga_exp.readout_spec_generator_sweep\archive\20210430.h5'
grp = '2'

# create figure
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.set_xlabel('Re(a)')
ax1.set_ylabel('Im(a)')
ax2 = fig.add_subplot(222)
ax2.set_xlabel('frequency (GHz)')
ax2.set_ylabel('|a|')
ax3 = fig.add_subplot(223)
ax3.set_xlabel('frequency (GHz)')
ax3.set_ylabel('angle(a)')
ax4 = fig.add_subplot(224)
ax4.set_xlabel('frequency (GHz)')
ax4.set_ylabel('20 log|a|')

f0, kc, ki = {}, {}, {}

for s in ['g', 'e']:
    # load data
    f = h5py.File(filename, 'r')
    im = np.array(f[grp+'/im_'+s+'/data']).mean(axis=1)
    re = np.array(f[grp+'/re_'+s+'/data']).mean(axis=1)
    freq = np.array(f[grp+'/freq/data'])
    f.close()
    
    # fit data
    data = re - 1j*im
    popt = af.fit_complex_a_out(freq, data, f_0=9.170e9, kc=0.5e6, ki=0.1e6, T=-60e-9)
    f0[s], kc[s], ki[s], _, _ = popt
    
    # plot data
    ax1.plot(np.real(data), np.imag(data))
    ax2.plot(freq*1e-9, np.abs(data))
    ax3.plot(freq*1e-9, np.angle(data, deg=True))
    ax4.plot(freq*1e-9, 20 * np.log(np.abs(data)) / np.log(10))
    
    # plot fits
    data_fit = af.complex_a_out(freq, *popt)
    ax1.plot(np.real(data_fit),np.imag(data_fit),'--', color='black')
    ax2.plot(freq*1e-9,np.abs(data_fit),'--', color='black')
    ax3.plot(freq*1e-9,np.angle(data_fit, deg=True),'--', color='black')
    ax4.plot(freq*1e-9, 20 * np.log(np.abs(data_fit)) / np.log(10), '--', color='black')

# print fit results and show figure
for s in ['g', 'e']:
    print('='*5 + ' ' + s + ' ' + '='*5)
    print('f0 = %f GHz' %(f0[s]*1e-9))
    print('kc = %f MHz' %(kc[s]*1e-6))
    print('ki = %f kHz' %(ki[s]*1e-3))
chi = f0['g']-f0['e']
print('='*10)
print('chi = %f MHz' % (chi*1e-6))
plt.show()

