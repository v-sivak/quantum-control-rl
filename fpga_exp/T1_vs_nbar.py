# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:43:02 2020

@author: Vladimir Sivak
"""


from init_script import *
from scipy.optimize import curve_fit

class T1_vs_nbar(FPGAExperiment):

    amp = RangeParameter((0.0, 1.0, 21))
    length = RangeParameter((0.0, 4e5, 101))
    loop_delay = FloatParameter(1e6)
    channel = (0, 3)
    T1_guess = 100 # guess for the fit in us

    def sequence(self):
        with scan_amplitude(self.channel, *self.amp):
            with scan_length(*self.length) as dyn_len:
                readout(init_state='se')
                qubit.flip()
                sync()
                constant_pulse(self.channel, dyn_len, amp='dynamic')
                delay(1000)
                readout()
                delay(self.loop_delay)

    def process_data(self):
        # postselect on initial measurement
        init_state = self.results['init_state'].threshold()
        self.results['postselected'] = self.results['default'].postselect(
                init_state, [0])[0].thresh_mean().data
        self.results['postselected'].ax_data = self.results['init_state'].ax_data[1:]
        self.results['postselected'].labels = self.results['init_state'].labels[1:]
        
        # fit data to exponential decay
        def exp_decay(x, a, b, c):
            return a*np.exp(-x/b)+c
        
        T1 = np.zeros(self.amp[-1])
        for i in range(self.amp[-1]):
            xdata = self.results['default'].ax_data[2] #time in ns
            ydata = np.mean(self.results['default'].data[:,i,:].real, axis=0)
            popt, pcov = curve_fit(exp_decay, xdata, ydata, 
                                   p0=[1, self.T1_guess*1e3, 0])
            T1[i] = popt[1]*1e-3 # convert to us
        self.results['T1'] = T1
        self.results['T1'].ax_data = [self.results['default'].ax_data[1]]
        self.results['T1'].labels = [self.results['default'].labels[1]]
        