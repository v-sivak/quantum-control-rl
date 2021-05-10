# -*- coding: utf-8 -*-
"""
Created on Thu May  6 13:39:28 2021

@author: qulab
"""


from init_script import *
from scipy.optimize import curve_fit
import numpy as np

class Wigner_with_cooling(FPGAExperiment):
    alpha = FloatParameter(2.0)
    disp_range = RangeParameter((-4.0, 4.0, 101))
    flip_qubit = BoolParameter(True)

    readout_detune_MHz = FloatParameter(58)
    qubit_detune_MHz = FloatParameter(25)
    readout_amp = FloatParameter(0.55)
    qubit_amp = FloatParameter(0.6)
    qubit_ramp_t = IntParameter(200)
    readout_ramp_t = IntParameter(50)
    duration = IntParameter(1000)

    def sequence(self):

        self.qubit_detuned = qubit_ef
        self.qubit = qubit

        self.readout_detuned = readout_aux
        self.readout = readout

        sync()
        self.readout_detuned.set_detune(self.readout_detune_MHz*1e6)
        self.qubit_detuned.set_detune(self.qubit_detune_MHz*1e6)
        sync()

        readout_pump_time = self.duration+2*self.qubit_ramp_t-2*self.readout_ramp_t

        def cool_Murch():
            sync()
            self.readout_detuned.smoothed_constant_pulse(
                    readout_pump_time, amp=self.readout_amp, sigma_t=self.readout_ramp_t)
            self.qubit_detuned.smoothed_constant_pulse(
                    self.duration, amp=self.qubit_amp, sigma_t=self.qubit_ramp_t)
            sync()

        sync()
        self.readout_detuned.set_detune(self.readout_detune_MHz*1e6)
        self.qubit_detuned.set_detune(self.qubit_detune_MHz*1e6)
        self.qubit.set_detune(0)
        sync()

        with system.wigner_tomography(*self.disp_range):
            #delay(2000)


            sync()
            if self.flip_qubit:
                self.qubit.flip()
            else:
                delay(24)
            sync()

            cavity.displace(self.alpha)
#            cool_Murch()
            delay(self.duration + 2 * self.qubit_ramp_t)


    def plot(self, fig, data):
        # create phase space grid
        points = self.disp_range[-1]
        axis = np.linspace(*self.disp_range)
        x, y = np.meshgrid(axis, axis)

        # fit the blob to 2D Gaussian
        sigmaz = 1 - 2*self.results['default'].threshold()
        mean_sigmaz = sigmaz.mean(axis=0).data

        data = mean_sigmaz.ravel()

        ind = np.unravel_index(np.argmax(mean_sigmaz, axis=None), mean_sigmaz.shape)
        initial_guess = (0.7, axis[ind[1]], axis[ind[0]], 0.5, 0.5, 0, 0)
        popt1, pcov1 = curve_fit(gaussian2D, (x, y), data, p0=initial_guess)

        ind = np.unravel_index(np.argmin(mean_sigmaz, axis=None), mean_sigmaz.shape)
        initial_guess = (-0.7, axis[ind[1]], axis[ind[0]], 0.5, 0.5, 0, 0)
        popt2, pcov2 = curve_fit(gaussian2D, (x, y), data, p0=initial_guess)

        if np.sum(np.abs(pcov1)) > np.sum(np.abs(pcov2)):
            popt = popt2
        else:
            popt = popt1

        data_fitted = gaussian2D((x, y), *popt)
        self.popt = popt

        # plot data and fit
        ax = fig.add_subplot(111)
        im = ax.imshow(np.transpose(data.reshape(points, points)), origin='bottom',
                  extent=(x.min(), x.max(), y.min(), y.max()))
        ax.contour(x, y, np.transpose(data_fitted.reshape(points, points)), 2, colors='k')
        ax.set_title('(x,y)=(%.2f, %.2f)' %(popt[1], popt[2]), fontsize = 15)
        fig.colorbar(im)


def gaussian2D((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()
