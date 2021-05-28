# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 13:56:43 2021

@author: qulab
"""
from init_script import *
from conditional_displacement_compiler import ConditionalDisplacementCompiler
from scipy.optimize import curve_fit
import numpy as np


class CD_ge_wigner(FPGAExperiment):
    disp_range = RangeParameter((-4.0, 4.0, 101))
    
    flip_qubit = BoolParameter(False)

    # Parameters of the CD
    tau = IntParameter(50)
    beta = FloatParameter(1.0)
    cal_dir = StringParameter('')

    def sequence(self):
        
        C = ConditionalDisplacementCompiler(qubit_pulse_pad=4)
        (tau, alpha, phi_g, phi_e) = C.CD_params_fixed_tau_from_cal(self.beta, self.tau, self.cal_dir)
        cavity_pulse, qubit_pulse = C.make_pulse(tau, alpha, 0., 0.)

        with system.wigner_tomography(*self.disp_range):
            sync()
            if self.flip_qubit:
                qubit.flip()
                sync()
            cavity.array_pulse(*cavity_pulse)
            qubit.array_pulse(*qubit_pulse)
            sync()
            if not self.flip_qubit:
                qubit.flip()
                sync()

    def plot(self, fig, data):
        # create phase space grid
        points = self.disp_range[-1]
        axis = np.linspace(*self.disp_range)
        x, y = np.meshgrid(axis, axis)

        # fit the blob to 2D Gaussian
        sigmaz = 1 - 2*self.results['default'].threshold()
        mean_sigmaz = sigmaz.mean(axis=0).data

        ind = np.unravel_index(np.argmax(mean_sigmaz, axis=None), mean_sigmaz.shape)

        data = mean_sigmaz.ravel()
        initial_guess = (0.7, axis[ind[1]], axis[ind[0]], 0.5, 0.5, 0, 0)
        popt, pcov = curve_fit(gaussian2D, (x, y), data, p0=initial_guess)
        data_fitted = gaussian2D((x, y), *popt)
        self.popt = popt

        # plot data and fit
        ax = fig.add_subplot(111)
        im = ax.imshow(np.transpose(data.reshape(points, points)), origin='bottom',
                  extent=(x.min(), x.max(), y.min(), y.max()))
        ax.contour(x, y, np.transpose(data_fitted.reshape(points, points)), 2, colors='k')
        ax.set_title('(x,y)=(%.4f, %.4f)' %(popt[1], popt[2]), fontsize = 15)
        fig.colorbar(im)


def gaussian2D((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()
