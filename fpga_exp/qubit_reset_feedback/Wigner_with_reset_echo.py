# -*- coding: utf-8 -*-
"""
Created on Thu May  6 13:39:28 2021

@author: qulab
"""


from init_script import *
from scipy.optimize import curve_fit
import numpy as np

class Wigner_with_reset_echo(FPGAExperiment):
    alpha = FloatParameter(2.0)
    disp_range = RangeParameter((-4.0, 4.0, 101))
    flip_qubit = BoolParameter(True)

    echo_delay = IntParameter(0)
    final_delay = IntParameter(0)

    def sequence(self):

#        @subroutine
#        def reset_with_echo():
#            sync()
#            readout(wait_result=True, log=False)
#            sync()
#            qubit.flip() # echo pulse
#            sync()
#            delay(self.echo_delay)
#            if_then_else(qubit.measured_low(), 'flip', 'wait')
#            label_next('flip')
#            qubit.flip()
#            goto('continue')
#            label_next('wait')
#            delay(qubit.pulse.length)
#            label_next('continue')
#            delay(self.final_delay)
#            sync()

        @subroutine
        def reset_with_echo():
            sync()
            delay(self.echo_delay, channel=qubit.chan)
            qubit.flip() # echo pulse
            readout(wait_result=True, log=False, sync_at_beginning=False, m0='se')
            sync()
            if_then_else(qubit.measured_low(), 'flip', 'wait')
            label_next('flip')
            qubit.flip()
            goto('continue')
            label_next('wait')
            delay(qubit.pulse.length)
            label_next('continue')
            delay(self.final_delay)
            sync()


        with system.wigner_tomography(*self.disp_range):

            sync()
            if self.flip_qubit:
                qubit.flip()
            else:
                delay(qubit.pulse.length)
            sync()

            cavity.displace(self.alpha)
            reset_with_echo()


    def plot(self, fig, data):
        # create phase space grid
        points = self.disp_range[-1]
        axis = np.linspace(*self.disp_range)
        x, y = np.meshgrid(axis, axis)

        # fit the blob to 2D Gaussian
        sigmaz = 1 - 2*self.results['default'].threshold()
        mean_sigmaz = sigmaz.mean(axis=0).data

        data = mean_sigmaz.ravel()
        ind_max = np.unravel_index(np.argmax(mean_sigmaz, axis=None), mean_sigmaz.shape)
        ind_min = np.unravel_index(np.argmin(mean_sigmaz, axis=None), mean_sigmaz.shape)

        # plot data and fit
        ax = fig.add_subplot(111)
        im = ax.imshow(np.transpose(data.reshape(points, points)), origin='bottom',
                  extent=(x.min(), x.max(), y.min(), y.max()))

        ax.scatter([axis[ind_min[0]]],axis[ind_min[1]])
        ax.scatter([axis[ind_max[0]]],axis[ind_max[1]])
        fig.colorbar(im)

        try:
            initial_guess = (0.7, axis[ind_max[1]], axis[ind_max[0]], 0.5, 0.5, 0, 0)
            popt1, pcov1 = curve_fit(gaussian2D, (x, y), data, p0=initial_guess)
            self.popt1, self.pcov1 = popt1, pcov1

            initial_guess = (-0.7, axis[ind_min[0]], axis[ind_min[1]], 0.5, 0.5, 0, 0)
            popt2, pcov2 = curve_fit(gaussian2D, (x, y), data, p0=initial_guess)
            self.popt2, self.pcov2 = popt2, pcov2
        except:
            pass
        finally:
            self.popt = self.popt1
            data_fitted = gaussian2D((x, y), *self.popt)

            ax.contour(x, y, np.transpose(data_fitted.reshape(points, points)), 2, colors='k')
            ax.set_title('(x,y)=(%.2f, %.2f)' %(self.popt[1], self.popt[2]), fontsize = 15)




def gaussian2D((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()
