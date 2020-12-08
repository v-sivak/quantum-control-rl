from init_script import *
import numpy as np
import matplotlib.pyplot as plt


class Rabi_vs_detune(FPGAExperiment):

    selective = BoolParameter(False)
    amp = RangeParameter((-1.5, 1.5, 51))
    detune = RangeParameter((-100e3, 100e3, 11))
    loop_delay = FloatParameter(1e6)
    ssb_base = FloatParameter(-50e6)
    phi = FloatParameter(90.0)

    def sequence(self):
        pulse = qubit.selective_pulse if self.selective else qubit.pulse
        with qubit.scan_detune(*self.detune, ssb0=self.ssb_base):
            with pulse.scan_amplitude(*self.amp):
                readout(init_state='se')
                delay(1e3)
                pulse(amp='dynamic')
                readout()
                delay(self.loop_delay)

    def process_data(self):
        init_state = self.results['init_state'].threshold()
        self.results['thresholded'] = self.results['default'].threshold()
        self.results['postselected'] = self.results['default'].postselect(init_state, [0])[0].thresh_mean().data
        self.results['postselected'].ax_data = self.results['init_state'].ax_data[1:]
        self.results['postselected'].labels = self.results['init_state'].labels[1:]
        self.results['im'] = np.imag(np.average(self.results['default'].data, axis=0))
        self.results['im'].ax_data = self.results['default'].ax_data[1:]
        self.results['im'].labels = self.results['default'].labels[1:]
        self.results['re'] = np.real(np.average(self.results['default'].data, axis=0))
        self.results['re'].ax_data = self.results['default'].ax_data[1:]
        self.results['re'].labels = self.results['default'].labels[1:]
        self.results['rotated'] = np.imag(np.exp(1.0j*self.phi*np.pi/180.0)*np.average(self.results['default'].data, axis=0))
        self.results['rotated'].ax_data = self.results['default'].ax_data[1:]
        self.results['rotated'].labels = self.results['default'].labels[1:]
        self.results['crotated'] = np.real(np.exp(1.0j*self.phi*np.pi/180.0)*np.average(self.results['default'].data, axis=0))
        self.results['crotated'].ax_data = self.results['default'].ax_data[1:]
        self.results['crotated'].labels = self.results['default'].labels[1:]

        mean_data = np.mean(self.results['default'].data, axis=0)
        ind = np.unravel_index(np.argmax(mean_data, axis=None), mean_data.shape)
        self.results['detune_amp'] = (self.results['default'].ax_data[1][ind[0]],
                                      self.results['default'].ax_data[2][ind[1]])

    def plot(self, fig, data):

        detunings = data['default'].ax_data[1]
        amplitudes = data['default'].ax_data[2]
        mean_data = np.mean(data['default'].data, axis=0).real
        ind = np.unravel_index(np.argmax(mean_data, axis=None), mean_data.shape)
        opt_detune, opt_amp = detunings[ind[0]], amplitudes[ind[1]]

        ax = fig.add_subplot(111)
        ax.set_xlabel(data['default'].labels[1])
        ax.set_ylabel(data['default'].labels[2])
        ax.pcolormesh(detunings, amplitudes, np.transpose(mean_data))
        ax.scatter([opt_detune], [opt_amp])
        ax.text(opt_detune, opt_amp,
                'detuning = %.0f kHz \n amplitude = %.3f' %(opt_detune, opt_amp))
