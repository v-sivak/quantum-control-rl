# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 11:51:29 2020

@author: qulab
"""

from init_script import *
import numpy as np
import matplotlib.pyplot as plt

class blind_fidelity(FPGAExperiment):

    delay_time = IntParameter(5e3)
    loop_delay= IntParameter(1e6)
    flip_qubit = BoolParameter(False)
    threshold_g = FloatParameter(0)
    threshold_e = FloatParameter(0)

    def sequence(self):

        qubit.flip()
        readout(traj_e='rel', state_e='se0')
        delay(self.delay_time)
        readout(traj_e='rel', state_e='se0')
        delay(self.loop_delay)


        readout(traj_g='rel', state_g='se0')
        delay(self.delay_time)
        readout(traj_g='rel', state_g='se0')
        delay(self.loop_delay)

    def update(self):
        readout.set_envelope(self.results['opt_env'].data)

    def plot(self, fig, data):

        # Plot mean e trajectory after postselection
        ax1 = fig.add_subplot(321)
        e = data['postselected_e'].data.mean(axis=0)
        ax1.set_title('e')
        ax1.get_xaxis().set_visible(False)
        ax1.plot(np.arange(len(e)), e.real, label='Re')
        ax1.plot(np.arange(len(e)), e.imag, label='Im')
        ax1.plot(np.arange(len(e)), np.abs(e), label='Abs')

        # Plot mean g trajectory after postselection
        ax2 = fig.add_subplot(323, sharex=ax1)
        g = data['postselected_g'].data.mean(axis=0)
        ax2.set_title('g')
        ax2.get_xaxis().set_visible(False)
        ax2.plot(np.arange(len(g)), g.real, label='Re')
        ax2.plot(np.arange(len(g)), g.imag, label='Im')
        ax2.plot(np.arange(len(e)), np.abs(g), label='Abs')

        # Plot the optimal demodulation envelope
        ax3 = fig.add_subplot(325, sharex=ax1)
        env = data['opt_env'].data
        ax3.set_title('optimal envelope')
        ax3.plot(np.arange(len(g)), env.real, label='Re')
        ax3.plot(np.arange(len(g)), env.imag, label='Im')
        ax3.plot(np.arange(len(e)), np.abs(env), label='Abs')

        # Plot histogram of 'e'
        ax4 = fig.add_subplot(322)
        e = (data['traj_e'].data[:,1,:]*env).sum(axis=1) # use only the second measurement
        e_post = (data['postselected_e'].data*env).sum(axis=1)
        fidelity = np.sum(e>0)/float(len(e))
        blind_fidelity = np.sum(e_post>0)/float(len(e_post))
        bins = 100
        binedges = np.linspace(min(e),max(e),bins+1)
        bincenters = [np.mean(binedges[i:i+1]) for i in range(len(binedges)-1)]
        H_e, _ = np.histogram(e, bins=binedges)
        H_e_post, _ = np.histogram(e_post, bins=binedges)
        ax4.plot(bincenters, H_e, label='%.3f'%fidelity)
        ax4.plot(bincenters, H_e_post, label='%.3f'%blind_fidelity)
        ax4.legend()

        # Plot histogram of 'g'
        ax5 = fig.add_subplot(324, sharex=ax4)
        g = (data['traj_g'].data[:,1,:]*env).sum(axis=1)
        g_post = (data['postselected_g'].data*env).sum(axis=1)
        fidelity = np.sum(g<0)/float(len(g))
        blind_fidelity = np.sum(g_post<0)/float(len(g_post))
        bins = 100
        binedges = np.linspace(min(g),max(g),bins+1)
        bincenters = [np.mean(binedges[i:i+1]) for i in range(len(binedges)-1)]
        H_g, _ = np.histogram(g, bins=binedges)
        H_g_post, _ = np.histogram(g_post, bins=binedges)
        ax5.plot(bincenters, H_g, label='%.3f'%fidelity)
        ax5.plot(bincenters, H_g_post, label='%.3f'%blind_fidelity)
        ax5.legend()

    def process_data(self):
        """
        Postselect trajectories based on two consequtive measurements and
        use them to define a better envelope and measure "blind" fidelity of readout.

        """
        thresh_g = self.threshold_g
        thresh_e = self.threshold_e
        mean = {}
        for s in ['g', 'e']:
            postselected = [] # list of indices which should be postselected
            state = self.results['state_'+s].data
            N = len(state)
            for i in range(int(N)):
                if (s=='e' and state[i,0]>thresh_e) or (s=='g' and state[i,0]<thresh_g):
                    postselected.append(i)
            data_postselected = self.results['traj_'+s].data[postselected,1,:]
            self.results.create('postselected_'+s, data_postselected)
            mean[s] = data_postselected.mean(axis=0)

        optimal_envelope = mean['e'].conj() - mean['g'].conj()
        self.results.create('opt_env', optimal_envelope)
