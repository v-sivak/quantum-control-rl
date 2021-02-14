# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 15:46:08 2020

@author: Vladimir Sivak
"""
import numpy as np
from math import sqrt, pi
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

class SemiclassicalPhaseSpaceSimulator():
    
    def __init__(self):
        self.reset()
        self.chi = 180e3
    
    def reset(self):
        self.phase = dict(g=0, e=0)
        self.alpha = dict(g=0, e=0)
        self.angle = dict(g=0, e=0)

        nbar = np.load(r'Z:\tmp\for Vlad\from_vlad\nbar.npy')
        freq_g = np.load(r'Z:\tmp\for Vlad\from_vlad\freq_g.npy')
        freq_e = np.load(r'Z:\tmp\for Vlad\from_vlad\freq_e.npy')

        # self.spline_e = CubicSpline(nbar, freq_e)
        # self.spline_g = CubicSpline(nbar, freq_g)


    def nbar(self, s):
        return np.abs(self.alpha[s])**2

    def freq(self, s, nbar):
        # if s=='g': return self.spline_g(nbar)
        # if s=='e': return self.spline_e(nbar)
        sign = 1 if s=='g' else -1
        return 1/2*sign*184e3 + 1/2*sign*200*nbar - 120*nbar
    
    def rotate(self, s, phi):
        self.angle[s] += phi
    
    def displace(self, alpha):
        for s in ['g','e']:
            delta_alpha = alpha * np.exp(1j*self.angle[s])
            delta_phase = np.imag(delta_alpha*np.conj(self.alpha[s]))
            self.alpha[s] += delta_alpha
            self.phase[s] += delta_phase
    
    def delay(self, tau):
        for s in ['g','e']:
            phi = 2*pi * self.freq(s, self.nbar(s)) * tau 
            self.rotate(s, phi)

    def flip(self):
        param = {}
        param['e'] = self.phase['g']
        param['g'] = self.phase['e']
        self.phase = param

        param = {}
        param['e'] = self.angle['g']
        param['g'] = self.angle['e']
        self.angle = param

        param= {}
        param['e'] = self.alpha['g']
        param['g'] = self.alpha['e']
        self.alpha = param

    def ideal_conditional_displacement(self, beta, tau):
        alpha = beta / 2 / np.sin(2*pi*self.chi*tau)
        self.displace(alpha)
        self.delay(tau)
        self.displace(-alpha*np.cos(2*pi*self.chi*tau/2))
        self.flip()
        self.displace(-alpha*np.cos(2*pi*self.chi*tau/2))
        self.delay(tau)
        self.displace(alpha*np.cos(2*pi*self.chi*tau))
        return alpha


    def optimized_conditional_displacement(self, beta, tau):
    
        def cost_fn(alpha):
            phi = 2*pi*(self.freq('g',alpha**2)-self.freq('e',alpha**2))*tau
            return (beta - 2*alpha*np.sin(phi))**2
        
        def find_alpha(beta, tau):
            alpha_guess = beta / 2 / np.sin(2*pi*self.chi*tau)
            res = minimize(cost_fn, x0=alpha_guess, method='Nelder-Mead')
            return res.x  
        
        alpha = find_alpha(beta, tau)
        
        phi_g = 2*pi * self.freq('g', alpha**2) * tau
        phi_e = 2*pi * self.freq('e', alpha**2) * tau
        
        self.displace(alpha)
        self.delay(tau)
        self.displace(-alpha*np.cos((phi_g-phi_e)/2)*np.exp(-1j*(phi_e+phi_g)/2))
        self.flip()
        self.displace(-alpha*np.cos((phi_g-phi_e)/2)*np.exp(-1j*(phi_e+phi_g)/2))
        self.delay(tau)
        self.displace(alpha*np.cos(phi_g-phi_e)*np.exp(-1j*(phi_e+phi_g)))
        self.rotate('g', -(phi_e+phi_g))
        self.rotate('e', -(phi_e+phi_g))
        return alpha




sim = SemiclassicalPhaseSpaceSimulator()


# for different gate durations, plot the Re and Im of the CD amplitude
# also plot the required photon number during the execution of CD gate
beta_cd = sqrt(2*pi)
times = np.linspace(40, 400, 100) * 1e-9
beta = dict(g=[], e=[])
nbars = []
for t in times:
    sim.reset()
    alpha = sim.optimized_conditional_displacement(beta_cd, t)
    for s in ['g', 'e']:
        beta[s].append(sim.alpha[s])
    nbars.append(alpha**2)

fig, axes = plt.subplots(2,1, sharex=True)
axes[1].set_xlabel('tau (ns)')
colors = dict(g='red', e='black')
axes[0].set_ylabel('Re & Im')
axes[0].plot(times*1e9, np.ones_like(times)*beta_cd/2, linestyle='dotted', color='blue')
axes[0].plot(times*1e9, -np.ones_like(times)*beta_cd/2, linestyle='dotted', color='blue')
for s in ['g', 'e']:
    axes[0].plot(times*1e9, np.real(beta[s]), color=colors[s], linestyle='--')
    axes[0].plot(times*1e9, np.imag(beta[s]), color=colors[s], linestyle='-', label=s)
axes[0].legend(loc='upper right')
axes[1].set_ylabel('nbar')
axes[1].plot(times*1e9, nbars)

