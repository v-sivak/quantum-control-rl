# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 15:08:33 2021

@author: qulab
"""


import os
import matplotlib.pyplot as plt
import numpy as np
from fpga_lib import *
from fpga_lib.dsl.result import Results
from scipy.optimize import curve_fit


exp_dir = r'D:\DATA\exp\gkp_exp.CD_gate.CD_fixed_time_amp_cal\archive'
fname = '20210521.h5'
group = 11
file_name = os.path.join(exp_dir, fname)

grp_name = str(group)
res = Results.create_from_file(file_name, grp_name)
data = res['postselected'].data.data
ax_data = res['postselected'].ax_data
labels = res['postselected'].labels
optimal_amps = res['optimal_amp'].data
tau = res.run_params['tau']
alphas, betas = ax_data

fig, ax = plt.subplots(1, 1)
ax.pcolormesh(ax_data[1], ax_data[0], data)
ax.set_xlabel(labels[1])
ax.set_ylabel(labels[0])
ax.plot(betas, optimal_amps, color='black', marker='o')

def linear(beta, a, b):
    return a * beta + b


# first clean up NaN from the array (where fit failed)
mask = np.isnan(optimal_amps)
optimal_amps = optimal_amps[np.where(mask==False)]
betas = betas[np.where(mask==False)]

# now clean up ouliers
if 1:
    popt, pcov = curve_fit(linear, betas, optimal_amps)
    a, b = popt
    
    sigma = np.abs(optimal_amps - linear(betas, *popt))
    mask = sigma>np.mean(sigma)
    optimal_amps = optimal_amps[np.where(mask==False)]
    betas = betas[np.where(mask==False)]

# now fit this to a linear function
popt, pcov = curve_fit(linear, betas, optimal_amps)
a, b = popt

betas_new = np.linspace(0, 0.5, 10)
ax.plot(betas_new, linear(betas_new, a, b), color='red')

calibrations_dir = r'D:\DATA\exp\2021-05-13_cooldown\CD_fixed_time_amp_cal'
calibrations_dir = os.path.join(calibrations_dir, 'tau='+ str(tau)+'ns')
if not os.path.isdir(calibrations_dir):
    os.mkdir(calibrations_dir)

filename = os.path.join(calibrations_dir, 'linear_fit.npz')
np.savez(filename, a=a, b=b, tau_ns=tau)
