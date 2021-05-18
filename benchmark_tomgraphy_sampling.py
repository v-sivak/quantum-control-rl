# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 09:47:40 2020

@author: Vladimir Sivak
"""

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="1"


import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import complex64 as c64
from math import pi
import matplotlib.pyplot as plt
from rl_tools.tf_env import helper_functions as hf
from simulator.utils import expectation, basis, measurement, normalize, batch_dot
from simulator import operators as ops
import numpy as np

"""
Benchmarking of the overlap fidelity estimator.

Wigner / characteristic function are computed with Monte Carlo method by 
sampling points in phase space. The sampling distribution here is ~ Wigner**2 
according to these two papers:
    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.107.210404
    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.106.230501

"""

N = 40 # Hilbert space truncation

state = basis(2,N)
target_state = normalize(basis(2,N)+1/4*basis(1,N))[0]

window_size = 12
alpha_samples = 100
msmt_samples = 1

BUFFER_SIZE = 20000

T = ops.TranslationOperator(N)
parity = ops.parity(N)

F_true = float(tf.math.real(batch_dot(state, target_state))**2)
print('True overlap fidelity: %.5f' % F_true)

def phase_estimation(psi, U, angle, sample=False):
    I = ops.identity(N)
    angle = tf.cast(tf.reshape(angle, angle.shape+[1,1]), c64)
    phase = tf.math.exp(1j * angle)
    Kraus = {}
    Kraus[0] = 1/2*(I + phase*U)
    Kraus[1] = 1/2*(I - phase*U)
    return measurement(psi, Kraus, sample)

# Fill the buffer with serial rejection sampling (slow)
def fill_buffer(target_state, buffer=[], target_vals=[], samples=1000):
    # Distributions for rejection sampling
    L = window_size/2
    P = tfp.distributions.Uniform(low=[[-L,-L]], high=[[L,L]])
    P_v = tfp.distributions.Uniform(low=0.0, high=1.0)
        
    # rejection sampling of phase space points
    for i in range(samples):
        cond = True
        while cond:
            point = tf.squeeze(hf.vec_to_complex(P.sample()))
            target_state_translated = tf.linalg.matvec(T(-point), target_state)
            W_target = expectation(target_state_translated, parity)
            W_target = tf.math.real(tf.squeeze(W_target))
            cond = P_v.sample() > tf.math.abs(W_target)**2
        buffer.append(point)
        target_vals.append(W_target)
    return buffer, target_vals


# Fill buffer with batch rejection sampling (fast)
def fill_buffer_v2(target_state, buffer=[], target_vals=[], samples=1000):
    # Distributions for rejection sampling
    L = window_size/2
    P = tfp.distributions.Uniform(low=[[-L,-L]]*samples, high=[[L,L]]*samples)
    P_v = tfp.distributions.Uniform(low=[0.0]*samples, high=[1.0]*samples)
        
    # batch rejection sampling of phase space points
    cond = True
    accepted = tf.zeros([samples], tf.bool)
    accepted_points = tf.zeros([samples], tf.complex64)
    accepted_targets = tf.zeros([samples], tf.float32)
    while cond:
        points = tf.squeeze(hf.vec_to_complex(P.sample()))
        target_state_translated = tf.linalg.matvec(T(-points), target_state)
        W_target = expectation(target_state_translated, parity, reduce_batch=False)
        W_target = tf.math.real(tf.squeeze(W_target))
        reject = P_v.sample() > tf.math.abs(W_target)**2
        mask = tf.logical_and(tf.logical_not(reject), tf.logical_not(accepted))
        accepted = tf.logical_or(mask, accepted)
        accepted_points = tf.where(mask, points, accepted_points)
        accepted_targets = tf.where(mask, W_target, accepted_targets)
        cond = not tf.reduce_all(accepted)
    
    buffer += list(accepted_points)
    target_vals += list(accepted_targets)
    return buffer, target_vals

buffer, target_vals = fill_buffer_v2(target_state, samples=BUFFER_SIZE)
print('Buffer filled.')

def estimate_fidelity(samples, sample=True):
    # can uniformly sample points from the buffer
    index = tf.cast(tf.math.round(tf.random.uniform([samples])*len(buffer)), tf.int32)
    points, targets = tf.gather(buffer, index), tf.gather(target_vals, index)
    
    # do a single Wigner measurement in a batch of phase space points
    psi = tf.linalg.matvec(T(-points), state)
    _, msmt = phase_estimation(psi, parity, tf.zeros(samples), sample=sample)
    
    # Make a noisy Monte Carlo estimate of the overlap integral
    estimator = tf.squeeze(msmt) / targets
    estimator = tf.where(tf.math.is_nan(estimator), 0, estimator)
    estimator = tf.where(tf.math.is_inf(estimator), 0, estimator)
    return tf.reduce_mean(estimator)


# look at estimators with these many samples
num_points = [10,30,100,300,1000,3000,6000]

# for each type of estimator (i.e. a different number of phase space points), 
# construct a bunch of them and measure the mean and variance. 
means, stds = [], []
for n in num_points:
    F_samples = []
    for i in range(50):
        F = estimate_fidelity(n, sample=True)
        F_samples.append(F)
    mean, std = np.mean(F_samples), np.std(F_samples)
    print('(Mean, std) fidelity: (%.5f, %.5f)' %(mean, std))
    stds.append(std)
    means.append(mean)

# plot std of the estimator vs num_points. It should go as 1/sqrt(num_points)
fig, ax = plt.subplots(1,1)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylabel('Standard deviation of estimator')
ax.set_xlabel('Number of phase space points')
ax.plot(num_points, stds, marker='.')
plt.tight_layout()

fig, ax = plt.subplots(1,1)
ax.set_xscale('log')
ax.set_ylabel('Estimated fidelity')
ax.set_xlabel('Number of phase space points')
ax.errorbar(num_points, means, yerr=stds, linestyle='none', marker='.')
ax.plot(num_points, np.ones(len(num_points))*F_true)
plt.tight_layout()
# There might be some bias resulting from the finite buffer size. If we were
# re-sampling a new buffer for each iteration, there would be no bias.