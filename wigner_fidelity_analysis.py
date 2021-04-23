# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 13:44:39 2021

@author: Vladimir Sivak
"""


"""
This uses a usual definition of the Wigner function, such that it is confined
to [-2/pi, 2/pi] range. The qubit sigma_z gives [-1,1], so the measurement 
results are rescaled by scale=2/pi.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi
from simulator import operators, utils
import tensorflow as tf
from tensorflow import complex64 as c64
# from plotting import plot_config

# Get experimental data
scale = 2/pi

wigner = {}
data = np.load(r'Z:\tmp\for Vlad\from_vlad\wigner_fock_4.npz')
wigner['g'] = data['wigner_g'] * scale
wigner['e'] = data['wigner_e'] * scale
wigner['avg'] = (wigner['g'] + wigner['e']) / 2
xs, ys = data['xs'], data['ys']

# find the area of elementary square in phase space
A = (xs[-1] - xs[0]) / (len(xs)-1) * (ys[-1] - ys[0]) / (len(ys)-1)

def background_boundary(W):
    return np.mean([W[:,0], W[:,-1], W[0,:], W[-1,:]])

def background_corners(W):
    d = 4
    return np.mean([W[:d,:d], W[:d,-d:], W[-d:,:d], W[-d:,-d:]])

def background_diff(W):
    return np.mean(W - wigner['avg'])

background, norm, wigner_corrected, max_val = {}, {}, {}, {}
for s in ['g', 'e', 'avg']:
    # the integral in phase space is supposed to be 1
    background[s] = background_diff(wigner[s])
    norm[s] = np.sum(wigner[s] - background[s]) * A
    wigner_corrected[s] = (wigner[s] - background[s]) / norm[s]
    max_val[s] = np.max(np.abs(wigner_corrected[s])/scale)

# Now find target Wigner
N = 100 
fock = 4
state = utils.basis(fock,N)
D = operators.DisplacementOperator(N)
P = operators.parity(N)

x = tf.constant(xs, dtype=c64)
y = tf.constant(ys, dtype=c64)
one = tf.constant([1]*len(y), dtype=c64)
onej = tf.constant([1j]*len(x), dtype=c64)
grid = tf.tensordot(x, one, axes=0) + tf.tensordot(onej, y, axes=0)
grid_flat = tf.reshape(grid, [-1])

state = tf.broadcast_to(state, [grid_flat.shape[0], state.shape[1]])
state_translated = tf.linalg.matvec(D(-grid_flat), state)
W = scale * utils.expectation(state_translated, P, reduce_batch=False)
W_grid = tf.reshape(W, grid.shape)
wigner_target = W_grid.numpy().real


# Fidelity and purity
F, purity = {}, {}
for s in ['g', 'e', 'avg']:
    F[s] = pi * np.sum(wigner_corrected[s] * wigner_target) * A
    purity[s] = pi * np.sum(wigner_corrected[s]**2) * A

# Plot 2D Wigner
fig, axes = plt.subplots(3, 3, figsize=(10,10), sharey=True, sharex=True, dpi=300)
for i, s in enumerate(['g', 'e', 'avg']):
    # measured
    axes[i,0].pcolormesh(xs, ys, np.transpose(wigner[s]), 
                         cmap='RdBu_r', vmin=-scale, vmax=scale)
    axes[i,0].set_title('Measured '+s)
    axes[i,0].set_aspect('equal')
    # subtracted background and normalized
    axes[i,1].pcolormesh(xs, ys, np.transpose(wigner_corrected[s]), 
                         cmap='RdBu_r', vmin=-scale, vmax=scale)
    axes[i,1].set_title('Corrected '+s)
    axes[i,1].set_aspect('equal')
    # target
    axes[i,2].pcolormesh(x, y, np.transpose(wigner_target), 
                         cmap='RdBu_r', vmin=-scale, vmax=scale)
    axes[i,2].set_title('Target')
    axes[i,2].set_aspect('equal')


fig, ax = plt.subplots(1,1)
ax.pcolormesh(x, y, np.transpose(wigner['g']-wigner['e']), cmap='RdBu_r')





# # Plot 1D cuts of Wigner along x and p
# fig, axes = plt.subplots(2,1, sharex=True, dpi=300)
# axes[0].plot(xs, np.zeros_like(xs), color='k')
# axes[0].plot(xs, wigner_target[int(len(xs)/2),:], color='k', linestyle='--')
# axes[0].plot(xs, wigner[int(len(xs)/2),:])
# axes[0].plot(xs, wigner[0,:])
# axes[0].set_title('Re')
# axes[0].set_ylim(-scale,scale)

# axes[1].plot(ys, np.zeros_like(ys), color='k')
# axes[1].plot(ys, wigner_target[:,int(len(ys)/2)], color='k', linestyle='--')
# axes[1].plot(ys, wigner[:,int(len(ys)/2)])
# axes[1].plot(ys, wigner[:,0])
# axes[1].set_title('Im')
# axes[1].set_ylim(-scale,scale)



