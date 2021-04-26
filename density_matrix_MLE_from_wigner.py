# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 19:32:52 2021

@author: Vladimir Sivak
"""

import tensorflow as tf
from tensorflow import complex64 as c64
from math import pi
import numpy as np
# Use the GitHub version of TFCO
# !pip install git+https://github.com/google-research/tensorflow_constrained_optimization
import tensorflow_constrained_optimization as tfco
import matplotlib.pyplot as plt

matmul = tf.linalg.matmul
real = tf.math.real
imag = tf.math.imag
trace = tf.linalg.trace

# TODO: add analytic construction of displaced parity

# ----- load experimental wigner
scale = 2/pi

def get_normalized_exp_wigner():
    wigner = {}
    data = np.load(r'Z:\tmp\for Vlad\from_vlad\wigner_fock_4_improved.npz')
    wigner['g'] = data['wigner_g'] * scale
    wigner['e'] = data['wigner_e'] * scale
    wigner['avg'] = (wigner['g'] + wigner['e']) / 2
    xs, ys = data['xs'], data['ys']
    
    # find the area of elementary square in phase space
    area = (xs[-1] - xs[0]) / (len(xs)-1) * (ys[-1] - ys[0]) / (len(ys)-1)
    # normalize wigner
    norm = np.sum(wigner['avg']) * area
    wigner_normalized = wigner['avg'] / norm
    return wigner_normalized, xs, ys

wigner_exp, xs, ys = get_normalized_exp_wigner()

wigner_flat = tf.reshape(wigner_exp, [-1])
wigner_flat = tf.cast(wigner_flat, dtype=tf.float32)

N = 7 # dimension of the reconstructed density matrix

# ----- create displaced parity matrix
xs_mesh, ys_mesh = np.meshgrid(xs, ys, indexing='ij')
grid = tf.cast(xs_mesh + 1j*ys_mesh, c64)
grid_flat = tf.reshape(grid, [-1])

def create_displaced_parity_tf():
    from simulator import operators as ops
    N_large = 100 # dimension used to compute the tomography matrix
    D = ops.DisplacementOperator(N_large)
    P = ops.parity(N_large)

    displaced_parity = matmul(matmul(D(grid_flat), P), D(-grid_flat))
    # Convert to lower-dimentional Hilbert space; shape=[N_alpha,N,N]
    displaced_parity = displaced_parity[:,:N,:N] 
    
    displaced_parity_re = real(displaced_parity)
    displaced_parity_im = imag(displaced_parity)
    return (displaced_parity_re, displaced_parity_im)

disp_parity_re, disp_parity_im = create_displaced_parity_tf()


# ----- create parameterization of the density matrix
A = tf.Variable(tf.random.uniform([N,N]), dtype=tf.float32, name='A')
B = tf.Variable(tf.random.uniform([N,N]), dtype=tf.float32, name='B')

def loss_fn():
    rho_im = B - tf.transpose(B)
    rho_re = A + tf.transpose(A)
    W = scale * trace(matmul(rho_re, disp_parity_re) - matmul(rho_im, disp_parity_im))
    loss = tf.reduce_mean((wigner_flat - W)**2)
    return loss

# ----- create constrainted minimization problem
class ReconstructionMLE(tfco.ConstrainedMinimizationProblem):
    def __init__(self, loss_fn, weights):
        self._loss_fn = loss_fn
        self._weights = weights

    @property
    def num_constraints(self):
        return 2

    def objective(self):
        return loss_fn()

    def constraints(self):
        A, B = self._weights
        # it works with inequality constraints
        trace_le_1 = trace(A + tf.transpose(A)) - 1
        trace_gr_1 = 1 - trace(A + tf.transpose(A))
        constraints = tf.stack([trace_le_1, trace_gr_1])
        return constraints


problem = ReconstructionMLE(loss_fn, [A, B])
 
optimizer = tfco.LagrangianOptimizer(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    num_constraints=problem.num_constraints)

for i in range(1000):
    optimizer.minimize(problem, var_list=[A, B])
    if i % 200 == 0:
        print(f'step = {i}')
        print(f'loss = {loss_fn()}')
        print(f'constraints = {problem.constraints()}')
        

rho_im, rho_re = B - tf.transpose(B), A + tf.transpose(A)
rho = tf.cast(rho_re, c64) +1j*tf.cast(rho_im, c64)

W = scale * trace(matmul(rho_re, disp_parity_re) - matmul(rho_im, disp_parity_im))
wigner_reconstructed = tf.reshape(W, grid.shape)


# Plot 2D Wigner
fig, axes = plt.subplots(1, 2, sharey=True, dpi=300)
fig.suptitle('Wigner')
axes[0].set_title('Measured')
axes[1].set_title('Reconstructed')
axes[0].set_aspect('equal')
axes[1].set_aspect('equal')

axes[0].pcolormesh(xs, ys, np.transpose(wigner_exp), 
                   cmap='RdBu_r', vmin=-scale, vmax=scale)
axes[1].pcolormesh(xs, ys, np.transpose(wigner_reconstructed), 
                   cmap='RdBu_r', vmin=-scale, vmax=scale)


# Plot density matrix
fig, axes = plt.subplots(1, 2, sharey=True, dpi=300)
fig.suptitle('Density matrix')
axes[0].set_title('Real')
axes[1].set_title('Imag')
axes[0].set_aspect('equal')
axes[1].set_aspect('equal')

axes[0].pcolormesh(np.transpose(rho_re), cmap='RdBu_r', vmin=-1, vmax=1)
axes[1].pcolormesh(np.transpose(rho_im), cmap='RdBu_r', vmin=-1, vmax=1)


# Purity
purity = real(trace(matmul(rho, rho)))
print(f'Purity {purity}')

# Fidelity
fock = 4
rho_target = tf.linalg.diag(tf.one_hot(fock, N, dtype=c64))
fidelity = real(trace(matmul(rho_target, rho)))
print(f'Fidelity {fidelity}')

# Eigenvalues
eigvals, _ = tf.linalg.eigh(rho)
eigvals = tf.sort(real(eigvals), direction='DESCENDING')
print(f'Eigenvalues {eigvals}')