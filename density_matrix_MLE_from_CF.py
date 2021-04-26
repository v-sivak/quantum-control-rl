# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 11:01:42 2021

@author: qulab
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

# TODO: add analytic construction of displacement matrix

# ----- load experimental CF
def get_normalized_exp_CF():
    data = np.load(r'E:\VladGoogleDrive\Qulab\GKP\MLE\binomial_pz.npz')
    cf_normalized = data['cf_scaled']
    xs, ys = data['betas_x'], data['betas_y']
    return cf_normalized, xs, ys

CF_exp, xs, ys = get_normalized_exp_CF()
CF_flat = tf.reshape(CF_exp, [-1])


N = 7 # dimension of the reconstructed density matrix

# ----- create displaced parity matrix
xs_mesh, ys_mesh = np.meshgrid(xs, ys, indexing='ij')
grid = tf.cast(xs_mesh + 1j*ys_mesh, c64)
grid_flat = tf.reshape(grid, [-1])

def create_disp_op_tf():
    from simulator import operators as ops
    N_large = 100 # dimension used to compute the tomography matrix
    D = ops.DisplacementOperator(N_large)
    # Convert to lower-dimentional Hilbert space; shape=[N_alpha,N,N]
    disp_op = D(grid_flat)[:,:N,:N] 
    return real(disp_op), imag(disp_op)

disp_re, disp_im = create_disp_op_tf()


# ----- create parameterization of the density matrix
A = tf.Variable(tf.random.uniform([N,N]), dtype=tf.float32, name='A')
B = tf.Variable(tf.random.uniform([N,N]), dtype=tf.float32, name='B')

def loss_fn():
    rho_im = B - tf.transpose(B)
    rho_re = A + tf.transpose(A)
    CF_re = trace(matmul(rho_re, disp_re) - matmul(rho_im, disp_im))
    CF_im = trace(matmul(rho_re, disp_im) + matmul(rho_im, disp_re))
    loss_re = tf.reduce_mean((real(CF_flat) - CF_re)**2)
    loss_im = tf.reduce_mean((imag(CF_flat) - CF_im)**2)
    return loss_re + loss_im

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
        

# ----- get the reconstructed density matrix and CF
rho_im, rho_re = B - tf.transpose(B), A + tf.transpose(A)
rho = tf.cast(rho_re, c64) +1j*tf.cast(rho_im, c64)

CF_re = trace(matmul(rho_re, disp_re) - matmul(rho_im, disp_im))
CF_im = trace(matmul(rho_re, disp_im) + matmul(rho_im, disp_re))

CF_re_reconstructed = tf.reshape(CF_re, grid.shape)
CF_im_reconstructed = tf.reshape(CF_im, grid.shape)

# Plot 2D CF
fig, axes = plt.subplots(2, 2, sharey=True, sharex=True, dpi=300, figsize=(6,6))
plot_kwargs = dict(cmap='seismic', vmin=-1, vmax=1)
for ax in axes.ravel(): ax.set_aspect('equal')
axes[0,0].set_title('Measured Re')
axes[0,1].set_title('Reconstructed Re')
axes[1,0].set_title('Measured Im')
axes[1,1].set_title('Reconstructed Im')

axes[0,0].pcolormesh(xs, ys, np.transpose(real(CF_exp)), **plot_kwargs)
axes[1,0].pcolormesh(xs, ys, np.transpose(imag(CF_exp)), **plot_kwargs)
axes[0,1].pcolormesh(xs, ys, np.transpose(CF_re_reconstructed), **plot_kwargs)
axes[1,1].pcolormesh(xs, ys, np.transpose(CF_im_reconstructed), **plot_kwargs)


# Plot density matrix
fig, axes = plt.subplots(1, 2, sharey=True, dpi=300)
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
import qutip as qt
state_target = (qt.basis(N,0)+qt.basis(N,4)).unit()
rho_target = tf.cast(qt.ket2dm(state_target).full(), c64)
fidelity = real(trace(matmul(rho_target, rho)))
print(f'Fidelity {fidelity}')

# Eigenvalues
eigvals, _ = tf.linalg.eigh(rho)
eigvals = tf.sort(real(eigvals), direction='DESCENDING')
print(f'Eigenvalues {eigvals}')