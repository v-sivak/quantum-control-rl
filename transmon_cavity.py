# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 15:35:07 2020

@author: Vladimir Sivak
"""

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
from tensorflow import complex128 as c64 # it is what it is
import matplotlib.pyplot as plt
from math import sqrt, pi
import simulator.operators as ops
from simulator.utils import tensor, Kronecker_product, basis, expectation

from distutils.version import LooseVersion
if LooseVersion(tf.__version__) >= "2.2":
    diag = tf.linalg.diag
else:
    import numpy as np
    diag = np.diag

"""
IMPORTANT: 
    For this simulation I found that it's necessary to use double precision,
    otherwise the results are complete shit. So before running this make sure 
    that c64 actually corresponds to tf.complex128 in this script as well as
    in simulator.utils and simulator.operators 
"""


N_qb = 200 # truncation in charge basis
L = 15 # truncation in transmon basis
N_cav = 1000

def Hamiltonian(Ej, Ec, freq_a, V):
    """
    In the final Hamiltonian the levels should be interpreted (qubit, cavity):
    
    eigvals[0] : (0,0)
    eigvals[1] : (0,1)
    eigvals[2] : (1,0)
    eigvals[4] : (1,1)
    eigvals[5] : (2,0)
    
    """
    plot_charge_matrix = False
    plot_transmon_levels = False
    
    # diagonalize transmon in charge basis
    n = tf.cast(diag(tf.range(-N_qb/2, N_qb/2)), c64)
    cos_phi = tf.cast((diag(tf.ones(N_qb-1), k=1) + diag(tf.ones(N_qb-1), k=-1))/2, c64)
    
    H = -Ej * cos_phi + 4 * Ec * n**2
    eigvals, U = tf.linalg.eigh(H) # H = U  @ diag(eigvals) @ U_dag
    
    # project onto subspace of L transmon levels
    H_L = tf.linalg.diag(eigvals[:L])
    n_L = (tf.linalg.adjoint(U) @ n @ U)[:L,:L]
    
    # operators in joint transmon-cavity Hilbert space
    H_cav = freq_a * tensor([ops.identity(L), ops.num(N_cav)])
    H_qb = tensor([H_L, ops.identity(N_cav)])
    H_coupling = V * tensor([n_L, ops.position(N_cav)])

    if plot_transmon_levels:
        fig, ax = plt.subplots(1,1)
        ax.set_ylabel('Transmon energy level (Hz)')
        ax.plot(range(25), eigvals[:25], marker='.', linestyle='none')
    
    if plot_charge_matrix:
        fig, ax = plt.subplots(1,1)
        ax.set_title('Charge matrix in transmon basis')
        ax.pcolormesh(range(L), range(L), np.transpose(np.abs(n_L)))

    H0 = H_cav + H_qb
    H = H0 + H_coupling
    
    # free up that sweet memory
    del(n, cos_phi, H_L, n_L, H_cav, H_qb, H_coupling)
    return H0, H

def plot_probs_folded(psi):
    psi_folded = tf.reshape(psi, shape=[L,N_cav])
    fig, ax = plt.subplots(1,1)
    ax.set_ylabel('Cavity levels')
    ax.set_xlabel('Transmon levels')
    ax.pcolormesh(np.arange(L), np.arange(N_cav), np.transpose(np.abs(psi_folded)**2))


# Measured observables
freq_c  = 4.480526e9
freq_q = 5.51491e9
alpha = 205e6
chi = 184e3


FIT_MODEL_PARAMETERS = False
# Fit model parameters using measured observables
if FIT_MODEL_PARAMETERS:
    # Calculate guess for model parameters from measured observables
    # (based on lowest order perturbation theory in g and cosine expansion)
    Ec = alpha
    Ej = (freq_q + Ec - (freq_q - freq_c - Ec) / (Ec + 2*chi) * chi)**2 / 8 / Ec
    freq_a = freq_c + (freq_q - freq_c - Ec)/(Ec + 2*chi) * chi
    g = sqrt((freq_q - freq_c - Ec) * (freq_q - freq_c + 2*chi) / (Ec + 2*chi)**2 * chi * Ec)
    phi_zpf = (2 * Ec / Ej) ** (1/4)
    V = sqrt(8) * g * phi_zpf
    
    def cost_fn(x):
        Ej, Ec, freq_a, V = x
        H0, H = Hamiltonian(Ej, Ec, freq_a, V)
        eigvals, _ = tf.linalg.eigh(H)
        
        cost_freq_q = (eigvals[2]-eigvals[0]-freq_q)**2 / freq_q
        cost_freq_c = (eigvals[1]-eigvals[0]-freq_c)**2 / freq_c
        cost_chi = ((eigvals[1]-eigvals[0])-(eigvals[4]-eigvals[2])-chi)**2 / chi
        cost_alpha = ((eigvals[2]-eigvals[0])-(eigvals[5]-eigvals[2])-alpha)**2 / alpha
        
        cost = tf.math.real(cost_freq_q + cost_freq_c + cost_chi + cost_alpha)
        return float(cost)
    
    # minimize the cost
    from scipy.optimize import minimize
    res = minimize(cost_fn, x0=[Ej, Ec, freq_a, V], method='Nelder-Mead',
                   options=dict(maxiter=300))
    Ej, Ec, freq_a, V = res.x
else:
    Ec = 1.88593725e+08
    Ej = 2.16094988e+10
    freq_a = 4.48103644e+09
    V = 2.28795771e+07


# create operators in the joint Hilbert space
H0, H = Hamiltonian(Ej, Ec, freq_a, V)
U = ops.HamiltonianEvolutionOperator(H)
U0 = ops.HamiltonianEvolutionOperator(H0)
D = ops.DisplacementOperator(N_cav, tensor_with=[ops.identity(L), None])
q = tensor([ops.identity(L), ops.position(N_cav)])
p = tensor([ops.identity(L), ops.momentum(N_cav)])


SIMULATE_TIME_EVOLUTION = False
# simulate rotation of the displaced state
# Because H=const, this can be done with large steps in time
if SIMULATE_TIME_EVOLUTION:
    dt = 10e-9
    STEPS = 100
    times = tf.range(STEPS, dtype=tf.float32) * dt
    
    alpha = 20
    vac = Kronecker_product([basis(0,L), basis(0,N_cav)])
    psi0 = tf.linalg.matvec(D(alpha), vac)
    psi, psi_int = psi0, psi0
    U_dt, U0_dt = U(dt), U0(dt)
    U0_t = ops.identity(L*N_cav)
    qs, ps = [], []
    
    for t in times:    
        qs.append(expectation(psi_int, q))
        ps.append(expectation(psi_int, p))
    
        psi = tf.linalg.matvec(U_dt, psi)
        U0_t = U0_dt @ U0_t
        # interaction picture to cancel out fast rotation at freq_q & freq_c
        psi_int = tf.linalg.matvec(tf.linalg.adjoint(U0_t), psi)
    
    plot_probs_folded(psi0)
    plot_probs_folded(psi)
    
    # plot time evolution of average position and momentum
    fig, ax = plt.subplots(1,1)
    ax.set_xlabel('Time (ns)')
    ax.plot(times*1e9, qs, label='q')
    ax.plot(times*1e9, ps, label='p')



# TODO: this is not reliable, the result strongly depends on dt
SIMULATE_DISPLACEMENT_SWEEP = False
# measure cavity rotation speed for array of displacement amplitudes
if SIMULATE_DISPLACEMENT_SWEEP:
    dt = 100e-9
    alphas = np.linspace(1, 24, 41)
    
    freqs = {}
    ps, qs = {}, {}
    U_dt, U0_dt = U(dt), U0(dt)
    for s in [0,1]: # qubit states
        freqs[s] = []
        ps[s], qs[s] = [], []
        vac = Kronecker_product([basis(s,L), basis(0,N_cav)])
        for alpha in alphas:
            psi0 = tf.linalg.matvec(D(tf.cast(alpha, c64)), vac)
            psi = tf.linalg.matvec(U_dt, psi0)
            # interaction picture to cancel out fast rotation at freq_q & freq_c
            psi_int = tf.linalg.matvec(tf.linalg.adjoint(U0_dt), psi)
            p_dt = expectation(psi_int, p)
            q_dt = expectation(psi_int, q)
            ps[s].append(p_dt)
            qs[s].append(q_dt)
            freq = - 1 / (2*pi*dt) * np.arctan(p_dt / q_dt)
            # freq = - 1 / (2*pi*dt) * np.arcsin(p_dt / sqrt(2) / alpha)
            freqs[s].append(freq)
    
    freq_avg = (freqs[0][0] + freqs[1][0]) / 2
    freqs = {s: np.array(freqs[s]) - freq_avg for s in [0,1]}
    
    nbars = alphas ** 2
    fig, ax = plt.subplots(1,1)
    ax.set_ylabel('Rotation frequency (kHz)')
    ax.set_xlim(-10,900)
    ax.set_ylim(-220,150)
    ax.plot(nbars, freqs[1]*1e-3)
    ax.plot(nbars, freqs[0]*1e-3)
    
    # load and plot data from experiment 
    fudge_factor = 1.47 # 1.47
    nbar_exp = np.load(r'Z:\tmp\for Vlad\from_vlad\nbar.npy')
    freq_g = np.load(r'Z:\tmp\for Vlad\from_vlad\freq_g.npy')*1e-3
    freq_e = np.load(r'Z:\tmp\for Vlad\from_vlad\freq_e.npy')*1e-3
    ax.plot(nbar_exp/fudge_factor, freq_g, marker='.', linestyle='none')
    ax.plot(nbar_exp/fudge_factor, freq_e, marker='.', linestyle='none')






DO_BASIC_QUANTUM_NUMBER_ASSIGNMENT = True

if DO_BASIC_QUANTUM_NUMBER_ASSIGNMENT:
    # diagonalize transmon in charge basis
    n = tf.cast(diag(tf.range(-N_qb/2, N_qb/2)), c64)
    cos_phi = tf.cast((diag(tf.ones(N_qb-1), k=1) + diag(tf.ones(N_qb-1), k=-1))/2, c64)
    H_transmon = -Ej * cos_phi + 4 * Ec * n**2
    eigvals_transmon, _ = tf.linalg.eigh(H_transmon) # H = U  @ diag(eigvals) @ U_dag
    eigvals_transmon -= eigvals_transmon[0]
    
    # diagonalize the full Hamiltonian
    eigvals, U = tf.linalg.eigh(H)
    
    r = np.zeros([L, N_cav]) # to store energy difference from uncoupled state
    for m in range(L):
        for n in range(N_cav):
            # simple state classification: find the closest hybridized state vector 
            psi = Kronecker_product([basis(m,L), basis(n,N_cav)])
            dot = tf.linalg.matmul(psi,U)
            ind = int(tf.math.argmax(tf.math.abs(dot),axis=1))
            dE = (eigvals[ind] - eigvals[0]) - (n * freq_a + eigvals_transmon[m])
            r[m,n] = dE
    
    
    fig, ax = plt.subplots(1,1)
    ax.set_xlabel('Transmon')
    ax.set_ylabel('Cavity')
    plt.pcolormesh(range(L), range(N_cav), np.transpose(r), cmap='seismic')
    
    
    # Plot rotation speed, E[n+1]-E[n] (why this?)
    fig, ax = plt.subplots(1,1)
    ax.set_ylabel('Rotation frequency (Hz)')
    ax.set_xlabel('nbar')
    d = {s:r[s][1:]-r[s][:-1] for s in range(L)}
    offset = (d[0][0] + d[1][0]) / 2
    ax.plot(range(N_cav-1), d[0]-offset)
    ax.plot(range(N_cav-1), d[1]-offset)
    ax.set_ylim(-200e3,200e3)
    
    # fudge_factor = 1
    # nbar_exp = np.load(r'Z:\tmp\for Vlad\from_vlad\nbar.npy')
    # freq_g = np.load(r'Z:\tmp\for Vlad\from_vlad\freq_g.npy')
    # freq_e = np.load(r'Z:\tmp\for Vlad\from_vlad\freq_e.npy')
    # ax.plot(nbar_exp/fudge_factor, freq_g, marker='.', linestyle='none')
    # ax.plot(nbar_exp/fudge_factor, freq_e, marker='.', linestyle='none')
    
    fudge_factor = 1
    nbar_exp = np.load(r'C:\Users\qulab\Downloads\nbar.npy')
    freq_g = np.load(r'C:\Users\qulab\Downloads\freq_g.npy')
    freq_e = np.load(r'C:\Users\qulab\Downloads\freq_e.npy')
    ax.plot(nbar_exp/fudge_factor, freq_g, marker='.', linestyle='none')
    ax.plot(nbar_exp/fudge_factor, freq_e, marker='.', linestyle='none')
    plt.tight_layout()