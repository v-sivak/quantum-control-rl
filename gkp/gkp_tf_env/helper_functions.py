#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 21:52:25 2020

@author: Vladimir Sivak
"""

import qutip as qt
import numpy as np
from math import pi, sqrt
import matplotlib.pyplot as plt

# TODO: add docs


def exp_decay(x, a, b):
    return a*np.exp(-x/b)

def gauss_decay(x, a, b):
    return a*np.exp(-(x/b)**2)

def GKP_state(N, S, chan):
    
    # Check if the matrix is simplectic
    Omega = np.array([[0,1],[-1,0]])
    if not np.allclose(S.T @ Omega @ S ,Omega):
        raise Exception('S is not symplectic')

    a  = qt.destroy(N)
    q_op = (a + a.dag())/sqrt(2.0)
    p_op = 1j*(a.dag() - a)/sqrt(2.0)
    
    # Deafine stabilizers
    Sq = (2j*sqrt(pi)*(S[0,0]*q_op + S[1,0]*p_op)).expm()
    Sp = (-2j*sqrt(pi)*(S[0,1]*q_op + S[1,1]*p_op)).expm()
    Sc = (2j*sqrt(pi)*((S[0][0]-S[0][1])*q_op + (S[1][0]-S[1][1])*p_op)).expm()
    stabilizers = {'S_p' : Sp, 'S_q' : Sq, 'S_c' : Sc}
    
    # Deafine Pauli operators
    z =  (1j*sqrt(pi)*(S[0,0]*q_op + S[1,0]*p_op)).expm()
    x = (-1j*sqrt(pi)*(S[0,1]*q_op + S[1,1]*p_op)).expm()
    y = (1j*sqrt(pi)*((S[0][0]-S[0][1])*q_op + (S[1][0]-S[1][1])*p_op)).expm()
    paulis = {'X' : x, 'Y' : y, 'Z' : z}

    displacements = {'S_q': 2*sqrt(pi)*(-S[1,0]+1j*S[0,0]),
                     'Z'  : sqrt(pi)*(-S[1,0]+1j*S[0,0]),
                     'S_p': 2*sqrt(pi)*(S[1,1]-1j*S[0,1]),
                     'X'  : sqrt(pi)*(S[1,1]-1j*S[0,1]),
                     'S_c': 2*sqrt(pi)*((S[1,1]-S[1,0])+1j*(S[0,0]-S[0,1])),
                     'Y'  : sqrt(pi)*((S[1,1]-S[1,0])+1j*(S[0,0]-S[0,1]))}
    
    ops = [Sq, Sp, Sc, x, y, z]    
    # Define Hermitian Paulis and stablizers
    ops = [(op + op.dag())/2.0 for op in ops]
    # pass them through the channel 
    ops = [chan(op) for op in ops]
    Sq, Sp, Sc, x, y, z = ops
    
    # find 'Z+' as groundstate of this Hamiltonian
    d = (- Sq - Sp - Sc - z).groundstate()
    zero = (d[1]).unit()
    one = (x*d[1]).unit()

    states = {'Z+' : zero, 
              'Z-' : one,
              'X+' : (zero + one).unit(), 
              'X-' : (zero - one).unit(),
              'Y+' : (zero + 1j*one).unit(), 
              'Y-' : (zero - 1j*one).unit()}

    return stabilizers, paulis, states, displacements


def epsilon_normalizer(N, epsilon):
    a = qt.destroy(N)
    n_op = a.dag()*a
    G = (-epsilon*n_op).expm()
    G_inv = (epsilon*n_op).expm()
    return lambda rho: G*rho*G_inv


def plot_wigner(state, cmap='seismic', title=None, savepath=None):
    xvec = np.linspace(-7,7,81) #(-5,5,81)
    W = qt.wigner(state, xvec, xvec, g=sqrt(2))
    fig, ax = plt.subplots(figsize=(6,5))
    p = ax.pcolormesh(xvec, xvec, W, cmap=cmap, vmin=-1, vmax=+1) #'RdBu_r'
    
    ax.plot([sqrt(pi), sqrt(pi)/2, 0, 0], [0, 0, sqrt(pi), sqrt(pi)/2], 
            linestyle='none', marker='.',color='black')
    
    fig.colorbar(p, ax=ax)
    if title:
        ax.set_title(title)
    if savepath:
        plt.savefig(savepath)
        
# TODO: add support for batch plotting
def plot_wigner_tf_wrapper(state, *args, **kwargs):
    try:
        assert state.shape[0]==1
    except:
        raise ValueError('Batch plotting is not supported')        
    state = state.numpy()[0]
    N = int(len(state))
    state = state.reshape((N,1))
    dims = [[N], [1]]
    state = qt.Qobj(state, dims=dims, type='ket')
    
    plot_wigner(state, *args, **kwargs)

