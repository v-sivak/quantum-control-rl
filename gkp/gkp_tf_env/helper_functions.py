#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 21:52:25 2020

@author: Vladimir Sivak
"""
import os
from math import pi, sqrt
import qutip as qt
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py
from time import time
from scipy.optimize import curve_fit

# TODO: add docs

def exp_decay(x, a, b):
    return a*np.exp(-x/b)

def gauss_decay(x, a, b):
    return a*np.exp(-(x/b)**2)

def GKP_state(tensorstate, N, S):
    """
    Thanks to Alec for providing base code for this function.
    
    """
    # Check if the matrix is simplectic
    Omega = np.array([[0,1],[-1,0]])
    if not np.allclose(S.T @ Omega @ S ,Omega):
        raise Exception('S is not symplectic')

    a  = qt.destroy(N)
    q_op = (a + a.dag())/sqrt(2.0)
    p_op = 1j*(a.dag() - a)/sqrt(2.0)
    
    # Deafine stabilizers
    Sz = (2j*sqrt(pi)*(S[0,0]*q_op + S[1,0]*p_op)).expm()
    Sx = (-2j*sqrt(pi)*(S[0,1]*q_op + S[1,1]*p_op)).expm()
    Sy = (2j*sqrt(pi)*((S[0][0]-S[0][1])*q_op + (S[1][0]-S[1][1])*p_op)).expm()
    stabilizers = {'S_x' : Sx, 'S_z' : Sz, 'S_y' : Sy}    
    
    # Deafine Pauli operators
    z =  (1j*sqrt(pi)*(S[0,0]*q_op + S[1,0]*p_op)).expm()
    x = (-1j*sqrt(pi)*(S[0,1]*q_op + S[1,1]*p_op)).expm()
    y = (1j*sqrt(pi)*((S[0][0]-S[0][1])*q_op + (S[1][0]-S[1][1])*p_op)).expm()
    paulis = {'X' : x, 'Y' : y, 'Z' : z}

    displacements = {'S_z': 2*sqrt(pi)*(-S[1,0]+1j*S[0,0]),
                     'Z'  : sqrt(pi)*(-S[1,0]+1j*S[0,0]),
                     'S_x': 2*sqrt(pi)*(S[1,1]-1j*S[0,1]),
                     'X'  : sqrt(pi)*(S[1,1]-1j*S[0,1]),
                     'S_y': 2*sqrt(pi)*((S[1,1]-S[1,0])+1j*(S[0,0]-S[0,1])),
                     'Y'  : sqrt(pi)*((S[1,1]-S[1,0])+1j*(S[0,0]-S[0,1]))}
    
    # Define Hermitian Paulis and stablizers
    ops = [Sz, Sx, Sy, x, y, z]
    ops = [(op + op.dag())/2.0 for op in ops]
    # pass them through the channel 
    chan = epsilon_normalizer(N)
    ops = [chan(op) for op in ops]
    Sz, Sx, Sy, x, y, z = ops
    
    # find 'Z+' as groundstate of this Hamiltonian
    d = (- Sz - Sx - Sy - z).groundstate()
    zero = (d[1]).unit()
    one  = (x*d[1]).unit()

    states = {'Z+' : zero, 
              'Z-' : one,
              'X+' : (zero + one).unit(), 
              'X-' : (zero - one).unit(),
              'Y+' : (zero + 1j*one).unit(), 
              'Y-' : (zero - 1j*one).unit()}

    # Tensordot everything with qubit
    if tensorstate:
        for key, val in stabilizers.items():
            stabilizers[key] = qt.tensor(qt.identity(2), val)
        for key, val in paulis.items():
            paulis[key] = qt.tensor(qt.identity(2), val)
        for key, val in states.items():
            states[key] = qt.tensor(qt.basis(2,0), val)

    return stabilizers, paulis, states, displacements


def epsilon_normalizer(N, epsilon=0.1):
    a = qt.destroy(N)
    n_op = a.dag()*a
    G = (-epsilon*n_op).expm()
    G_inv = (epsilon*n_op).expm()
    return lambda rho: G*rho*G_inv



def plot_wigner(state, tensorstate, cmap='seismic', title=None, savepath=None):
    if tensorstate: state = qt.ptrace(state, 1)
    xvec = np.linspace(-7,7,81)
    W = qt.wigner(state, xvec, xvec, g=sqrt(2))
    fig, ax = plt.subplots(figsize=(6,5))
    # p = ax.pcolormesh(xvec, xvec, W, cmap=cmap, vmin=-1, vmax=+1) #'RdBu_r'
    # ax.plot([sqrt(pi), sqrt(pi)/2, 0, 0], [0, 0, sqrt(pi), sqrt(pi)/2], 
    #         linestyle='none', marker='.',color='black')

    p = ax.pcolormesh(xvec/sqrt(pi), xvec/sqrt(pi), W, cmap=cmap, vmin=-1, vmax=+1) #'RdBu_r'   
    ax.plot([1, 1/2, 0, 0], [0, 0, 1, 1/2], 
            linestyle='none', marker='.',color='black')
    fig.colorbar(p, ax=ax)
    plt.grid()
    if title: ax.set_title(title)
    if savepath: plt.savefig(savepath)


# TODO: add support for batch plotting
def plot_wigner_tf_wrapper(state, tensorstate=False, *args, **kwargs):
    try:
        assert state.shape[0]==1
    except:
        raise ValueError('Batch plotting is not supported')
    state = state.numpy()[0]

    if tensorstate:
        N = int(len(state) / 2)
        state = state.reshape((2*N,1))
        dims = [[2, N], [1, 1]]
    else:
        N = int(len(state))
        state = state.reshape((N,1))
        dims = [[N], [1]]
    
    state = qt.Qobj(state, dims=dims, type='ket')    
    plot_wigner(state, tensorstate, *args, **kwargs)


def vec_to_complex(a):
    """
    Convert vectorized action of shape [batch_size,2] to complex-valued
    action of shape (batch_size,)
    
    """
    return tf.complex(a[:,0], a[:,1])


def fit_logical_lifetime(env, policy, plot=True, save_dir=None, 
                         states=['X+', 'Y+', 'Z+'], reps=1):
    """
    Fit exponential decay of logical Pauli eigenstates and extract T1.
    
    Args:
        env (GKP): instance of GKP environment
        policy (TFPolicy): instance of TFPolicy
        plot (bool): flag to plot the T1 data together with the fit
        save_dir (str): directory to save data in hdf5 format
        states (list): list of Pauli eigenstates 'X+', 'Y+', 'Z+'
        reps (int): number of repetitions to do in series (use if can't batch 
                    enough episodes due to GPU memory limit)
    """
    B = env.batch_size
    rewards = {s : np.zeros((env.episode_length, B*reps)) for s in states}
    mean_rewards = {s : np.zeros(env.episode_length) for s in states}
    fit_params = {s : np.zeros(2) for s in states}

    steps = np.arange(env.episode_length)
    times = steps*float(env.step_duration)
    
    assert env.reward_mode in ['fidelity', 'fidelity_with_code_flips']
    
    for s in states:            
        if '_env' in env.__dir__(): 
            env._env.init = s
        else:
            env.init = s
        
        # Collect batch of episodes, loop if can't fit in GPU memory
        for i in range(reps):
            time_step = env.reset()
            policy_state = policy.get_initial_state(B)
            j = 0
            while not time_step.is_last()[0]:
                t = time()
                action_step = policy.action(time_step, policy_state)
                policy_state = action_step.state
                time_step = env.step(action_step.action)
                rewards[s][j][i*B:(i+1)*B] = time_step.reward
                j += 1
                print('%d: Time %.3f sec' %(j, time()-t))
        
        mean_rewards[s] = rewards[s].mean(axis=1) # average across episodes
        fit_params[s], _ = curve_fit(exp_decay, times, mean_rewards[s],
                               p0=[1, env.T1_osc])
    
    # Save data
    if save_dir:
        h5file = h5py.File(os.path.join(save_dir, 'logical_lifetime'), 'a')
        try:
            for s in states:
                grp = h5file.create_group(s)
                grp.create_dataset('rewards', data=rewards[s])
                grp.create_dataset('times (s)', data=times)
                grp.attrs['T1 (s)'] = fit_params[s][1]
                grp.attrs['Exp scaling factor'] = fit_params[s][0]
        finally:
            h5file.close()
    
    # Plot average reward from every time step and T1 exponential fit
    if plot:
        fig, ax = plt.subplots(1,1)
        ax.set_xlabel('Step')
        ax.set_ylabel(r'$\langle X\rangle$, $\langle Y\rangle$, $\langle Z\rangle$')
        palette = plt.get_cmap('tab10')
        for i, s in enumerate(states):
            ax.plot(steps, mean_rewards[s], color=palette(i),
                    linestyle='none', marker='.')
            ax.plot(steps, exp_decay(times, fit_params[s][0], fit_params[s][1]), 
                    label = s + ' : %.2f us' %(fit_params[s][1]*1e6),
                    linestyle='--', color=palette(i))
        ax.set_title('Logical lifetime')
        ax.legend()
    
    return fit_params