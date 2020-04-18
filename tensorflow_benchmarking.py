# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 12:57:18 2020

@author: Vladimir Sivak
"""

import os
os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"]="2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


import numpy as np
import scipy as sc
import tensorflow as tf
import matplotlib.pyplot as plt
from time import time
from numpy import sqrt, pi

from gkp.utils.rl_train_utils import save_log



# This works in nightly version, argument k is ignored in tf2.1
# TODO: update the script after new stable realease 
# a = tf.linalg.diag(tf.math.sqrt(tf.range(N, dtype=tf.float32)), k=1)

# TODO: fix this one
# @tf.function
# def batch_displacement(N, displacements):
#     # constract oscillator creation/destruction operators 
#     a = [[0 if j!=i+1 else sqrt(j) for j in tf.range(N)] for i in tf.range(N)]
#     a = tf.constant(a, dtype=tf.complex64, shape = [1,N,N])
#     a_dag = tf.linalg.adjoint(a)
#     # create batch of displacements and exponentiate
#     tf_alpha = tf.cast(displacements, dtype=tf.complex64)
#     tf_alpha = tf.reshape(tf_alpha, [tf_alpha.shape[0],1,1])
#     tf_batch = tf_alpha * a_dag - tf.math.conj(tf_alpha) * a
#     exp =  tf.linalg.expm(tf_batch)
#     return exp


if __name__ == '__main__':
    """
    This script is for benchmarking of tensorflow against scipy on a simple
    task of generating multiple displacement operators which basically reduces 
    to matrix exponetiation. 
    
    """
    # Check what goodies you have
    from tensorflow.python.client import device_lib
    print('='*70)
    print(device_lib.list_local_devices())
    print('='*70)
    print(tf.config.experimental.list_physical_devices('GPU'))
    print('='*70)

    # Setup benchmarking parameters
    displacements = [3*np.exp(1j*phi) for phi in np.linspace(0,pi,100)]
    displacements = np.array(displacements, dtype=complex)
    Hilbert_space_size = np.arange(50, 450, 50)

    
    times = {
        'scipy' : [],
        'tf-CPU i7-7700K' : [],
        'tf-GPU 1050Ti' : [],
        'tf-GPU 2080Ti' : []
    }

    for j, N in enumerate(Hilbert_space_size):

        ### Scipy
        t = time()
        a = np.array(([[0 if j!=i+1 else sqrt(j) for j in range(N)]
                       for i in range(N)]))
        a_dag = np.conjugate(a.transpose())
        exp = []
        for alpha in displacements:
            D = sc.linalg.expm((alpha * a_dag - np.conj(alpha) * a))
            exp.append(D)
        times['scipy'].append(time()-t)

        ### TensorFlow CPU
        t = time()
        with tf.device('/device:CPU:0'):
            # constract oscillator creation/destruction operators 
            a = [[0 if j!=i+1 else sqrt(j) for j in range(N)] for i in range(N)]
            a = tf.constant(a, dtype=tf.complex64, shape = [1,N,N])
            a_dag = tf.linalg.adjoint(a)
            # create batch of displacements and exponentiate
            tf_alpha = tf.constant(displacements, dtype=tf.complex64)
            tf_alpha = tf.reshape(tf_alpha, [tf_alpha.shape[0],1,1])
            tf_batch = tf_alpha * a_dag - tf.math.conj(tf_alpha) * a
            exp =  tf.linalg.expm(tf_batch)
        times['tf-CPU i7-7700K'].append(time()-t)

        ### TensorFlow GPU
        t = time()
        with tf.device('/device:GPU:0'):
            # constract oscillator creation/destruction operators 
            a = [[0 if j!=i+1 else sqrt(j) for j in range(N)] for i in range(N)]
            a = tf.constant(a, dtype=tf.complex64, shape = [1,N,N])
            a_dag = tf.linalg.adjoint(a)
            # create batch of displacements and exponentiate
            tf_alpha = tf.constant(displacements, dtype=tf.complex64)
            tf_alpha = tf.reshape(tf_alpha, [tf_alpha.shape[0],1,1])
            tf_batch = tf_alpha * a_dag - tf.math.conj(tf_alpha) * a
            exp =  tf.linalg.expm(tf_batch)
        times['tf-GPU 2080Ti'].append(time()-t)

        ### TensorFlow GPU
        t = time()
        with tf.device('/device:GPU:1'):
            # constract oscillator creation/destruction operators 
            a = [[0 if j!=i+1 else sqrt(j) for j in range(N)] for i in range(N)]
            a = tf.constant(a, dtype=tf.complex64, shape = [1,N,N])
            a_dag = tf.linalg.adjoint(a)
            # create batch of displacements and exponentiate
            tf_alpha = tf.constant(displacements, dtype=tf.complex64)
            tf_alpha = tf.reshape(tf_alpha, [tf_alpha.shape[0],1,1])
            tf_batch = tf_alpha * a_dag - tf.math.conj(tf_alpha) * a
            exp =  tf.linalg.expm(tf_batch)    
        times['tf-GPU 1050Ti'].append(time()-t)

        ### Tensorflow function
        # t = time()
        # tf_displacements = tf.constant(displacements, dtype=tf.complex64)
        # tf_N = tf.constant(N)
        # batch_displacement(tf_N, tf_displacements)
        # tf_fn_gpu_time.append(time()-t)

    # Plot stuff
    fig, ax = plt.subplots(1,1)
    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Hilbert space size')
    ax.set_yscale('log')
    plt.grid(True, which="both")
    for key, val in times.items():
        ax.plot(Hilbert_space_size, val, label=key)
    ax.legend()

    # Save stuff
    # dic = times
    # dic['Hilbert_space_size'] = Hilbert_space_size
    # dic['displacements'] = displacements
    # save_log(dic, groupname='analysis_pc', filename='tf_benchmarking.hdf5')



