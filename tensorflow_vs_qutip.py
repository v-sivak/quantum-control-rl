# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 10:30:25 2020

@author: Vladimir Sivak
"""

# use this if you are running multiple scripts on multiple GPUs
import os
os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"]="2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


import numpy as np
import qutip as qt
import scipy as sc
import tensorflow as tf
import matplotlib.pyplot as plt
from time import time
from numpy import sqrt, pi




"""
Use this to determine your hardware names:
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices()) 
"""

if __name__ == '__main__':


    displacements = [2*sqrt(pi), -2*sqrt(pi)*1j, 0.4, -0.4, 0.1, 3j, -3.5j] # displacements values
    displacements = [3*np.exp(1j*phi) for phi in np.linspace(0,pi,100)]
    displacements = np.array(displacements, dtype=complex)
    # Hilbert_space_size = np.power(2,range(1,12))
    Hilbert_space_size = np.arange(50,250,50)
    points = len(Hilbert_space_size)
    average = 1
    
    qt_time = np.zeros((average,points))
    sc_time = np.zeros((average,points))
    tf_time = np.zeros((average,points))
    tf_gpu1050_time = np.zeros((average,points))
    tf_gpu2080_time = np.zeros((average,points))


    for i in range(average):
        for j, N in enumerate(Hilbert_space_size):
            ### Qutip
            t = time()
            for alpha in displacements:
                D1 = qt.displace(N, alpha)
            qt_time[i,j] = time()-t
    
            ### Scipy
            t = time()
            a = np.array(([[0 if j!=i+1 else sqrt(j) for j in range(N)] 
                           for i in range(N)]))
            a_dag = np.conjugate(a.transpose())
            for alpha in displacements:
                D2 = qt.Qobj(sc.linalg.expm((alpha * a_dag - np.conj(alpha) * a)))
            sc_time[i,j] = time()-t

            ### TensorFlow CPU
            t = time()
            with tf.device('/device:CPU:0'):
                a = tf.constant([[0 if j!=i+1 else sqrt(j) for j in range(N)] 
                                 for i in range(N)], dtype=tf.complex64)
                a_dag = tf.linalg.adjoint(a)
                batch = tf.convert_to_tensor([alpha * a_dag - np.conj(alpha) * a 
                                              for alpha in displacements])
                exp =  tf.linalg.expm(batch).numpy()
                for k, alpha in enumerate(displacements):
                    D3 = qt.Qobj(exp[k])
            tf_time[i,j] = time()-t

            ### TensorFlow GPU
            t = time()
            with tf.device('/device:GPU:0'):
                a = tf.constant([[0 if j!=i+1 else sqrt(j) for j in range(N)] 
                                 for i in range(N)], dtype=tf.complex64)
                a_dag = tf.linalg.adjoint(a)
                batch = tf.convert_to_tensor([alpha * a_dag - np.conj(alpha) * a 
                                              for alpha in displacements])
                exp =  tf.linalg.expm(batch).numpy()
                for k, alpha in enumerate(displacements):
                    D4 = qt.Qobj(exp[k])
            tf_gpu2080_time[i,j] = time()-t

            ### TensorFlow GPU
            t = time()
            with tf.device('/device:GPU:1'):
                a = tf.constant([[0 if j!=i+1 else sqrt(j) for j in range(N)] 
                                 for i in range(N)], dtype=tf.complex64)
                a_dag = tf.linalg.adjoint(a)
                batch = tf.convert_to_tensor([alpha * a_dag - np.conj(alpha) * a 
                                              for alpha in displacements])
                exp =  tf.linalg.expm(batch).numpy()
                for k, alpha in enumerate(displacements):
                    D4 = qt.Qobj(exp[k])
            tf_gpu1050_time[i,j] = time()-t


    fig, ax = plt.subplots(1,1)
    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Hilbert space size')
    ax.plot(Hilbert_space_size, qt_time.mean(axis=0), label='qutip')
    ax.plot(Hilbert_space_size, tf_time.mean(axis=0), label='tensorflow-CPU')    
    ax.plot(Hilbert_space_size, tf_gpu1050_time.mean(axis=0), label='tensorflow-GPU-1050Ti')  
    ax.plot(Hilbert_space_size, tf_gpu2080_time.mean(axis=0), label='tensorflow-GPU-2080Ti')  
    ax.plot(Hilbert_space_size, sc_time.mean(axis=0), label='scipy')
    ax.legend()





