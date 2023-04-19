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

def vec_to_complex(a):
    """
    Convert vectorized action of shape [batch_size,2] to complex-valued
    action of shape (batch_size,)

    """
    return tf.complex(a[:,0], a[:,1])
