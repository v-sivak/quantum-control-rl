"""
Simulator helper functions

Created on Sun Jul 26 20:55:36 2020

@author: Henry Liu
"""
import tensorflow as tf
from tensorflow.keras.backend import batch_dot


def matrix_flatten(tensor):
    """Takes a tensor of arbitrary shape and returns a "flattened" vector of matrices.

    This is useful to get the correct broadcasting shape for batch operations.

    Args:
        tensor (Tensor([x, y, ...])): A tensor with arbitrary shape

    Returns:
        Tensor([numx * numy * ..., 1, 1]): A "flattened" vector of matrices
    """
    tensor = tf.reshape(tensor, [-1])
    tensor = tf.reshape(tensor, shape=[tensor.shape[0], 1, 1])
    return tensor


def normalize(state, dtype=tf.complex64):
    """
    Batch normalization of the wave function.

    Input:
        state -- batch of state vectors; shape=[batch_size,NH]

    """
    norm = tf.math.real(batch_dot(tf.math.conj(state), state))
    norm = tf.cast(tf.math.sqrt(norm), dtype=dtype)
    state = state / norm
    return state


def expectation(psi, O, reduce_batch=True):
    """
    Expectation of operator 'O' with respect to a batch of states 'psi'.

    Input:
        psi (Tensor([batch_size,N], c64)): batch of state vectors
        O (Tensor([N,N], c64)): operator on a Hilbert space
        reduce_batch (bool): flag to reduce over batch dimension.
    Returns:
        O_batch (Tensor([batch_size,1], c64)): returns this if reduce_batch==False;
            expectation of 'O' in each state of the batch. 
        O_expect (Tensor([], c64)): returns this if reduce_batch==True;
            expectation of 'O' for ensamble of states in the batch.
    """    
    psi = normalize(psi)
    O_batch = batch_dot(tf.math.conj(psi), tf.linalg.matvec(O, psi))
    if reduce_batch: 
        O_expect = tf.math.reduce_mean(O_batch)
        return O_expect
    else:
        return O_batch