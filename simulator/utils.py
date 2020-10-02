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


def expectation(psi, Op, reduce_batch=True):
    """
    Expectation of operator 'Op' in state 'psi'. 
    Supports various batching options:
        
        1) psi.shape = [batch_size,N]
           Op.shape  = [batch_size,N,N]
           
           Returns a batch of expectation values of shape=[batch_size,1] 
        
        2) psi.shape = [1,N]
           Op.shape  = [batch_size,N,N]
           
           Broadcasts 'psi' and returns a batch of expectation values of 
           shape=[batch_size,1] 

        3) psi.shape = [batch_size,N]
           Op.shape  = [N,N]
           
           If reduce_batch=False, returns a batch of expectation values of 
           shape=[batch_size,1]. If reduce_batch=True, reduces over a batch 
           of states and returns a single expectation value of shape [].

    """
    psi = normalize(psi)
    
    # batched operator, batched or single wavefunction
    if len(Op.shape) == 3:
        assert Op.shape[1] == Op.shape[2] # N
        assert Op.shape[1] == psi.shape[1] # N
        
        # if single wavefunction, broadcase to batched
        if psi.shape[0] == 1:
            psi = tf.broadcast_to(psi, [Op.shape[0],psi.shape[1]])
    
        assert Op.shape[0] == psi.shape[0] # batch_size
        
        expect_batch = batch_dot(tf.math.conj(psi), batch_dot(Op, psi))
        
    # single operator, batched wavefunction
    if len(Op.shape) == 2:
        assert Op.shape[0] == Op.shape[1] # N
        assert Op.shape[0] == psi.shape[1] # N

        expect_batch = batch_dot(tf.math.conj(psi), tf.linalg.matvec(Op, psi))

        if reduce_batch: 
            expect_batch_reduced = tf.math.reduce_mean(expect_batch)
            return expect_batch_reduced

    return expect_batch