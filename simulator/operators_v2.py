# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 22:38:34 2020

@author: Vladimir Sivak
"""
import tensorflow as tf
from tensorflow import complex64 as c64
from distutils.version import LooseVersion
from math import pi
import inspect

if LooseVersion(tf.__version__) >= "2.2":
    diag = tf.linalg.diag
else:
    import numpy as np
    diag = np.diag  # k=1 option is broken in tf.linalg.diag in TF 2.1 (#35761)

"""
Operators are wrapped as instances of tf.linalg.LinearOperator which gives 
access to a powerful API with a whole bunch of useful features, such as:

Attributes: shape, batch_shape, domain_dimension, is_self_adjoint, ...

Methods: adjoint, to_dense, eigvals, matmul, matvec, inverse, ...

More detailed documentation:
    https://www.tensorflow.org/api_docs/python/tf/linalg/LinearOperator

Example 1:
    
    I = identity(2) # identity on the qubit
    a_dag = create(100) # creation operator on the oscillator
    # construct an operator on a joint Hilbert space and act with it on vacuum
    fock1 = tf.linalg.LinearOperatorKronecker([I, a_dag]).matvec(vac)
"""

# TODO: maybe re-define matvec to include explicit normalization
# TODO: have different decorators for batched and un-batched operators
def batch_operator(func):
    """
    Wrap the output of <func> into tf.linalg.LinearOperator object.
    
    """
    def wrapper(*args, **kwargs):
        name = kwargs['name'] if 'name' in kwargs.keys() else func.__name__
        operator_matrix = func(*args, **kwargs)
        return tf.linalg.LinearOperatorFullMatrix(
            operator_matrix, is_square=True, name=name)
    return wrapper


### Constant operators

@batch_operator
def sigma_x():
    return tf.constant([[0., 1.], [1., 0.]], dtype=c64)


@batch_operator
def sigma_y():
    return tf.constant([[0.j, -1.j], [1.j, 0.j]], dtype=c64)


@batch_operator
def sigma_z():
    return tf.constant([[1., 0.], [0., -1.]], dtype=c64)


@batch_operator
def identity(N):
    """Returns an identity operator in the Fock basis.

    Args:
        N (int): Dimension of Hilbert space

    Returns:
        Tensor([N, N], tf.complex64): NxN identity operator
    """
    return tf.eye(N, dtype=c64)


@batch_operator
def destroy(N):
    """Returns a destruction (lowering) operator in the Fock basis.

    Args:
        N (int): Dimension of Hilbert space

    Returns:
        Tensor([N, N], tf.complex64): NxN creation operator
    """
    a = diag(tf.sqrt(tf.range(1, N, dtype=tf.float32)), k=1)
    return tf.cast(a, dtype=c64)


@batch_operator
def create(N):
    """Returns a creation (raising) operator in the Fock basis.

    Args:
        N (int): Dimension of Hilbert space

    Returns:
        Tensor([N, N], tf.complex64): NxN creation operator
    """
    return destroy(N).adjoint().to_dense()


@batch_operator
def num(N):
    """Returns the number operator in the Fock basis.

    Args:
        N (int): Dimension of Hilbert space

    Returns:
        Tensor([N, N], tf.complex64): NxN number operator
    """
    return tf.cast(diag(tf.range(0, N)), dtype=c64)

# TODO: add is_self_adjoint attribute to be passed to the wrapper
@batch_operator
def position(N):
    """Returns the position operator in the Fock basis.

    Args:
        N (int): Dimension of Hilbert space

    Returns:
        Tensor([N, N], tf.complex64): NxN position operator
    """
    # Preserve max precision in intermediate calculations until final cast
    sqrt2 = tf.sqrt(tf.constant(2, dtype=c64))
    a_dag = create(N).to_dense()
    a = destroy(N).to_dense()
    return tf.cast((a_dag + a) / sqrt2, dtype=c64)


@batch_operator
def momentum(N):
    """Returns the momentum operator in the Fock basis.

    Args:
        N (int): Dimension of Hilbert space

    Returns:
        Tensor([N, N], tf.complex64): NxN momentum operator
    """
    # Preserve max precision in intermediate calculations until final cast
    sqrt2 = tf.sqrt(tf.constant(2, dtype=c64))
    a_dag = create(N).to_dense()
    a = destroy(N).to_dense()
    return tf.cast(1j * (a_dag - a) / sqrt2, dtype=c64)


@batch_operator
def parity(N):
    """Returns the photon number parity operator in the Fock basis.

    Args:
        N (int): Dimension of Hilbert space

    Returns:
        Tensor([N, N], tf.complex64): NxN photon number parity operator
    """
    pm1 = tf.where(tf.math.floormod(tf.range(N),2)==1, -1, 1)
    return diag(tf.cast(pm1, dtype=c64))


### Parametrized operators

# TODO: generalize to arbitrary generators 
# (i.e. phase space rotations, Hamiltonian evolution etc)
    
class TranslationOperator():
    """ 
    Batch translation operator. 
    
    Example:
        T = TranslationOperator(100)
        alpha = tf.constant([1.23+0.j, 3.56j, 2.12+1.2j])
        T(alpha) # shape=[3,100,100]
        state = T(alpha).matvec(state)
    """
    def __init__(self, N):
        """
        Pre-diagonalize position and momentum operators.
        
        Args:
            N (int): Dimension of Hilbert space
        
        """
        self.N = N
        p = momentum(N).to_dense()
        q = position(N).to_dense()
        
        # Pre-diagonalize
        (self._eig_q, self._U_q) = tf.linalg.eigh(q)
        (self._eig_p, self._U_p) = tf.linalg.eigh(p)
        self._qp_comm = tf.linalg.diag_part(q @ p - p @ q)


    @batch_operator
    @tf.function
    def compute_BCH(self, amplitude, *args, **kwargs):
        """Calculates T(amplitude) for a batch of amplitudes using BCH.

        Args:
            amplitude (Tensor([B1, ..., Bb], c64)): A batch of amplitudes

        Returns:
            Tensor([B1, ..., Bb, N, N], c64): A batch of T(amplitude)
        """
        # Reshape amplitude for broadcast against diagonals
        amplitude = tf.cast(tf.expand_dims(amplitude, -1), dtype=c64)

        # Take real/imag of amplitude for the commutator part of the expansion
        re_a = tf.cast(tf.math.real(amplitude), dtype=c64)
        im_a = tf.cast(tf.math.imag(amplitude), dtype=c64)

        # Exponentiate diagonal matrices
        expm_q = tf.linalg.diag(tf.math.exp(1j * im_a * self._eig_q))
        expm_p = tf.linalg.diag(tf.math.exp(-1j * re_a * self._eig_p))
        expm_c = tf.linalg.diag(tf.math.exp(-0.5 * re_a * im_a * self._qp_comm))

        # Apply Baker-Campbell-Hausdorff
        return tf.cast(
            self._U_q
            @ expm_q
            @ tf.linalg.adjoint(self._U_q)
            @ self._U_p
            @ expm_p
            @ tf.linalg.adjoint(self._U_p)
            @ expm_c,
            dtype=c64,
        )
    
    def __call__(self, amplitude):
        return self.compute_BCH(amplitude, name='TranslationOperator')


class DisplacementOperator(TranslationOperator):
    """ 
    Batch displacement operator D(amplitude) = T(amplitude * sqrt(2)).
    
    """
    def __call__(self, amplitude):
        sqrt2 = tf.math.sqrt(tf.constant(2, dtype=c64))
        return self.compute_BCH(amplitude*sqrt2, name='DisplacementOperator')



### Quantum states

# TODO: make a new decorator

# TODO: make it work with arbirary batch_shape
@batch_operator
def Fock(N, n, batch_shape=[1]):
    return tf.stack([tf.one_hot(n, N, dtype=c64)]*batch_shape[0])

@batch_operator
def qstate(state='g', batch_shape=[1]):
    n = {'g':0, 'e':1}
    return tf.stack([tf.one_hot(n[state], 2, dtype=c64)]*batch_shape[0])