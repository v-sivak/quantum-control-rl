"""
Common operators for Quantum Harmonic Oscillator. Function names follow QuTIP.

Created on Sun Jul 26 20:55:36 2020

@author: Henry Liu
"""
from distutils.version import LooseVersion

import tensorflow as tf
from math import pi

if LooseVersion(tf.__version__) >= "2.2":
    diag = tf.linalg.diag
else:
    import numpy as np

    diag = np.diag  # k=1 option is broken in tf.linalg.diag in TF 2.1 (#35761)


def identity(N, dtype=tf.complex64):
    """Returns an identity operator in the Fock basis.

    Args:
        N (int): Dimension of Hilbert space
        dtype (tf.dtypes.DType, optional): Returned dtype. Defaults to c64.

    Returns:
        Tensor([N, N], dtype): NxN identity operator
    """
    return tf.eye(N, dtype=dtype)


def destroy(N, dtype=tf.complex64):
    """Returns a destruction (lowering) operator in the Fock basis.

    Args:
        N (int): Dimension of Hilbert space
        dtype (tf.dtypes.DType, optional): Returned dtype. Defaults to c64.

    Returns:
        Tensor([N, N], dtype): NxN creation operator
    """
    a = diag(tf.sqrt(tf.range(1, N, dtype=tf.float64)), k=1)
    return tf.cast(a, dtype=dtype)


def create(N, dtype=tf.complex64):
    """Returns a creation (raising) operator in the Fock basis.

    Args:
        N (int): Dimension of Hilbert space
        dtype (tf.dtypes.DType, optional): Returned dtype. Defaults to c64.

    Returns:
        Tensor([N, N], dtype): NxN creation operator
    """
    # Preserve max precision in intermediate calculations until final cast
    return tf.cast(tf.linalg.adjoint(destroy(N, dtype=tf.complex128)), dtype=dtype)


def num(N, dtype=tf.complex64):
    """Returns the number operator in the Fock basis.

    Args:
        N (int): Dimension of Hilbert space
        dtype (tf.dtypes.DType, optional): Returned dtype. Defaults to c64.

    Returns:
        Tensor([N, N], dtype): NxN number operator
    """
    return tf.cast(diag(tf.range(0, N)), dtype=dtype)


def position(N, dtype=tf.complex64):
    """Returns the position operator in the Fock basis.

    Args:
        N (int): Dimension of Hilbert space
        dtype (tf.dtypes.DType, optional): Returned dtype. Defaults to c64.

    Returns:
        Tensor([N, N], dtype): NxN position operator
    """
    # Preserve max precision in intermediate calculations until final cast
    sqrt2 = tf.sqrt(tf.constant(2, dtype=tf.complex128))
    a_dag = create(N, dtype=tf.complex128)
    a = destroy(N, dtype=tf.complex128)
    return tf.cast((a_dag + a) / sqrt2, dtype=dtype)


def momentum(N, dtype=tf.complex64):
    """Returns the momentum operator in the Fock basis.

    Args:
        N (int): Dimension of Hilbert space

    Returns:
        Tensor([N, N], c64): NxN momentum operator
    """
    # Preserve max precision in intermediate calculations until final cast
    sqrt2 = tf.sqrt(tf.constant(2, dtype=tf.complex128))
    a_dag = create(N, dtype=tf.complex128)
    a = destroy(N, dtype=tf.complex128)
    return tf.cast(1j * (a_dag - a) / sqrt2, dtype=dtype)


def parity(N, dtype=tf.complex64):
    """Returns the photon number parity operator in the Fock basis.

    Args:
        N (int): Dimension of Hilbert space
        dtype (tf.dtypes.DType, optional): Returned dtype. Defaults to c64.

    Returns:
        Tensor([N, N], dtype): NxN photon number parity operator
    """
    diag = tf.where(tf.math.floormod(tf.range(N),2)==1, -1, 1)
    return tf.linalg.diag(tf.cast(diag, tf.complex64))
