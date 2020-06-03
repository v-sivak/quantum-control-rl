"""
A few different implementations of displacement operator calculation

Created on Wed Jun 10 12:44:33 2020

@author: Henry Liu
"""
import numpy as np
import scipy as sc
import tensorflow as tf

import utils


def gen_displace_scipy(N):
    """Returns a function to calculate displacements using scipy

    Calling this function calculates the common variables and closes over them.

    Args:
        N (int): Dimension of Hilbert space

    Returns:
        Callable[[int], Tensor([num, N, N])]: Displacement function for dim N
    """
    a = np.diag(np.sqrt(np.arange(1, N, dtype=np.complex64)), k=1)
    a_dag = np.conjugate(a.transpose())
    vexpm = np.vectorize(sc.linalg.expm, signature="(n,n)->(n,n)")

    def displace(alphas):
        """Calculates D(alpha) for a batch of alphas

        Args:
            alphas (Tensor([num])): A batch of num alphas

        Returns:
            Tensor([num, N, N]): A batch of D(alphas)
        """
        # Reshape for broadcasting [num, 1, 1]
        alphas = np.reshape(alphas, [alphas.shape[0], 1, 1])
        return vexpm(alphas * a_dag - np.conj(alphas) * a)

    return displace


def gen_displace(N):
    """Returns a function to calculate displacements using tensorflow expm

    Calling this function calculates the common variables and closes over them.

    Args:
        N (int): Dimension of Hilbert space

    Returns:
        Callable[[int], Tensor([num, N, N], c64)]: Displacement function for dim N
    """
    a = utils.destroy(N)
    a_dag = utils.create(N)

    def displace(alphas):
        """Calculates D(alpha) for a batch of alphas

        Args:
            alphas (Tensor([num], c64)): A batch of num alphas

        Returns:
            Tensor([num, N, N], c64): A batch of D(alphas)
        """
        # Reshape for broadcasting [num, 1, 1]
        alphas = tf.reshape(alphas, [alphas.shape[0], 1, 1])
        return tf.linalg.expm(alphas * a_dag - tf.math.conj(alphas) * a)

    return displace
