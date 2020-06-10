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


def gen_displace_BCH(N):
    """Returns a function to calculate displacements using tensorflow with the
    Baker-Campbell-Hausdorff formula (BCH).

    Using the BCH formula allows us to perform matrix exponentiation *only* on
    diagonal matrices, which removes the complicated `expm` implementation (using the
    scaling/squaring method and Pade approximation) and changes it to an exponential of
    scalar values along the diagonal.

    Although both should, in theory, still have an O(N^3) complexity, the constant
    multiplier is very different, and there may be additional speedups that can be
    applied to the O(N^3) matrix multiplication steps in BCH.

    Calling this function calculates the common variables and closes over them. Since
    we need to diagonalize q and p in this step, this may take significant time. Of
    course, calling the returned displace(alphas) will be much faster in comparison.

    Args:
        N (int): Dimension of Hilbert space

    Returns:
        Callable[[int], Tensor([num, N, N], c64)]: Displacement function for dim N
    """
    a = utils.destroy(N, dtype=tf.complex128)
    a_dag = utils.create(N, dtype=tf.complex128)

    # Convert raising/lowering to position/momentum
    sqrt2 = tf.math.sqrt(tf.constant(2, dtype=tf.complex128))
    q = (a_dag + a) / sqrt2
    p = (a_dag - a) * 1j / sqrt2

    # Diagonalize q and p
    eig_q, U_q = tf.linalg.eigh(q)
    eig_p, U_p = tf.linalg.eigh(p)
    U_q_dag = tf.linalg.adjoint(U_q)
    U_p_dag = tf.linalg.adjoint(U_p)

    # Calculate the commutator numerically. I'm not sure if this is a little strange in
    # theory, but assuming that [q, p] = j causes significant errors...
    comm = tf.linalg.diag_part(q @ p - p @ q)

    def displace(alphas):
        """Calculates D(alpha) for a batch of alphas

        Args:
            alphas (Tensor([num], c64)): A batch of num alphas

        Returns:
            Tensor([num, N, N], c64): A batch of D(alphas)
        """
        # Scale alpha and reshape from broadcast against the diagonals
        alphas = sqrt2 * tf.cast(
            tf.reshape(alphas, [alphas.shape[0], 1]), dtype=tf.complex128
        )

        # Take real/imag of alphas for the commutator part of the expansion
        re_a = tf.cast(tf.math.real(alphas), dtype=tf.complex128)
        im_a = tf.cast(tf.math.imag(alphas), dtype=tf.complex128)

        # Exponentiate diagonal matrices
        expm_q = tf.linalg.diag(tf.math.exp(1j * im_a * eig_q))
        expm_p = tf.linalg.diag(tf.math.exp(-1j * re_a * eig_p))
        expm_c = tf.linalg.diag(tf.math.exp(-0.5 * re_a * im_a * comm))

        # Apply Baker-Campbell-Hausdorff
        return tf.cast(
            U_q @ expm_q @ U_q_dag @ U_p @ expm_p @ U_p_dag @ expm_c,
            dtype=tf.complex64,
        )

    return displace
