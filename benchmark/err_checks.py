"""
Various checks for the returned displacement operator.

In most cases, coeff_err is probably the one you want to use.

Created on Tue Jun 09 12:03:52 2020

@author: Henry Liu
"""
import tensorflow as tf
import numpy as np

from . import utils


def rel_diff(a, b):
    """Calculate the Fro-norm of the relative difference of the operator matrix values

    This seems to be more sensitive than is useful. Also, both expm and BCH can be
    wrong in certain conditions, so we don't necessarily have a "reference" value.

    Args:
        a (Tensor([num, N, N], c64)): Array of num D(alpha) operators
        b (Tensor([num, N, N], c64)): Array of num D(alpha) operators

    Returns:
        (float, float): (mean, max) of the relative difference
    """
    a = tf.cast(a, dtype=tf.complex128)
    b = tf.cast(b, dtype=tf.complex128)

    # Numerically, the values close to 0 are not going to be 0, so we need to filter
    # out small differences that may be large compared to an underlying reference of
    # "close to 0", i.e. 1e-8 / 1e-14 should be ignored. We can do this with
    # np.isclose:
    close = np.isclose(a, b, atol=1e-5)
    a = np.where(~close, a, 0)
    b = np.where(~close, b, 0)

    # Compute relative diff
    diffs = tf.norm(
        tf.math.divide_no_nan(2 * tf.math.abs(a - b), tf.math.abs(a) + tf.math.abs(b)),
        ord="fro",
        axis=[-2, -1],
    )
    return tf.math.reduce_mean(diffs), tf.math.reduce_max(diffs)


def comm_err(test):
    """Calculate error in expectation value of [a, a_dag]

    Args:
        test (Tensor([num, N, N], c64)): Array of num D(alpha) operators

    Returns:
        (float, float): (mean, max) of % diff in commutator
    """
    # Construct operators
    a = utils.destroy(test.shape[1], dtype=tf.complex128)
    a_dag = utils.create(test.shape[1], dtype=tf.complex128)
    comm = a @ a_dag - a_dag @ a

    # Prepare coherent state from our test
    coh = coherent(test)

    # <alpha|[a, a_dag]|alpha> should be 1
    expected = tf.constant(1, dtype=tf.complex128)
    evs = tf.reduce_sum(tf.math.conj(coh) * tf.linalg.matvec(comm, coh), 1)
    diffs = tf.math.abs(evs - expected)
    return tf.math.reduce_mean(diffs), tf.math.reduce_max(diffs)


def coeff_err_states(test, alphas):
    """Calculate the number of Fock states where the coherent state coefficients
    deviate from analytic by >0.1% (lower is better)

    Args:
        test (Tensor([num, N, N], c64)): Array of num D(alpha) operators
        alphas (Tensor([num], c64)): Array of num alphas

    Returns:
        (float, float): (mean, max) number of Fock states above threshold
    """
    analytic = coherent_analytic(test.shape[1], alphas)
    numeric = coherent(test)

    # Check if values are within 0.1%
    # atol probably can be lower given experimental measurement precision?
    close = np.isclose(numeric, analytic, rtol=1e-3, atol=1e-8)
    num_not_close = tf.reduce_sum(tf.cast(~close, dtype=tf.uint16), axis=1)

    return tf.reduce_mean(num_not_close), tf.reduce_max(num_not_close)


def coeff_err(test, alphas):
    """Calculate the absolute error of the coherent state coefficients versus analytic

    For a given state, we sum the errors, which should provide somewhat of a weighting
    by state amplitudes.

    Args:
        test (Tensor([num, N, N], c64)): Array of num D(alpha) operators
        alphas (Tensor([num], c64)): Array of num alphas

    Returns:
        (float, float): (mean, max) of sum of state amplitude absolute error
    """
    analytic = coherent_analytic(test.shape[1], alphas)
    numeric = coherent(test)

    abs_diff = tf.reduce_sum(tf.math.abs(numeric - analytic), axis=1)
    return tf.reduce_mean(abs_diff), tf.reduce_max(abs_diff)


##################
# Helper Methods #
##################


def coherent(displ_op):
    """Generate coherent state (displaced vacuum) from displacement op

    Args:
        displ_op (Tensor([num, N, N], c64)): Array of num D(alpha) operators

    Returns:
        Tensor([num, N], c64): Array of num coherent states
    """
    displ_op = tf.cast(displ_op, dtype=tf.complex128)
    vac = tf.cast(tf.one_hot(0, displ_op.shape[1]), dtype=tf.complex128)
    return tf.linalg.matvec(displ_op, vac)


def coherent_analytic(N, alphas):
    """Compute coherent state with analytic formula

    Same as http://qutip.org/docs/latest/modules/qutip/states.html#coherent

    Args:
        N (int): Size of Fock space
        alphas (Tensor([num], c64)): Array of num alphas

    Returns:
        Tensor([num, N], c128): Array of num coherent states
    """
    alphas = tf.cast(alphas, dtype=tf.complex128)
    alphas = tf.reshape(alphas, [alphas.shape[0], 1])

    # Calculate coefficient of |0>
    zero_state = tf.cast(
        tf.math.exp(-0.5 * tf.math.square(tf.math.abs(alphas))), dtype=tf.complex128,
    )

    # Calculate alpha / sqrt(n)
    sqrtn = tf.sqrt(tf.cast(tf.range(1, N), dtype=tf.complex128))
    higher_states = tf.math.divide_no_nan(alphas, sqrtn)

    return tf.math.cumprod(tf.concat([zero_state, higher_states], axis=1), axis=1)
