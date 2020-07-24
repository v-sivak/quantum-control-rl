import time
from distutils.version import LooseVersion

import numpy as np
import tensorflow as tf

if LooseVersion(tf.__version__) >= "2.2":
    random_generator = tf.random.Generator
else:
    random_generator = tf.random.experimental.Generator

g1 = random_generator.from_non_deterministic_state()
g2 = g1.split()[0]


def random_alphas(num, maxval=5):
    """Generate random alphas for benchmarking:
        - Amplitude in [0, maxval)
        - Phase in [0, pi)

    Note that the max amplitude of 5 may return higher errors for small
    Fock spaces (e.g. N = 50).

    Args:
        num (int): Number of alphas to generate
        maxval (int, optional): Maximum amplitude range. Defaults to 5.

    Returns:
        Tensor([num], c64): Array of alphas in a tensor
    """
    phases = tf.complex(
        tf.constant(0, dtype=tf.float32),
        g1.uniform(
            [num], maxval=tf.constant(np.pi, dtype=tf.float32), dtype=tf.float32
        ),
    )
    amplitudes = tf.cast(
        g2.uniform(
            [num], maxval=tf.constant(maxval, dtype=tf.float32), dtype=tf.float32
        ),
        dtype=tf.complex64,
    )
    return amplitudes * tf.math.exp(phases)


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
