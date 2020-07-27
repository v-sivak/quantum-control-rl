"""
Helper methods for benchmarking

Created on Fri Jun 19 03:14:44 2020

@author: Henry Liu
"""
import time
from distutils.version import LooseVersion

import numpy as np
import tensorflow as tf

from . import err_checks

if LooseVersion(tf.__version__) >= "2.2":
    random_generator = tf.random.Generator
    diag = tf.linalg.diag
else:
    random_generator = tf.random.experimental.Generator
    diag = np.diag  # k=1 option is broken in tf.linalg.diag in TF 2.1 (#35761)

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
    return tf.linalg.adjoint(destroy(N, dtype=dtype))


def benchmark(
    gen_displace, N, num, batchsize, dataset=False, graph=False, check_err=True
):
    """Runs a randomized benchmark of displacement operator generation, D(alpha)

    Args:
        gen_displace ([type]): [description]
        N (int): Dimension of Hilbert space
        num (int): Total number of alphas to benchmark
        batchsize (int): Number of alphas per batch (only used with Dataset)
        dataset (bool, optional): Enable tf.data.Datset API. Defaults to False.
        graph (bool, optional): Enable tf.function compilation. Defaults to False.
        check_err (bool, optional): Check error against analytic coherent state. Defaults to True.

    Returns:
        (float x 4): (total time per alpha, loop time per alpha, mean error, max error)
    """
    alphas = random_alphas(num)

    # Dataset API adds some overhead, we can compare to the in-memory case
    if dataset:
        b_alphas = tf.data.Dataset.from_tensor_slices(alphas).batch(batchsize)

    # Initialize the constants used in the displacement generation
    # For the direct expm method, this is just creating a, a_dag
    # For BCH, we diagonalize the q, p operators
    start_time = time.perf_counter()
    f = gen_displace(N)
    init_time = time.perf_counter() - start_time

    if graph:  # Enable tf.function and tf.autograph
        f = tf.function(f)
        f = f.get_concrete_function(tf.TensorSpec(shape=[num], dtype=tf.complex64))

    # Repeat batch 3x to iron out timing fluctuations
    repeat_times = []
    for _ in range(3):
        if dataset:  # Loop through each Dataset batch
            start_time = time.perf_counter()
            for l_alpha in b_alphas:  # Be careful of memory limitations here
                results = f(l_alpha)  # Calculate error of last batch only
                alphas = l_alpha
            loop_time = time.perf_counter() - start_time
        else:
            start_time = time.perf_counter()
            results = f(alphas)
            loop_time = time.perf_counter() - start_time
        repeat_times.append(loop_time)

    # We take the minimum time from above. This is typically representative of
    # a lower bound, as higher times are often caused by other processes
    # interfering with timing accuracy. See Python's timeit.repeat docs.
    total_time = (min(repeat_times) + init_time) / num
    loop_time = min(repeat_times) / num

    if check_err:
        mean_err, max_err = err_checks.coeff_err(results, alphas)
    else:
        mean_err, max_err = float("inf"), float("inf")

    return total_time, loop_time, mean_err, max_err
