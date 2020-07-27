"""
Benchmark for scaling of BCH algorithm vs. number of alphas

For various Hilbert space dims, we benchmark with scaling number of alphas.
This gives an estimate of the BCH performance for larger spaces and sustained
throughput of alphas. For a given dimension, we'd expect a linear relationship with
number of alphas (which would show up as a constant line, since the time graphed is per
alpha), but there's some overhead and memory transfer to consider.

Created on Fri Jun 19 23:07:09 2020

@author: Henry Liu
"""
import time
import math
import socket
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from . import utils, displace

# gpu = tf.config.experimental.list_physical_devices('GPU')[0]
# tf.config.experimental.set_memory_growth(gpu, True)
# tf.debugging.set_log_device_placement(True)

total_t = []  # Total time per alpha, including initial diagonalization
loop_t = []  # Loop time per alpha, excluding diagonalization

sizes = [400, 800, 1600, 3200]  # Hilbert space sizes
num_alphas = [100, 500, 1000, 5000]  # Total number of displacements

for N in sizes:
    batchsize = math.floor(2.5e8 / N ** 2)  # Conservative estimate of max batchsize
    N_total_t = []  # Total times for this Hilbert space size (N)
    N_loop_t = []  # Loop times for this Hilbert space size (N)

    for num in num_alphas:
        total_time, loop_time, mean_err, max_err = utils.benchmark(
            displace.gen_displace_BCH, N, num, batchsize, dataset=True, check_err=False
        )

        print("%i\t%i\t%.2E\t%.2E" % (N, num, total_time, loop_time))
        N_total_t.append(total_time)
        N_loop_t.append(loop_time)

    total_t.append(N_total_t)
    loop_t.append(N_loop_t)

fig, ax = plt.subplots(1, 1)
ax.set_ylabel(r"Time Per $D(\alpha)$ (s)")
ax.set_xlabel("Number of Displacements")
ax.set_yscale("log")
plt.grid(True, which="both")
for idx, N in enumerate(sizes):
    ax.plot(num_alphas, total_t[idx], label="N={} (Total)".format(N))
    ax.plot(num_alphas, loop_t[idx], label="N={} (Loop)".format(N))
ax.legend()
plt.tight_layout()
output_dir = Path(__file__).parent.absolute().joinpath("results")
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir.joinpath(benchmarkN-%s-%i.png" % (socket.gethostname(), time.time())))
