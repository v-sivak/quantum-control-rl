"""
Benchmark to compare various methods of displacement calculation

For example, we compare scipy to TF, and expm vs the BCH formula

The benchmark.sbatch script has some typical benchmarks of interest (fits in memory,
larger than memory, different batch size)

Created on Fri Jun 19 03:14:44 2020

@author: Henry Liu
"""
import os
import time
import math
import socket
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# from rl_tools.utils.rl_train_utils import save_log
from . import utils, displace

# Import/initialize MPI if we detect running in slurm with multiple tasks
has_mpi = False
if os.environ.get("SLURM_NTASKS") and int(os.environ.get("SLURM_NTASKS")) > 1:
    from mpi4py import MPI

    has_mpi = True
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()


def mpi_master():
    return rank == 0 if has_mpi else True


# TODO: add some saving function for the data to plot offline
if __name__ == "__main__":
    """
    This script is for benchmarking of tensorflow against scipy on a simple
    task of generating multiple displacement operators which basically reduces
    to matrix exponetiation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num",
        help="Total alphas to generate. Set to 0 to make equal to batchsize.",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--dataset",
        help="Enable tf.data.Dataset. Required if --num is larger than memory.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--batchsize",
        help="tf.data.Datset batchsize. Set to 0 to calculate adaptively based on N",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--maxsize", help="Largest Hilbert space size", type=int, default=1600
    )
    parser.add_argument(
        "--minsize", help="Smallest Hilbert space size", type=int, default=50
    )
    parser.add_argument(
        "--stepsize", help="Hilbert space range stepsize", type=int, default=100
    )

    args = parser.parse_args()
    hilbert_space_range = np.arange(
        args.minsize, args.maxsize + 1, args.stepsize
    )  # Add 1 to args.maxsize to make inclusive

    times = defaultdict(list)  # Total time per alpha, incl. initial diagonalization
    loop_times = defaultdict(list)  # Loop time per alpha, excl. diagonalization
    reference = None  # To compare generated operators against a reference value

    def benchmark(name, gen_displace, N, **kwargs):
        """Wrapper/closure around utils.benchmark for convenience.

        This just allows us to include parsed args and save/print the benchmarked
        time and error more easily.

        Args:
            name (string): Name of benchmark (for plot label)
            gen_displace ((Callable[[int], Callable])): The displacement function generator
            N (int): Dimension of Hilbert space
        """
        # Set "adaptive" batchsize, roughly safe for both expm/BCH routines in 32GB
        if args.batchsize == 0:
            batchsize = math.floor(1e8 / N ** 2)
        else:
            batchsize = args.batchsize

        # Match total number to batchsize
        if args.num == 0:
            num = batchsize
        else:
            num = args.num

        total_time, loop_time, mean_err, max_err = utils.benchmark(
            gen_displace, N, num, batchsize, dataset=args.dataset, **kwargs
        )  # Run the benchmark

        times[name].append(total_time)
        loop_times[name].append(loop_time)

        if mpi_master():
            print(
                "Run %i %-20s\tMean err: %.2E  Max err: %.2E  Time/alpha: %.2E  Loop Time/alpha: %.2E"
                % (N, name, mean_err, max_err, total_time, loop_time)
            )

    if mpi_master():  # MPI MASTER
        # Check what goodies you have
        from tensorflow.python.client import device_lib

        gpus = tf.config.experimental.list_physical_devices("GPU")
        print("=" * 70)
        print(device_lib.list_local_devices())
        print("=" * 70)
        print(gpus)
        print("=" * 70)

        for N in hilbert_space_range:
            # benchmark("scipy", displace.gen_displace_scipy, N)

            # Run TensorFlow benchmarks
            benchmark("tf-eager", displace.gen_displace, N)
            benchmark("tf-eager-BCH", displace.gen_displace_BCH, N)

            if has_mpi:
                benchmark("tf-eager-BCH-MPI", displace.gen_displace_MPI_BCH, N)

            # Multi-GPU benchmark using CentralStorageStrategy
            # This will error if enabled without multiple compute devices available
            # benchmark("tf-eager-distribute", displace.gen_displace_distribute_BCH, N)

    else:  # MPI WORKER
        for N in hilbert_space_range:

            # We can just run benchmark again on the workers, the gen_displace_MPI_BCH
            # routine will automatically handle the worker logic
            benchmark(
                "tf-eager-BCH-MPI", displace.gen_displace_MPI_BCH, N, check_err=False,
            )

    if mpi_master():
        # Plot stuff
        fig, ax = plt.subplots(1, 1)
        ax.set_ylabel(r"Time Per $D(\alpha)$ (s)")
        ax.set_xlabel("Hilbert space size")
        plt.grid(True, which="both")

        for key, val in times.items():
            ax.plot(hilbert_space_range, val, label=key)

        for key, val in loop_times.items():
            ax.plot(hilbert_space_range, val, label="{}-loop".format(key))

        ax.legend()
        plt.figtext(0.5, 0.95, vars(args), ha="center")
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        output_dir = Path(__file__).parent.absolute().joinpath("results")
        output_dir.mkdir(parents=True, exist_ok=True)

        run = time.time()
        plt.savefig(
            output_dir.joinpath("benchmark-%s-%i.png" % (socket.gethostname(), run))
        )

        ax.set_yscale("log")
        ax.set_xscale("log")
        plt.savefig(
            output_dir.joinpath("benchmark-%s-%i-log.png" % (socket.gethostname(), run))
        )
        # Save stuff
        # dic = times
        # dic['hilbert_space_range'] = hilbert_space_range
        # dic['displacements'] = displacements
        # save_log(dic, groupname='analysis_pc', filename='tf_benchmarking.hdf5')
