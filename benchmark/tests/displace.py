"""
Tests for displace

Created on Thu Jun 25 11:42:08 2020

@author: Henry Liu
"""
import sys, os
import unittest

import tensorflow as tf

# This is for "multi-GPU" later, we just make 3 virtual CPUs to test
# We need to do this before importing the rest (which initializes TF)
tf.config.set_logical_device_configuration(
    tf.config.list_physical_devices("CPU")[0],
    [
        tf.config.LogicalDeviceConfiguration(),
        tf.config.LogicalDeviceConfiguration(),
        tf.config.LogicalDeviceConfiguration(),
    ],
)

# Prepend parent directory to search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import displace, utils, err_checks


class TestDisplace(unittest.TestCase):
    def assertDCoeffErrorBound(self, d_func):
        mean_err = err_checks.coeff_err(d_func(self.alphas), self.alphas)[0]
        self.assertLess(mean_err, 9e-7)

    def setUp(self):
        self.N = 100
        self.alphas = utils.random_alphas(100, maxval=1)

    def test_gen_displace_scipy(self):
        f = displace.gen_displace_scipy(self.N)
        self.assertDCoeffErrorBound(f)       

    def test_gen_displace(self):
        f = displace.gen_displace(self.N)
        self.assertDCoeffErrorBound(f)

    def test_gen_displace_BCH(self):
        f = displace.gen_displace_BCH(self.N)
        self.assertDCoeffErrorBound(f)

    def test_gen_displace_distribute_BCH(self):
        strategy = tf.distribute.experimental.CentralStorageStrategy(
            compute_devices=["CPU:0", "CPU:1", "CPU:2"], parameter_device="CPU:0"
        )
        f = displace.gen_displace_distribute_BCH(self.N, strategy=strategy)
        self.assertDCoeffErrorBound(f)


if __name__ == "__main__":
    unittest.main()
