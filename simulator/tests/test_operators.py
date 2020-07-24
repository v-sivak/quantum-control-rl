import unittest

import tensorflow as tf
import qutip as qt
import numpy as np
import numpy.testing as npt

from simulator.operators import *


class TestOperators(unittest.TestCase):
    def setUp(self):
        self.N = 100

    def test_identity(self):
        npt.assert_array_equal(identity(self.N), qt.identity(self.N))

    def test_num(self):
        npt.assert_array_equal(num(self.N), qt.num(self.N))

    def test_destroy(self):
        npt.assert_array_equal(
            destroy(self.N), tf.cast(qt.destroy(self.N).full(), dtype=tf.complex64)
        )

    def test_create(self):
        npt.assert_array_equal(
            create(self.N), tf.cast(qt.create(self.N).full(), dtype=tf.complex64)
        )
