"""
Tests for simulator.operators

Created on Sun Jul 26 20:55:36 2020

@author: Henry Liu
"""
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

    def test_position(self):
        a = qt.destroy(self.N)
        a_dag = qt.create(self.N)
        q = (a_dag + a) / np.sqrt(2)
        npt.assert_array_equal(position(self.N), tf.cast(q.full(), dtype=tf.complex64))

    def test_momentum(self):
        a = qt.destroy(self.N)
        a_dag = qt.create(self.N)
        p = 1j * (a_dag - a) / np.sqrt(2)
        npt.assert_array_equal(momentum(self.N), tf.cast(p.full(), dtype=tf.complex64))
