# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 14:36:31 2020

@author: Vladimir Sivak
"""

import tensorflow as tf
from math import pi, sqrt
from simulator.utils import measurement, tensor
from simulator.hilbert_spaces.base import HilbertSpace
from simulator import operators as ops

class OscillatorOscillator(HilbertSpace):

    def __init__(self, *args, N1=100, N2=100, channel='quantum_jumps', **kwargs):
        """
        Args:
            N1 (int, optional): Size of the first oscillator Hilbert space.
            N2 (int, optional): Size of the second oscillator Hilbert space.
            channel (str, optional): model of the error channel, either 'diffusion'
                    or 'quantum_jumps'.
        """
        self._N1 = N1
        self._N2 = N2
        self.K_osc1 = 1
        self.K_osc2 = 1
        self.T1_osc1 = 300e-6
        self.T1_osc2 = 300e-6
        super().__init__(self, *args, channel=channel, **kwargs)

    def _define_fixed_operators(self):
        N1 = self.N1
        N2 = self.N2
        
        self.I = tensor([ops.identity(N1), ops.identity(N2)])
        self.a1 = tensor([ops.destroy(N1), ops.identity(N2)])
        self.a1_dag = tensor([ops.create(N1), ops.identity(N2)])
        self.a2 = tensor([ops.identity(N1), ops.destroy(N2)])
        self.a2_dag = tensor([ops.identity(N1), ops.create(N2)])
        self.q1 = tensor([ops.position(N1), ops.identity(N2)])
        self.p1 = tensor([ops.momentum(N1), ops.identity(N2)])
        self.n1 = tensor([ops.num(N1), ops.identity(N2)])
        self.q2 = tensor([ops.identity(N1), ops.position(N2)])
        self.p2 = tensor([ops.identity(N1), ops.momentum(N2)])
        self.n2 = tensor([ops.identity(N1), ops.num(N2)])
        self.parity1 = tensor([ops.parity(N1), ops.identity(N2)])
        self.parity2 = tensor([ops.identity(N1), ops.parity(N2)])
        
        tensor_with = [None, ops.identity(N2)]
        self.translate1 = ops.TranslationOperator(N1, tensor_with=tensor_with)
        self.displace1 = lambda a: self.translate1(sqrt(2)*a)
        self.rotate1 = ops.RotationOperator(N1, tensor_with=tensor_with)

        tensor_with = [ops.identity(N1), None]
        self.translate2 = ops.TranslationOperator(N2, tensor_with=tensor_with)
        self.displace2 = lambda a: self.translate2(sqrt(2)*a)
        self.rotate2 = ops.RotationOperator(N2, tensor_with=tensor_with)


    @property
    def _hamiltonian(self):
        Kerr1 = -1 / 2 * (2 * pi) * self.K_osc1 * self.n1 * self.n1
        Kerr2 = -1 / 2 * (2 * pi) * self.K_osc2 * self.n2 * self.n2
        return Kerr1 + Kerr2

    @property
    def _collapse_operators(self):
        photon_loss1 = sqrt(1/self.T1_osc1) * self.a1
        photon_loss2 = sqrt(1/self.T1_osc2) * self.a2

        return [photon_loss1, photon_loss2]

    @property
    def N1(self):
        return self._N1

    @property
    def N2(self):
        return self._N2