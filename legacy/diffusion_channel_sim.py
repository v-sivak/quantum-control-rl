# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 13:32:27 2020

@author: Vladimir Sivak
"""
import tensorflow as tf
import tensorflow_probability as tfp
from simulator.utils import normalize
from tensorflow.keras.backend import batch_dot


# # Not compatible with new simulator API. 
# # Can set it up like so (in HilbertSpace __init__ method, for instance)
# if channel == 'diffusion':
#     self.mcsim = DiffusionChannelSim(self.translate)
#     def simulate(psi, time):
#         diffusion_sigma = tf.math.sqrt(diffusion_rate * time)
#         return self.mcsim.run(psi, diffusion_sigma)
#     self.simulate = tf.function(simulate)
    
class DiffusionChannelSim:
    
    def __init__(self, translate_op):
        self.translate = translate_op
    
    def run(self, psi, diffusion_sigma):
        """
        Apply random displacements to a batch of trajectories.

        Args:
            psi -- batch of state vectors; shape=[b,NH]
            diffusion_sigma (float) -- standard deviation of the Gaussian for 
                sampling of translation amplitudes
        """
        batch_size = psi.shape[0]
        P = tfp.distributions.Normal(
            loc=[[0.0]]*batch_size, scale=[[diffusion_sigma]]*batch_size)
        re_alpha = tf.cast(P.sample(), tf.complex64)
        im_alpha = tf.cast(P.sample(), tf.complex64)
        T = self.translate(re_alpha + 1j*im_alpha)
        psi = normalize(psi)
        psi = batch_dot(T, psi)
        psi = normalize(psi)
        return psi