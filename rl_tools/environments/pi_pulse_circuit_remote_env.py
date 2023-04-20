# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 15:33:46 2023

@author: qulab
"""

import tensorflow as tf
from rl_tools.tf_env.tf_env import TFEnvironmentQuantumControl
from tf_agents import specs
#from simulator.hilbert_spaces import OscillatorQubit

class QuantumCircuit(TFEnvironmentQuantumControl):
    """
    In the open-loop control case, there are no intermediate observations.
    Therefore, control sequence can be generated fully from time step 0 to T
    and sent over to the remote client which will test this sequence and return
    the reward. This way all communication with remote environment can be lumped
    into the reward circuit.

    To ensure compatibility with the rest of the training pipeline, it is
    convenient to create this empty circuit. The rest of the code will package
    the time-steps in the correct format expected by the tf-agents library.

    """
    @property
    def _control_circuit_spec(self):
        spec = {'amplitude'  : specs.TensorSpec(shape=[1], dtype=tf.float32),
                'detune'  : specs.TensorSpec(shape=[1], dtype=tf.float32)
                }
        return spec
