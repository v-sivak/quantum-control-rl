# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:30:15 2020

@author: Vladimir Sivak
"""
import os

import numpy as np
import tensorflow as tf
from math import sqrt, pi
from tensorflow import keras
from tensorflow.math import real, imag
from tf_agents.policies import fixed_policy, tf_policy
from tf_agents.trajectories import policy_step
from tf_agents import specs
from tf_agents.utils import nest_utils
from tf_agents.utils import common
from tf_agents.specs import tensor_spec
from scipy.integrate import quad

__all__ = ['IdlePolicy', 'ScriptedPolicy', 'BayesianFeedbackPolicy',
           'SupervisedNeuralNet']


class IdlePolicy(fixed_policy.FixedPolicy):
    """
    Do nothing policy (zero on all actuators).

    """
    def __init__(self, time_step_spec, action_spec):
        zero_action = tensor_spec.zero_spec_nest(action_spec)
        super(IdlePolicy, self).__init__(zero_action,
                                         time_step_spec, action_spec)



class ScriptedPolicy(tf_policy.Base):
    """
    Policy that follows script of actions.

    Actions are parametrized according to different gates in the quantum
    circuit executed by the agent at each time step. Action components
    include 'alpha', 'beta', 'phi' and for certain type of circuit 'epsilon'.

    Policy has its own memory / clock which stores the current round number.

    """
    def __init__(self, time_step_spec, action_script):
        """
        Input:
            time_step_spec -- see tf-agents docs

            action_script -- module or class with attributes 'alpha', 'beta',
                             'epsilon', 'phi' and 'period'.
        """
        self.period = action_script.period # periodicity of the protocol
        self.script = action_script.script # load the script of actions

        # Calculate specs and call init of parent class
        self.dims_map = {'alpha' : 2, 'beta' : 2, 'epsilon' : 2,
                         'phi' : 1, 'theta' : 1}
        spec = lambda x: specs.TensorSpec(shape=[x], dtype=tf.float32)
        action_spec = {a : spec(self.dims_map[a]) for a in self.script.keys()}
        policy_state_spec = specs.TensorSpec(shape=[], dtype=tf.int32)

        for a, val in self.script.items():
            if self.dims_map[a] == 2:
                A = tf.stack([real(val), imag(val)], axis=-1)
            elif self.dims_map[a] == 1:
                A = tf.constant(val, shape=[self.period,1])
            self.script[a] = tf.cast(A, tf.float32)

        super(ScriptedPolicy, self).__init__(time_step_spec, action_spec,
                                              policy_state_spec,
                                              automatic_state_reset=True)
        self._policy_info = ()


    def _action(self, time_step, policy_state, seed):
        i = policy_state[0] % self.period # position within the policy period
        out_shape = nest_utils.get_outer_shape(time_step, self._time_step_spec)
        action = {}
        for a in self.script:
            A = common.replicate(self.script[a][i], out_shape)
            if a == 'alpha': # do Markovian feedback
                A *= time_step.observation['msmt'][:,-1,:]
                if policy_state[0] == 0: A *= 0
            action[a] = A

        return policy_step.PolicyStep(action, policy_state+1, self._policy_info)



class BayesianFeedbackPolicy(tf_policy.Base):
    """
    Policy that follows script of actions.

    Actions are parametrized according to different gates in the quantum
    circuit executed by the agent at each time step. Action components
    include 'alpha', 'beta', 'phi' and for certain type of circuit 'epsilon'.

    Policy has its own memory / clock which stores the current round number.

    """
    def __init__(self, time_step_spec, K=3, eps=0.18, sigma=0.8):
        """
        Args:
            time_step_spec: see tf-agents docs
            K (int): number of repetitions of the sharpening rounds for each
                quadrature. For Markovian protocol set K=1.
            eps (float): translation amplitude for trimming rounds
            sigma (float): std of the Gaussian prior for the error

        """
        self.K = K
        self.period = 2*K+2 # period of the whole protocol

        # translation amplitudes for the square code
        b_amp = 2*sqrt(pi)
        a_amp = sqrt(pi)

        self.script = {
            'alpha' : [a_amp+0j] + [0j, 0j]*K + [-1j*a_amp],
            'beta' : [b_amp+0j, 1j*b_amp]*K + [eps+0j, 1j*eps],
            'phi' : [pi/2]*self.period}

        # Calculate specs and call init of parent class
        self.dims_map = {'alpha' : 2, 'beta' : 2, 'phi' : 1}
        spec = lambda x: specs.TensorSpec(shape=[x], dtype=tf.float32)
        action_spec = {a : spec(self.dims_map[a]) for a in self.script.keys()}
        policy_state_spec = specs.TensorSpec(shape=[], dtype=tf.int32)

        for a, val in self.script.items():
            if self.dims_map[a] == 2:
                A = tf.stack([real(val), imag(val)], axis=-1)
            elif self.dims_map[a] == 1:
                A = tf.constant(val, shape=[self.period,1])
            self.script[a] = tf.cast(A, tf.float32)

        super(BayesianFeedbackPolicy, self).__init__(time_step_spec, action_spec,
                                              policy_state_spec,
                                              automatic_state_reset=True)
        self._policy_info = ()


        def prior(x):
            " Gaussian zero-centered prior with standard deviation 'sigma'. "
            return np.exp(-x**2/2/sigma**2)

        def posterior(x, s):
            """ Posterior distribution in the Bayesian non-adaptive scheme.
            Args:
                s (int): the total number of '+1' measurement outcomes. """
            p = (1-np.sin(x))**s * (1+np.sin(x))**(K-s) * prior(x)
            return p

        # For each 's' calculate the Bayesian estimate of the phase using
        # the formula arg( integrate{ exp(I*x)*posterior(x) dx} )
        phase_table = {}
        for s in range(K+1):
            Re = quad(lambda x: posterior(x,s)*np.cos(x), -np.inf, np.inf)[0]
            Im = quad(lambda x: posterior(x,s)*np.sin(x), -np.inf, np.inf)[0]
            phase = np.angle(Re + 1j*Im)
            phase_table[s] = phase

        def phase_func(s):
            """ Returns a phase estimate for the batched tensor 's'.
            Args:
                s (Tensor, int): tensor of shape [batch_size,1]
            """
            phase = 0
            for c in range(K+1):
                phase += tf.where(s==c, phase_table[c], 0)
            return phase

        def Bayesian_feedback(i, m):
            """ Compute Bayesian feedback given the measurement record.
            Args:
                i (int): position within the policy period.
                m (tensor, int): measurement record; shape [batch_size,K,1]
            """
            s = tf.reduce_sum(tf.where(m==1,1,0),axis=1)
            phase = tf.cast(phase_func(s), tf.complex64)
            quadrature = {2*K-1 : 1j, 2*K : -1 + 0j}
            A = phase * quadrature[int(i)] / (2*sqrt(pi))
            return tf.concat([real(A), imag(A)], axis=1)

        self.Bayesian_feedback = Bayesian_feedback


    def _action(self, time_step, policy_state, seed):
        i = policy_state[0] % self.period # position within the policy period
        out_shape = nest_utils.get_outer_shape(time_step, self._time_step_spec)
        action = {}
        for a in self.script.keys():
            A = common.replicate(self.script[a][i], out_shape)
            if a == 'alpha':
                m = time_step.observation['msmt']
                if i not in [2*self.K-1, 2*self.K]:
                    # feedback after trimming rounds is Markovian, and after
                    # intermediate sharpening rounds is simply zero.
                    A *= m[:,-1,:]
                    if policy_state[0] == 0: A *= 0
                else: # after K sharpening rounds do the Baysian feedback
                    A = self.Bayesian_feedback(i, m)
            action[a] = A

        return policy_step.PolicyStep(action, policy_state+1, self._policy_info)


class SupervisedNeuralNet(tf_policy.Base):
    """
    The neural network in this policy was trained in a supervised way to
    reproduce the results of the MarkovianPolicyV2. It was trained on finite
    chunks of history with horizon H=16 (pre-padded if necessary to fit this
    size). The environment needs to return observations with same horizon.

    """
    def __init__(self, time_step_spec):
        sim_dir = r'E:\VladGoogleDrive\Qulab\GKP\sims'
        # name = 'Benchmarking_HybridMarkovian4Rounds\supervised_lstm\lstm.hdf5'
        name = 'Benchmarking_HybridMarkovian4Rounds\supervised_dnn\dnn.hdf5'
        # name = 'Benchmarking_HybridMarkovian4Rounds\supervised_linear\shallow.hdf5'
        self.model = keras.models.load_model(os.path.join(sim_dir,name))
        action_spec = specs.TensorSpec(shape=[5], dtype=tf.float32)
        super(SupervisedNeuralNet, self).__init__(time_step_spec, action_spec)

    def _action(self, time_step, policy_state, seed):
        stacked_obs = tf.concat([time_step.observation['action'],
                          time_step.observation['msmt']], axis=2)
        action = self.model.predict(stacked_obs)
        action = tf.convert_to_tensor(action)
        return policy_step.PolicyStep(action)
