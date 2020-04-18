# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 20:10:01 2020

@author: Vladimir Sivak
"""
import numpy as np
import tensorflow as tf
from tf_agents.environments.tf_wrappers import TFEnvironmentBaseWrapper
from tf_agents import specs
from numpy import pi, sqrt
from tf_agents.utils import common
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts

class FlattenObservationsWrapperTF(TFEnvironmentBaseWrapper):
    """
    This is the same wrapper as FlattenObservationsWrapper from
    tf_agents/environments/wrappers but I changed a few things to make it 
    compatible with tf environments. The code is essentially copied with 
    minor chages. See docs from that original class.
    
    """

    def __init__(self, env, observations_whitelist=None):
    
        super(FlattenObservationsWrapperTF, self).__init__(env)
        # If observations whitelist is provided:
        #  Check that the environment returns a dictionary of observations.
        #  Check that the set of whitelist keys is a found in environment keys.       
        if observations_whitelist is not None:
            if not isinstance(env.observation_spec(), dict):
                raise ValueError(
            'If you provide an observations whitelist, the current environment '
            'must return a dictionary of observations! The returned observation'
            ' spec is type %s.' % (type(env.observation_spec())))

            # Check that observation whitelist keys are valid observation keys.
            if not (set(observations_whitelist).issubset(
                env.observation_spec().keys())):
                raise ValueError(
                'The observation whitelist contains keys not found in the '
                'environment! Unknown keys: %s' % list(
                    set(observations_whitelist).difference(
                        env.observation_spec().keys())))

        # Check that all observations have the same dtype. This dtype will be 
        # used to create the flattened TensorSpec.
        env_dtypes = list(
            set([obs.dtype for obs in env.observation_spec().values()]))
        if len(env_dtypes) != 1:
            raise ValueError('Observation spec must all have the same dtypes!'
                       ' Currently found dtypes: %s' % (env_dtypes))
        inferred_spec_dtype = env_dtypes[0]

        self._observation_spec_dtype = inferred_spec_dtype
        self._observations_whitelist = observations_whitelist
        # Update the observation spec in the environment.
        observations_spec = env.observation_spec()
        if self._observations_whitelist is not None:
            observations_spec = self._filter_observations(observations_spec)

        # Compute the observation length after flattening the observation items 
        # and nested structure. Observation specs are not batched.
        observation_total_len = sum(np.prod(observation.shape) for observation 
                                    in tf.nest.flatten(observations_spec))

        # Update the observation spec as a tensor of one-dimension.
        self._flattened_observation_spec = tensor_spec.TensorSpec(
            shape=(observation_total_len,),
            dtype=self._observation_spec_dtype,
            name='packed_observations')

    def _flatten_nested_observations(self, observations, is_batched): 

        def tf_flatten(x):
            if is_batched:
                return tf.reshape(x, [x.shape[0], -1])
            return tf.reshape(x, [-1])

        flat_observations = [tf_flatten(x) for x in tf.nest.flatten(observations)]
        axis = 1 if is_batched else 0
        return tf.concat(flat_observations, axis=axis)

    def _filter_observations(self, observations):
        filter_out = set(observations.keys()).difference(
            self._observations_whitelist)
        for filter_key in filter_out:
            del observations[filter_key]
        return observations

    def _pack_and_filter_timestep_observation(self, timestep):
        observations = timestep.observation
        if self._observations_whitelist is not None:
            observations = self._filter_observations(observations)

        return ts.TimeStep(
            timestep.step_type, timestep.reward, timestep.discount,
            self._flatten_nested_observations(
                observations, is_batched=self._env.batched))

    def time_step_spec(self):
        return ts.time_step_spec(self.observation_spec())

    def observation_spec(self):
        return self._flattened_observation_spec

    def _current_time_step(self):
        return self._pack_and_filter_timestep_observation(self._env.current_time_step())

    def _reset(self):
        return self._pack_and_filter_timestep_observation(self._env.reset())
    
    def _step(self, action):
        return self._pack_and_filter_timestep_observation(self._env.step(action))




class ActionWrapper(TFEnvironmentBaseWrapper):
    """
    Agent produces a 2d or 4d action:
        action[0:2] -- Re and Im of 'alpha'
        action[2:4] -- Re and Im of 'epsilon'
    
    Wrapper rescales it and adds additional action dimensions 'beta' and 'phi' 
    from the provided script of certain periodicity. 
    
    The resulting wrapped action is a 5-vector (7-vector) compatible with 
    environment in modes 'v1','v2' ('v3').
    
    """
    def __init__(self, env, action_script, quantum_circuit_type,
                 max_amplitude=1.0):
        """
        Input:
            env -- GKP environment
            action_script -- module or class with attributes 'beta', 'phi',
                             'period' and optionally 'mask'
            quantum_circuit_type -- one of the following: 'v1', 'v2', 'v3'
                                    used to infer action dimensions.
            max_amplitude -- amplitude used for rescaling of actions
            
        """
        self.quantum_circuit_type = quantum_circuit_type
        if quantum_circuit_type in ['v1','v2']:
            self.act_dim = 2
        if quantum_circuit_type in ['v3']:
            self.act_dim = 4
        
        super(ActionWrapper, self).__init__(env)
        self._action_spec = specs.BoundedTensorSpec(
            shape=[self.act_dim], dtype=tf.float32, minimum=-1, maximum=1)
        
        self.MAX_AMPLITUDE = max_amplitude # for rescaling the actions
        self.period = action_script.period # periodicity of the protocol
        if quantum_circuit_type in ['v3']:
            self.mask = action_script.mask # for masking reward rounds
        
        # load the script of actions
        self.beta = tf.constant(action_script.beta, 
                                shape=[self.period,1], dtype=tf.complex64)
        self.phi = tf.constant(action_script.phi, shape=[self.period,1])


    def wrap(self, action):
        """
        Wrap 2d (4d) batched action into 5d (7d) actions. 
        
        """
        # step counter to follow the script of periodicity 'period'
        i = self._env._elapsed_steps
        outer_shape = tf.constant(action.shape[0], shape=(1,))
        assert action.shape[1] == self.act_dim
        
        phi = common.replicate(self.phi[i%self.period], outer_shape)
        beta = common.replicate(self.beta[i%self.period], outer_shape)
        alpha = action[:,0:2]*self.MAX_AMPLITUDE
        
        if self.quantum_circuit_type in ['v3']:
            eps = action[:,2:4]*self.MAX_AMPLITUDE*self.mask[i%self.period]
            action = tf.concat([alpha, tf.math.real(beta), tf.math.imag(beta), 
                                eps, phi], axis=1)
        if self.quantum_circuit_type in ['v1','v2']:
            action = tf.concat([alpha, tf.math.real(beta), tf.math.imag(beta), 
                                phi], axis=1)
        return action

    def action_spec(self):
        return self._action_spec

    def _step(self, action):
        return self._env.step(self.wrap(action))
    
    