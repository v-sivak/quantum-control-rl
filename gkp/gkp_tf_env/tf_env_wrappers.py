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

from tensorflow_core._api.v2.math import real, imag

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
    Agent produces input action as real-valued tensor of shape [batch_size,?]
    
    Wrapper produces a dictionary with action components such as 'alpha', 
    'beta', 'epsilon', 'phi' as dictionary keys. Some action components are
    taken from the action script provided at initialization, and some are 
    taken from the input action produced by the agent. Parameter 'to_learn'
    controls which action components are to be learned.
    
    """
    def __init__(self, env, action_script, to_learn, max_amplitude=1.0):
        """
        Input:
            env -- GKP environment

            action_script -- module or class with attributes corresponding to
                             action components such as 'alpha', 'phi' etc

            max_amplitude -- amplitude used for rescaling of actions
            
            to_learn -- dictionary of bool values for action components 
            
        """
        # determine the size of the input action vector 
        dims_map = {'alpha' : 2, 'beta' : 2, 'epsilon' : 2, 'phi' : 1}
        self.input_dim = sum([dims_map[a] for a, C in to_learn.items() if C])
        
        super(ActionWrapper, self).__init__(env)
        self._action_spec = specs.BoundedTensorSpec(
            shape=[self.input_dim], dtype=tf.float32, minimum=-1, maximum=1)

        self.max_amplitude = max_amplitude # for rescaling the actions
        self.period = action_script.period # periodicity of the protocol
        self.to_learn = to_learn
        self.dims_map = dims_map
        
        # ordered list of action components
        action_order = ['alpha', 'beta', 'epsilon', 'phi']
        action_order = [a for a in action_order if a in to_learn.keys()]
        self.action_order = action_order
        
        # load the script of actions
        self.script = {}
        for a in to_learn.keys():
            if a in action_script.__dir__():
                a_tf = tf.constant(action_script.__getattr__(a),
                                   shape=[self.period,1], dtype=tf.complex64)
                self.script[a] = a_tf
            else:
                raise ValueError(a + ' is not in the provided action script.')


    def wrap(self, input_action):
        """
        Output:
            actions -- dictionary of batched actions. Dictionary keys are same
                       as supplied in 'to_learn'
        
        """
        # step counter to follow the script of periodicity 'period'
        i = self._env._elapsed_steps % self.period
        out_shape = tf.constant(input_action.shape[0], shape=(1,))
        assert input_action.shape[1] == self.input_dim
        
        action = {}
        for a in self.action_order:
            # if not learning: replicate scripted action     
            if not self.to_learn[a]:
                A = common.replicate(self.script[a][i], out_shape)
                if self.dims_map[a] == 2:
                    action[a] = tf.concat([real(A), imag(A)], axis=1)
                if self.dims_map[a] == 1:
                    action[a] = real(A)
            # if leanring: take a slice of input tensor in order
            else:
                action[a] = input_action[:,:self.dims_map(a)]
                action[a] *= self.max_amplitude
                input_action = input_action[:,self.dims_map(a):]
        return action

    def action_spec(self):
        return self._action_spec

    def _step(self, action):
        return self._env.step(self.wrap(action))
    
    