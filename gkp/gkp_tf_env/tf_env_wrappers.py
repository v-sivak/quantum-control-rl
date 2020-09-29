# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 20:10:01 2020

@author: Vladimir Sivak
"""
import numpy as np
import tensorflow as tf
from numpy import pi, sqrt
from tf_agents import specs
from tf_agents.utils import common, nest_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.environments.tf_wrappers import TFEnvironmentBaseWrapper


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
    Wrapper produces a dictionary with action components such as 'alpha',
    'beta', 'epsilon', 'phi' as dictionary keys. Some action components are
    taken from the action script provided at initialization, and some are
    taken from the input action produced by the agent. Parameter 'to_learn'
    controls which action components are to be learned. It is also possible
    to alternate between learned and scripted values with 'use_mask' flag.

    """
    def __init__(self, env, action_script, to_learn, use_mask=True):
        """
        Args:
            env: GKP environmen
            action_script: module or class with attributes corresponding to
                           action components such as 'alpha', 'phi' etc
            to_learn: dictionary of bool values for action components
            use_mask: flag to control masking of action components

        """
        super(ActionWrapper, self).__init__(env)

        self.scale = {'alpha' : 1, 'beta' : 1, 'epsilon' : 1, 'phi' : pi,
                      'theta' : 0.02}
        self.period = action_script.period # periodicity of the protocol
        self.to_learn = to_learn
        self.use_mask = use_mask
        self.mask = action_script.mask

        # load the script of actions and convert to tensors
        self.script = action_script.script
        for a, val in self.script.items():
            self.script[a] = tf.constant(val, dtype=tf.float32)

        self._action_spec = {a : specs.BoundedTensorSpec(
            shape = C.shape[1:], dtype=tf.float32, minimum=-1, maximum=1)
            for a, C in self.script.items() if self.to_learn[a]}

    def wrap(self, input_action):
        """
        Args:
            input_action (dict): nested tensor action produced by the neural
                                 net. Dictionary keys are those marked True
                                 in 'to_learn'.

        Returns:
            actions (dict): nested tensor action which includes all action
                            components expected by the GKP class.

        """
        # step counter to follow the script of periodicity 'period'
        i = self._env._elapsed_steps % self.period
        out_shape = nest_utils.get_outer_shape(input_action, self._action_spec)

        action = {}
        for a in self.to_learn.keys():
            C1 = self.use_mask and self.mask[a][i]==0
            C2 = not self.to_learn[a]
            if C1 or C2: # if not learning: replicate scripted action
                action[a] = common.replicate(self.script[a][i], out_shape)
            else: # if learning: rescale input tensor
                action[a] = input_action[a]*self.scale[a]

        return action

    def action_spec(self):
        return self._action_spec

    def _step(self, action):
        """
        Take the nested tensor 'action' produced by the neural net and wrap it
        into dictionary format expected by the environment.

        Residual feedback learning trick: multiply the neural net prediction
        of 'alpha' by the measurement outcome of the last time step. This
        ensures that the Markovian part of the feedback is present, and the
        agent can focus its efforts on learning residual part.

        """
        action = self.wrap(action)
        m = self._env.current_time_step().observation['msmt'][:,-1,:]
        action['alpha'] *= m
        return self._env.step(action)
