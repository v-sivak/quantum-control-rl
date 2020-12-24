# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 16:00:59 2020

@author: Vladimir Sivak
"""
import tensorflow as tf
import gin
from tf_agents.networks import encoding_network
from tf_agents.networks import network
from tf_agents.networks import actor_distribution_network
from tf_agents.utils import nest_utils

@gin.configurable
class ActorDistributionNetworkGKP(network.DistributionNetwork):
    """
    This subclass of network.DistributionNetwork creates a neural network with
    two heads: 
        -- one is doing the mapping 'clock', 'msmt' --> 'alpha', 'phi', etc
        -- another is doing 'clock' --> 'beta'        
    This is done to reduce the variance in the 'beta' action component by 
    eliminating its dependance on the measurement outcomes.
    
    For another implementation example, see network.actor_distribution_network
    """
    def __init__(self, 
        input_tensor_spec,
        output_tensor_spec,
        preprocessing_layers=None,
        preprocessing_combiner=None,
        fc_layer_params=(200, 100, 50),
        batch_squash=True,
        name='ActorDistributionNetworkGKP'):
        
        """Creates an instance of `ActorDistributionNetwork`.
    
        Args:
            input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing 
                the input.
            output_tensor_spec: A nest of `tensor_spec.BoundedTensorSpec` 
                representing the output.
            preprocessing_layers: A nest of `tf.keras.layers.Layer`
              representing preprocessing for the different observations.
              All of these layers must not be already built. For more details
              see the documentation of `networks.EncodingNetwork`.
            preprocessing_combiner: A keras layer that takes a flat
              list of tensors and combines them. Good options include
              `tf.keras.layers.Add` and `tf.keras.layers.Concatenate(axis=-1)`.
              This layer must not be already built. For more details see
              the documentation of `networks.EncodingNetwork`.
            fc_layer_params: Optional list of fully_connected parameters, where
              each item is the number of units in the layer.
            batch_squash: If True the outer_ranks of the observation are 
              squashed into the batch dimension. This allow encoding networks 
              to be used with observations with shape [BxTx...].
            name: A string representing name of the network.
    
        """
        # Initializer to use for the kernels of the conv and dense layers.
        kernel_initializer = tf.compat.v1.keras.initializers.glorot_uniform()
    
        # Encoder network for full input observations. Preprocessing should flatten
        # and concatenate all observation components. 
        input_encoder = encoding_network.EncodingNetwork(
            input_tensor_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            fc_layer_params=fc_layer_params,
            activation_fn=tf.keras.activations.relu,
            kernel_initializer=kernel_initializer,
            batch_squash=batch_squash,
            dtype=tf.float32)
        
        # Encoder network for 'clock' observation only. This network head is 
        # completely independent from the 'input_encoder'.
        clock_encoder = encoding_network.EncodingNetwork(
            input_tensor_spec['clock'],
            fc_layer_params=fc_layer_params,
            activation_fn=tf.keras.activations.relu,
            kernel_initializer=kernel_initializer,
            batch_squash=batch_squash,
            dtype=tf.float32)

        # Encoder network for 'const' observation only.
        const_encoder = encoding_network.EncodingNetwork(
            input_tensor_spec['const'],
            fc_layer_params=(),
            activation_fn=tf.keras.activations.relu,
            kernel_initializer=kernel_initializer,
            batch_squash=batch_squash,
            dtype=tf.float32)

        # Nest of projection networks corresponding to 'output_tensor_spec'.
        # There is a separate projection net for each action component.
        projection_networks = tf.nest.map_structure(
            actor_distribution_network._normal_projection_net, 
            output_tensor_spec)
        
        output_spec = tf.nest.map_structure(
            lambda proj_net: proj_net.output_spec, projection_networks)
    
        super(ActorDistributionNetworkGKP, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            output_spec=output_spec,
            name=name)
    
        self._input_encoder = input_encoder
        self._clock_encoder = clock_encoder
        self._const_encoder = const_encoder
        self._projection_networks = projection_networks
        self._output_tensor_spec = output_tensor_spec
        
        # list of action dimensions that are independent of the measurements
        self._clock_only_actions = ['beta','epsilon', 'eps1', 'eps2']
        self._const_actions = ['theta']
    
    @property
    def output_tensor_spec(self):
        return self._output_tensor_spec
    

    def call(self, observations, step_type, network_state, training=False):
        outer_rank = nest_utils.get_outer_rank(
            observations, self.input_tensor_spec)
        
        input_encoder_state, _ = self._input_encoder(
            observations,
            training=training)

        clock_encoder_state, _ = self._clock_encoder(
            observations['clock'],
            training=training)

        const_encoder_state, _ = self._const_encoder(
            observations['const'],
            training=training)
          
        # Apply projection networks to encoder outputs, producing a nest of
        # distributions 'output_actions'. All action components except 'beta'
        # use the output of the 'encoder', while 'beta' uses 'clock_encoder'
        output_actions = {}
        for a in self._output_tensor_spec.keys():
            if a in self._clock_only_actions:
                output_actions[a], _ = self._projection_networks[a](
                    clock_encoder_state, outer_rank, training=training)
            elif a in self._const_actions:
                output_actions[a], _ = self._projection_networks[a](
                    const_encoder_state, outer_rank, training=training)
            else:
                output_actions[a], _ = self._projection_networks[a](
                    input_encoder_state, outer_rank, training=training)

        return output_actions, network_state    
    
    