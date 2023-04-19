import numpy as np
import tensorflow as tf
from numpy import pi, sqrt
from tf_agents import specs
from tf_agents.utils import common, nest_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.environments.tf_wrappers import TFEnvironmentBaseWrapper


class ActionWrapper(TFEnvironmentBaseWrapper):
    """
    Wrapper produces a dictionary with action components such as 'alpha',
    'beta', 'epsilon', 'phi' as dictionary keys. Some action components are
    taken from the action script provided at initialization, and some are
    taken from the input action produced by the agent. Parameter 'to_learn'
    controls which action components are to be learned. It is also possible
    to alternate between learned and scripted values with 'use_mask' flag.

    """
    def __init__(self, env, action_script, scale, to_learn, use_mask=True,
                 learn_residuals=False):
        """
        Args:
            env: GKP environmen
            action_script: module or class with attributes corresponding to
                           action components such as 'alpha', 'phi' etc
            scale: dictionary of scaling factors for action components
            to_learn: dictionary of bool values for action components
            use_mask: flag to control masking of action components
            learn_residuals (bool): flag to learn residual over the scripted
                protocol. If False, will learn actions from scratch. If True,
                will learn a residual to be added to scripted protocol.

        """
        super(ActionWrapper, self).__init__(env)

        self.scale = scale
        self.period = action_script.period # periodicity of the protocol
        self.to_learn = to_learn
        self.use_mask = use_mask
        self.mask = action_script.mask
        self.learn_residuals = learn_residuals

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
                if self.learn_residuals:
                    action[a] += common.replicate(self.script[a][i], out_shape)

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
        return self._env.step(action)
