import os
import numpy as np
import tensorflow as tf
from math import sqrt, pi
from tensorflow import keras
from tf_agents.policies import fixed_policy
from tf_agents.trajectories import policy_step
from tf_agents import specs
from tf_agents.utils import nest_utils
from tf_agents.utils import common
from tf_agents.specs import tensor_spec
from scipy.integrate import quad

from rl_tools.utils.version_helper import TFPolicy

__all__ = ['IdlePolicy', 'ScriptedPolicy']


class IdlePolicy(fixed_policy.FixedPolicy):
    """
    Do nothing policy (zero on all actuators).

    """
    def __init__(self, time_step_spec, action_spec):
        zero_action = tensor_spec.zero_spec_nest(action_spec)
        super(IdlePolicy, self).__init__(zero_action,
                                         time_step_spec, action_spec)


class ScriptedPolicy(TFPolicy):
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

        # load the script of actions and convert to tensors
        self.script = action_script.script
        for a, val in self.script.items():
            self.script[a] = tf.constant(val, dtype=tf.float32)

        # Calculate specs and call init of parent class
        action_spec = {
            a : specs.TensorSpec(shape = C.shape[1:], dtype=tf.float32)
            for a, C in self.script.items()}

        policy_state_spec = specs.TensorSpec(shape=[], dtype=tf.int32)

        super(ScriptedPolicy, self).__init__(time_step_spec, action_spec,
                                              policy_state_spec,
                                              automatic_state_reset=True)
        self._policy_info = ()

    def _action(self, time_step, policy_state, seed):
        i = policy_state[0] % self.period # position within the policy period
        out_shape = nest_utils.get_outer_shape(time_step, self._time_step_spec)
        action = {}
        for a in self.script:
            action[a] = common.replicate(self.script[a][i], out_shape)

        return policy_step.PolicyStep(action, policy_state+1, self._policy_info)

