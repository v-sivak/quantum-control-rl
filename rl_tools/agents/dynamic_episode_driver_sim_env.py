from tf_agents.drivers import dynamic_episode_driver
from rl_tools.utils.version_helper import TFPolicy
from rl_tools.tf_env import env_init
from rl_tools.tf_env import tf_env_wrappers as wrappers
import importlib


class PolicyPlaceholder(TFPolicy):
    pass

class DynamicEpisodeDriverSimEnv(dynamic_episode_driver.DynamicEpisodeDriver):
    """
    This driver is a simple wrapper of the standard DynamicEpisodeDriver from
    tf-agents. It initializes a simulated environment from which data will be
    collected according to the agent's policy.

    """
    def __init__(self, env_kwargs, reward_kwargs, batch_size,
                 action_script, action_scale, to_learn,
                 learn_residuals=False, remote=False):
        """
        Args:
            env_kwargs (dict): optional parameters for training environment.
            reward_kwargs (dict): optional parameters for reward function.
            batch_size (int): number of episodes collected in parallel.
            action_script (str): name of action script. Action wrapper will
                select actions from this script if they are not learned.
            action_scale (dict, str:float): dictionary mapping action dimensions
                to scaling factors. Action wrapper will rescale actions produced
                by the agent's neural net policy by these factors.
            to_learn (dict, str:bool): dictionary mapping action dimensions to
                bool flags. Specifies if the action should be learned or scripted.
            learn_residuals (bool): flag to learn residual over the scripted
                protocol. If False, will learn actions from scratch. If True,
                will learn a residual to be added to scripted protocol.
            remote (bool): flag for remote environment to close the connection
                to a client upon finishing the training.
        """
        self.remote = remote
        # Create training env and wrap it
        env = env_init(batch_size=batch_size, reward_kwargs=reward_kwargs,
                        **env_kwargs)
        env = wrappers.ActionWrapper(env, action_script, action_scale, to_learn,
                                     learn_residuals=learn_residuals)

        # create dummy placeholder policy to initialize parent class
        dummy_policy = PolicyPlaceholder(env.time_step_spec(), env.action_spec())

        super().__init__(env, dummy_policy, num_episodes=batch_size)

    def setup(self, policy, observers):
        """Setup policy and observers for the driver."""
        self._policy = policy
        self._observers = observers or []

    def finish_training(self):
        if self.remote:
            self.env.server_socket.disconnect_client()

    def observation_spec(self):
        return self.env.observation_spec()

    def action_spec(self):
        return self.env.action_spec()

    def time_step_spec(self):
        return self.env.time_step_spec()
