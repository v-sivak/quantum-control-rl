import numpy as np
import tensorflow as tf
from tf_agents import specs
from tf_agents.environments import tf_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import tensor_spec


class TFEnvironmentQuantumControl(tf_environment.TFEnvironment):
    """
    Custom environment that follows TensorFlow Agents interface and allows to
    train a reinforcement learning agent to find quantum control policies.

    This implementation heavily relies on TensorFlow to do fast computations
    in parallel on GPU by adding batch dimension to all tensors. The speedup
    over all-Qutip implementation is about x100 on NVIDIA RTX 2080Ti.

    This is the base environment class for quantum control problems which
    incorporates simulation-independet methods. The QuantumCircuit subclasses
    inherit from this base class and from a simulation class. Subclasses
    implement 'control_circuit' which is ran at each time step. RL agent's
    actions are parametrized according to the sequence of gates applied at
    each time step, as defined by 'control_circuit'.

    Environment step() method returns TimeStep tuple whose 'observation'
    attribute stores the finite-horizon history of applied actions, measurement
    outcomes and state wavefunctions. User needs to define a wrapper for the
    environment if some components of this observation are to be discarded.
    """
    def __init__(
        self,
        action_spec={},
        T=4,
        batch_size=50,
        reward_kwargs={},
        **kwargs):
        """
        Args:
            T (int, optional): Periodicity of the 'clock' observation. Defaults to 4.
            batch_size (int, optional): Vectorized minibatch size. Defaults to 50.
            reward_kwargs (dict, optional): optional dictionary of parameters
                for the reward function of RL agent.

        """
        # Default simulation parameters
        self.T = T
        self.batch_size = batch_size

        self.setup_reward(reward_kwargs)
        self._epoch = 0

        observation_spec = {
            'clock' : specs.TensorSpec(shape=[self.T], dtype=tf.float32),
            'const' : specs.TensorSpec(shape=[1], dtype=tf.float32)}
        time_step_spec = ts.time_step_spec(observation_spec)

        super().__init__(time_step_spec, action_spec, self.batch_size)


    ### STANDARD


    def _step(self, action):
        """
        Execute one time step in the environment.

        Input:
            action -- dictionary of batched actions

        Output:
            TimeStep object (see tf-agents docs)

        """
        
        # Calculate rewards
        self._elapsed_steps += 1
        self._episode_ended = (self._elapsed_steps == self.T)

        # Add dummy time dimension to tensors and append them to history

        # don't add batch to history on the last time step,
        # since this batch of actions isn't actually used
        if not self._current_time_step_.is_last().numpy().all():
            for a in action.keys():
                self.history[a].append(action[a])

        # Make observations of 'msmt' of horizon H, shape=[batch_size,H]
        # measurements are selected with hard-coded attention step.
        # Also add clock of period 'T' to observations, shape=[batch_size,T]
        observation = {}
        C = tf.one_hot([self._elapsed_steps%self.T]*self.batch_size, self.T)
        observation['clock'] = C
        observation['const'] = tf.ones(shape=[self.batch_size,1])

        reward = self.calculate_reward(action)
        self._episode_return += reward

        if self._episode_ended:
            self._epoch += 1
            self._current_time_step_ = ts.termination(observation, reward)
        else:
            self._current_time_step_ = ts.transition(observation, reward)
        return self.current_time_step()


    def _reset(self):
        """
        Reset the state of the environment to an initial state. States are
        represented as batched tensors.

        Output:
            TimeStep object (see tf-agents docs)

        """
        self.info = {} # use to cache some intermediate results

        # Bookkeeping of episode progress
        self._episode_ended = False
        self._elapsed_steps = 0
        self._episode_return = 0

        # Initialize history of horizon H with actions=0 and measurements=1
        self.history = tensor_spec.zero_spec_nest(
            self.action_spec(), outer_dims=(self.batch_size,))
        for key in self.history.keys():
            self.history[key] = [self.history[key]]

        # Make observation of horizon H
        observation = {
            'clock' : tf.one_hot([0]*self.batch_size, self.T),
            'const'   : tf.ones(shape=[self.batch_size,1])}

        self._current_time_step_ = ts.restart(observation, self.batch_size)
        return self.current_time_step()

    def _current_time_step(self):
        return self._current_time_step_


    ### REWARD FUNCTIONS

    def setup_reward(self, reward_kwargs):
        """Setup the reward function based on reward_kwargs. """
        try:
            mode = reward_kwargs.pop('reward_mode')
            assert mode in ['zero',
                            'remote']
            self.reward_mode = mode
        except:
            raise ValueError('reward_mode not specified or not supported.')

        if mode == 'remote':
            """
            Required reward_kwargs:
                reward_mode (str): 'remote'
                N_msmt (int): number of measurements per protocol
                epoch_type (str): either 'training' or 'evaluation'
                server_socket (Socket): socket for communication
            """
            self.server_socket = reward_kwargs.pop('server_socket')
            self.calculate_reward = \
                lambda x: self.reward_remote(**reward_kwargs)


    def reward_remote(self, epoch_type):
        """
        Send the action sequence to remote environment and receive rewards.
        The data received from the remote env should be Pauli measurement
        (i.e., range from -1 to 1)
        outcomes of shape [batch_size].

        """

        

        # return 0 on all intermediate steps of the episode
        if self._elapsed_steps != self.T:
            return tf.zeros(self.batch_size, dtype=tf.float32)

        action_batch = {}
        for a in self.history.keys() - ['msmt']:
            # reshape to [batch_size, T, action_dim]
            action_history = np.array(self.history[a][1:])
            action_batch[a] = np.transpose(action_history,
                            axes=[1,0]+list(range(action_history.ndim)[2:]))

        # send action sequence and metadata to remote client
        message = dict(action_batch=action_batch,
                       batch_size=self.batch_size,
                       #N_msmt=N_msmt,
                       epoch_type=epoch_type,
                       epoch=self._epoch)

        self.server_socket.send_data(message)

        # receive sigma_z of shape [batch_size]
        msmt, done = self.server_socket.recv_data()
        msmt = tf.cast(msmt, tf.float32)
        #z = np.mean(msmt, axis=0)

        return msmt #tf.cast(z, tf.float32)





    @property
    def batch_size(self):
      return self._batch_size

    @batch_size.setter
    def batch_size(self, size):
      try:
        assert size>0 and isinstance(size,int)
        self._batch_size = size
      except:
        raise ValueError('Batch size should be positive integer.')
