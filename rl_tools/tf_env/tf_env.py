from abc import ABCMeta, abstractmethod
from math import sqrt, pi
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import complex64 as c64
from tf_agents import specs
from tf_agents.environments import tf_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import tensor_spec


class TFEnvironmentQuantumControl(tf_environment.TFEnvironment, metaclass=ABCMeta):
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
        *args,
        # Optional kwargs
        H=1,
        T=4,
        episode_length=20,
        batch_size=50,
        reward_kwargs={'reward_mode' : 'zero'},
        **kwargs):
        """
        Args:
            H (int, optional): Horizon for history returned in observations. Defaults to 1.
            T (int, optional): Periodicity of the 'clock' observation. Defaults to 4.
            episode_length (int, optional): Number of iterations in training episode. Defaults to 20.
            batch_size (int, optional): Vectorized minibatch size. Defaults to 50.
            reward_kwargs (dict, optional): optional dictionary of parameters
                for the reward function of RL agent.

        """
        # Default simulation parameters
        self.H = H
        self.T = T
        self.episode_length = episode_length
        self.batch_size = batch_size

        self.setup_reward(reward_kwargs)
        self._epoch = 0

        # Define action and observation specs
        self.control_circuit = self._control_circuit
        action_spec = self._control_circuit_spec

        observation_spec = {
            'clock' : specs.TensorSpec(shape=[self.T], dtype=tf.float32)}
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
        _, _, obs = self.control_circuit(0, action)

        # Calculate rewards
        self._elapsed_steps += 1
        self._episode_ended = (self._elapsed_steps == self.episode_length)

        # Add dummy time dimension to tensors and append them to history
        for a in action.keys():
            self.history[a].append(action[a])

        # Make observations of 'msmt' of horizon H, shape=[batch_size,H]
        # measurements are selected with hard-coded attention step.
        # Also add clock of period 'T' to observations, shape=[batch_size,T]
        observation = {}
        C = tf.one_hot([self._elapsed_steps%self.T]*self.batch_size, self.T)
        observation['clock'] = C

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
            self.history[key] = [self.history[key]]*self.H

        # Make observation of horizon H
        observation = {
            'clock' : tf.one_hot([0]*self.batch_size, self.T)}

        self._current_time_step_ = ts.restart(observation, self.batch_size)
        return self.current_time_step()

    def _current_time_step(self):
        return self._current_time_step_

    @abstractmethod
    def _control_circuit(self, psi, action):
        """
        Quantum circuit to run on every step, to be defined by the subclass.
        Input:
            psi (Tensor([batch_size,N], c64)): batch of states
            action (dict, 'alpha' : Tensor([batch_size,1,2], tf.float32),
                    'beta': Tensor([batch_size,1,2], tf.float32), etc):
                    dictionary of actions with keys representing different
                    operations (translations, rotations etc) as required by
                    the subclasses.

        Output:
            psi_final (Tensor([batch_size,N], c64)): batch of final states
            obs (Tensor([batch_size,1], float32)) measurement outcomes;
                In open-loop control problems can return a tensor of zeros.
        """
        return 0, 0, tf.ones((self.batch_size,1))


    ### REWARD FUNCTIONS

    def setup_reward(self, reward_kwargs):
        """Setup the reward function based on reward_kwargs. """
        try:
            mode = reward_kwargs.pop('reward_mode')
            assert mode in ['zero',
                            'stabilizer_remote']
            self.reward_mode = mode
        except:
            raise ValueError('reward_mode not specified or not supported.')

        if mode == 'zero':
            """
            Required reward_kwargs:
                reward_mode (str): 'zero'

            """
            self.calculate_reward = self.reward_zero


        if mode == 'stabilizer_remote':
            """
            Required reward_kwargs:
                reward_mode (str): 'stabilizer_remote'
                N_msmt (int): number of measurements per protocol
                epoch_type (str): either 'training' or 'evaluation'
                stabilizer_amplitudes (list, float): list of stabilizer dis-
                    placement amplitudes.
                stabilizer_signs (list, flaot): list of stabilizer signs
                server_socket (Socket): socket for communication
            """
            self.server_socket = reward_kwargs.pop('server_socket')
            self.calculate_reward = \
                lambda x: self.reward_stabilizer_remote(**reward_kwargs)


    def reward_stabilizer_remote(self, N_msmt, epoch_type, penalty_coeff,
                                 stabilizer_amplitudes, stabilizer_signs):
        """
        Send the action sequence to remote environment and receive rewards.
        The data received from the remote env should be sigma_z measurement
        outcomes of shape [2, N_stabilizers, N_msmt, batch_size].

        """
        # return 0 on all intermediate steps of the episode
        if self._elapsed_steps != self.episode_length:
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
                       N_msmt=N_msmt,
                       epoch_type=epoch_type,
                       epoch=self._epoch,
                       stabilizers=stabilizer_amplitudes,
                       stabilizer_signs=stabilizer_signs)

        self.server_socket.send_data(message)

        # receive sigma_z of shape [2 or 3, N_stabilizers, N_msmt, batch_size]
        # first dimension is interpreted as m1, m2, & m0 (optionally)
        msmt, done = self.server_socket.recv_data()
        msmt = tf.cast(msmt, tf.float32)
        mask = tf.where(msmt[0]==1, 1.0, 0.0) # [N_stabilizers, N_msmt, batch_size]

        m2 = msmt[1] * np.reshape(stabilizer_signs, [len(stabilizer_signs),1,1])
        # mask out trajectories where m1 != 'g'
        Z = np.ma.array(m2, mask = np.where(msmt[0]==1, 0.0, 1.0))

        # mask out trajectories where m0 != 'g'
        if msmt.shape[0] == 3:
            Z = np.ma.array(Z, mask = np.where(msmt[2]==1, 0.0, 1.0))

        z = np.mean(Z, axis=(0,1))

        # If training, include penalty for measuring m1='e'
        if epoch_type == 'training':
            z -= penalty_coeff * (1-np.mean(mask, axis=(0,1)))

        return tf.cast(z, tf.float32)


    @tf.function
    def reward_zero(self, *args):
        """
        Reward is always zero (use when not training).

        """
        return tf.zeros(self.batch_size, dtype=tf.float32)




    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, size):
        # if 'code_map' in self.__dir__():
        #     raise ValueError('Cannot change batch_size after initialization.')
        try:
            assert size>0 and isinstance(size,int)
            self._batch_size = size
        except:
            raise ValueError('Batch size should be positive integer.')
